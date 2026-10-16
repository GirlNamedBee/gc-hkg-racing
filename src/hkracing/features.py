"""Per-runner model features, each computed only from what was known before the race.

    build_features(conn) -> DataFrame, one row per starter

A starter is a runner with final win odds (withdrawn horses have none; pulled-up and fallen horses do).
Form features come from the horse's earlier runs in our results history (speed_figures must be built first,
see hkracing.speed). Jockey, trainer and draw records use only races on earlier dates.
Sectional features (finishing speed over the last 400m, race-shape-adjusted figures) come from
hkracing.sectionals, and barrier trial features from hkracing.trialform. Most features are expressed relative to the rest of the field, since the model compares
horses within a race.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .sectionals import run_measures
from .trialform import TRIAL_FEATURES, add_trial_features, trial_runs

JOCKEY_PRIOR = 10.0   # A/E shrinkage: pretend each jockey/trainer also had this many expected wins, all par
STRIKE_PRIOR = 30.0   # same for plain strike rates (wins vs 1/field)
DRAW_PRIOR = 30.0
ROLL = "730D"         # jockey/trainer window

# Feature columns fed to the win model, in report order.
FEATURES = [
    "fig_last_rel", "fig_mean3_rel", "fig_best5_rel", "fig_dist_rel", "no_fig",
    "early_rel", "early_x_draw",
    "draw_ae", "draw_frac",
    "weight_rel", "rating_rel",
    "jockey_sr", "trainer_sr", "jockey_ae", "trainer_ae",
    "log_days", "log_days2", "first_up", "debut", "log_runs", "won_last", "win_rate",
    "bw_change", "dist_change", "lbw_last", "class_drop", "surface_switch",
    "close_mean3_rel", "close_last_rel", "late_best5_rel", "fig_pa_mean3_rel",
] + TRIAL_FEATURES

SECTIONAL_FEATURES = ["close_mean3_rel", "close_last_rel", "late_best5_rel", "fig_pa_mean3_rel"]


def _load(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT u.race_id, u.horse_id, m.race_date, m.season, m.venue, r.race_no, r.class, r.distance_m,
               r.surface, r.course, u.draw, u.weight_lb, u.body_weight_lb, u.rating, u.jockey, u.trainer,
               u.finish_pos, u.lbw, u.running_pos, u.win_odds, f.fig
        FROM runners u JOIN races r USING (race_id) JOIN meetings m USING (meeting_id)
        LEFT JOIN speed_figures f USING (race_id, horse_id)
        WHERE u.win_odds > 0""", conn)
    df["race_date"] = pd.to_datetime(df.race_date)
    return df.sort_values(["race_date", "race_no", "race_id"]).reset_index(drop=True)


_CLASS_RANK = {"Group One": 0, "Group Two": 0, "Group Three": 0, "4 Year Olds": 1, "Class 1": 1, "Class 2": 2,
               "Class 3": 3, "Class 3 (Restricted)": 3, "Restricted Race": 4, "Class 4": 4,
               "Class 4 (Restricted)": 4, "Class 5": 5, "Griffin Race": 5}


def _race_basics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("race_id")
    df["field"] = g.horse_id.transform("size")
    inv = 1 / df.win_odds
    df["p_market"] = inv / inv.groupby(df.race_id).transform("sum")
    n_win = (df.finish_pos == 1).groupby(df.race_id).transform("sum")
    df["won"] = np.where(df.finish_pos == 1, 1 / n_win.clip(lower=1), 0.0)  # dead heats share the win
    first_call = pd.to_numeric(df.running_pos.str.split().str[0], errors="coerce")
    df["early"] = (first_call - 1) / (df.field - 1).clip(lower=1)  # 0 = led early, 1 = last
    df["class_rank"] = df["class"].map(_CLASS_RANK).fillna(4)
    return df


def _form(df: pd.DataFrame) -> pd.DataFrame:
    h = df.groupby("horse_id", sort=False)
    df["runs"] = h.cumcount()
    prev = lambda col, k=1: h[col].shift(k)
    df["days"] = (df.race_date - prev("race_date")).dt.days
    df["fig_last"] = prev("fig")
    figs_prev = h.fig.shift(1)
    df["fig_mean3"] = figs_prev.groupby(df.horse_id).transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["fig_best5"] = figs_prev.groupby(df.horse_id).transform(lambda s: s.rolling(5, min_periods=1).max())
    df["early_mean3"] = h.early.shift(1).groupby(df.horse_id).transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    df["bw_change"] = (df.body_weight_lb - prev("body_weight_lb")).fillna(0).clip(-40, 40)
    df["dist_change"] = ((df.distance_m - prev("distance_m")) / 100).fillna(0)
    df["lbw_last"] = prev("lbw").fillna(5).clip(0, 20)
    df["class_drop"] = (df.class_rank - prev("class_rank")).fillna(0)  # positive = dropping in class
    df["won_last"] = prev("won").fillna(0)
    df["win_rate"] = (h.won.cumsum() - df.won + 0.5) / (df.runs + 5)  # shrunk career strike rate
    df["surface_switch"] = ((df.surface != prev("surface")) & df.runs.gt(0)).astype(float)
    df["fig_dist"] = _best_at_distance(df)
    # Sectionals: closing speed against the rest of the field, last-400m figure, race-shape-adjusted figure.
    df["close_last"] = prev("close_rel")
    df["close_mean3"] = _rolling_prev(df, h, "close_rel", 3, "mean")
    df["late_best5"] = _rolling_prev(df, h, "late_fig", 5, "max")
    df["fig_pa_mean3"] = _rolling_prev(df, h, "fig_pa", 3, "mean")
    return df


def _rolling_prev(df: pd.DataFrame, h, col: str, k: int, how: str) -> pd.Series:
    """Rolling `how` of the horse's previous `k` values of `col` (today's run excluded)."""
    return h[col].shift(1).groupby(df.horse_id).transform(lambda s: getattr(s.rolling(k, min_periods=1), how)())


def _best_at_distance(df: pd.DataFrame, window: int = 8, tol: int = 200) -> pd.Series:
    """Best figure in the horse's last `window` runs within `tol` metres of today's distance."""
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("horse_id", sort=False).indices.items():
        figs = df.fig.to_numpy()[idx]
        dists = df.distance_m.to_numpy()[idx]
        for j in range(1, len(idx)):
            lo = max(0, j - window)
            ok = np.abs(dists[lo:j] - dists[j]) <= tol
            vals = figs[lo:j][ok]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                out[idx[j]] = vals.max()
    return pd.Series(out, index=df.index)


def _ae(df: pd.DataFrame, key: str | list[str], prior: float, window: str | None,
        expected: str = "p_market") -> pd.Series:
    """log((wins + prior) / (expected wins + prior)) from races on earlier dates only.

    With expected="p_market" this is wins above what the market expected (so riding favourites isn't
    skill); with "p_uniform" (1/field) it is a plain shrunk strike rate.
    """
    keys = [key] if isinstance(key, str) else key
    exp_col = expected
    daily = df.groupby(keys + ["race_date"], sort=False, dropna=False).agg(w=("won", "sum"), e=(exp_col, "sum")).reset_index()
    daily = daily.sort_values(keys + ["race_date"]).reset_index(drop=True)
    g = daily.groupby(keys, sort=False, dropna=False)
    if window:
        # Groups come out in daily's (sorted) row order, so the rolled values line up row for row.
        rolled = daily.set_index("race_date").groupby(keys, sort=False, dropna=False)[["w", "e"]] \
            .rolling(window, closed="left").sum()
        daily[["W", "E"]] = rolled.fillna(0).to_numpy()
    else:
        daily["W"] = g.w.cumsum() - daily.w
        daily["E"] = g.e.cumsum() - daily.e
    daily["ae"] = np.log((daily.W + prior) / (daily.E + prior))
    return df.merge(daily[keys + ["race_date", "ae"]], on=keys + ["race_date"], how="left")["ae"].fillna(0)


def _rel(df: pd.DataFrame, col: str) -> pd.Series:
    """Value minus the field average of the runners that have one; 0 where missing."""
    mean = df.groupby("race_id")[col].transform("mean")
    return (df[col] - mean).fillna(0)


def build_features(conn: sqlite3.Connection, fit_before: str = "2019/20") -> pd.DataFrame:
    """`fit_before`: first walk-forward test season; sectional pars are fitted on the seasons before it."""
    sec = run_measures(conn, fit_before)[["race_id", "horse_id", "close_rel", "late_fig", "fig_pa"]]
    df = _race_basics(_load(conn).merge(sec, on=["race_id", "horse_id"], how="left"))
    df = _form(df)
    df["p_uniform"] = 1 / df.field

    df["no_fig"] = df.fig_last.isna().astype(float)
    for col in ("fig_last", "fig_mean3", "fig_best5", "fig_dist", "close_mean3", "close_last", "late_best5",
                "fig_pa_mean3"):
        df[f"{col}_rel"] = _rel(df, col)
    df["early_rel"] = _rel(df, "early_mean3")
    df["draw_frac"] = ((df.draw - 1) / (df.field - 1).clip(lower=1)).clip(upper=1).fillna(0.5)
    df["early_x_draw"] = df.early_rel * (df.draw_frac - 0.5)
    df["weight_rel"] = _rel(df, "weight_lb")
    full_rating = df.rating.notna().groupby(df.race_id).transform("all")
    df["rating_rel"] = np.where(full_rating, _rel(df, "rating"), 0.0)

    df = df.reset_index(drop=True)
    # Draws are allotted at random, so a plain strike rate per draw is an unbiased draw-bias estimate.
    df["draw_ae"] = _ae(df, ["venue", "surface", "distance_m", "draw"], DRAW_PRIOR, None, "p_uniform").to_numpy()
    for who in ("jockey", "trainer"):
        df[f"{who}_ae"] = _ae(df, who, JOCKEY_PRIOR, ROLL).to_numpy()
        df[f"{who}_sr"] = _ae(df, who, STRIKE_PRIOR, ROLL, "p_uniform").to_numpy()

    df["debut"] = (df.runs == 0).astype(float)
    df["log_days"] = np.log1p(df.days.fillna(60).clip(upper=365))
    df["first_up"] = (df.days > 90).astype(float)
    df["log_days2"] = (df.log_days - 3.3) ** 2  # both quick backups and long breaks
    df["log_runs"] = np.log1p(df.runs)
    return add_trial_features(df, trial_runs(conn, fit_before))
