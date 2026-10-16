"""Barrier trial form: a figure for every trial run, then per-starter trial features.

    trial_runs(conn, fit_before="2019/20") -> DataFrame, one row per horse per trial
    add_trial_features(df, runs)           -> df with the TRIAL_FEATURES columns added

A trial figure is on the same footing as a race speed figure (hkracing.speed): one point is about one length
at 1200m, positive = faster. Times are compared on a log scale against a par for the trial course (track and
distance: the median batch winner over the seasons before `fit_before`), after taking off the day's variant
(the shrunk median of how far that day's batch winners on the track were off par), so

    trial_fig = (par + variant − 1000·log(time)) / 2

with runners more than CAP_LENGTHS behind the batch winner capped there. Trial figures sit well below race
figures (horses are rarely pushed), so features use them relative to the rest of the field.

Stewards' comments are reduced to three flags: `easy` (held together, not fully tested), `pushed` (urged,
hard ridden, weakened, tailed off) and `trouble` (slow to begin, held up for room, checked).

Per starter, using trials before the race date only:

    tr_count        log(1 + trials since the horse's last race; for a debutant, within TRIAL_WINDOW days)
    tr_recent       1 if the horse trialled within TRIAL_WINDOW days
    tr_days         log days since that trial (0 when none)
    tr_fig_rel      latest trial figure (within the window) against the field's runners that have one
    tr_best_rel     best trial figure since the last race (within the window), against the field
    tr_lbw          lengths behind in the latest trial (5 when none)
    tr_won          won the latest trial
    tr_easy, tr_pushed, tr_trouble   comment flags of the latest trial
    tr_fig_fresh    tr_fig_rel for first-up horses and debutants (trial form is all their recent form)
"""
from __future__ import annotations

import logging
import re
import sqlite3

import numpy as np
import pandas as pd

log = logging.getLogger("hkracing.trialform")

PERMILLE_PER_LENGTH = 2.0
CAP_LENGTHS = 15.0
VARIANT_PRIOR = 2.0       # batches of evidence at par that each day's variant is shrunk toward
TRIAL_WINDOW = 120        # days

TRIAL_FEATURES = ["tr_count", "tr_recent", "tr_days", "tr_fig_rel", "tr_best_rel", "tr_lbw", "tr_won",
                  "tr_easy", "tr_pushed", "tr_trouble", "tr_fig_fresh"]

_EASY = re.compile(r"under (?:a )?hold|not (?:fully )?(?:tested|asked|extended|ridden out)|conservativ|"
                   r"without being asked|untested|ridden quietly|hands and heels|in hand|eased", re.I)
_PUSHED = re.compile(r"hard ridden|urged|pushed|niggled|scrubbed|weakened|tailed off|gave ground|faded|"
                     r"laboured|struggled|one[- ]paced|no impression|outpaced|tired", re.I)
_TROUBLE = re.compile(r"slow(?:ly)? (?:to begin|away|into stride)|missed the break|blocked|checked|hampered|"
                      r"steadied|crowded|bumped|tightened|traffic|no room|awkward|jumped (?:out )?(?:left|right)",
                      re.I)


def _season(d: pd.Series) -> pd.Series:
    start = d.dt.year - (d.dt.month < 8)
    return start.astype(str) + "/" + ((start + 1) % 100).astype(str).str.zfill(2)


def trial_runs(conn: sqlite3.Connection, fit_before: str = "2019/20") -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT t.trial_id, t.trial_date, t.track, t.distance_m, t.batch_no, u.horse_id, u.place, u.lbw,
               u.time_s, u.comment
        FROM trial_runners u JOIN trials t USING (trial_id)
        WHERE u.time_s > 0""", conn)
    df["trial_date"] = pd.to_datetime(df.trial_date)
    df["course"] = df.track + "|" + df.distance_m.astype(str)
    df["y"] = 1000 * np.log(df.time_s)
    g = df.groupby("trial_id")
    df["y"] = np.minimum(df.y, g.y.transform("min") + CAP_LENGTHS * PERMILLE_PER_LENGTH *
                         1200 / df.distance_m)
    winner = df.y.where(df.place == 1)
    fit = _season(df.trial_date) < fit_before
    par = winner[fit].groupby(df.course[fit]).median()
    par = par.reindex(df.course.unique()).fillna(winner.groupby(df.course).median())
    df["par"] = df.course.map(par)

    # Day variant: shrunk median of the day's batch winners against par, per track.
    w = df[df.place == 1].assign(res=lambda x: x.y - x.par)
    day = w.groupby(["trial_date", "track"]).res.agg(["median", "size"])
    day["variant"] = day["median"] * day["size"] / (day["size"] + VARIANT_PRIOR)
    df = df.merge(day[["variant"]].reset_index(), on=["trial_date", "track"], how="left")
    df["variant"] = df.variant.fillna(0.0)
    # Per-mille at 1200m terms, so one point is about one length whatever the distance.
    df["fig"] = (df.par + df.variant - df.y) / PERMILLE_PER_LENGTH * df.distance_m / 1200

    text = df.comment.fillna("")
    df["easy"] = text.str.contains(_EASY).astype(float)
    df["pushed"] = text.str.contains(_PUSHED).astype(float)
    df["trouble"] = text.str.contains(_TROUBLE).astype(float)
    df["won"] = (df.place == 1).astype(float)
    log.info("%d trial runs, %d horses, %d trial days", len(df), df.horse_id.nunique(),
             df.trial_date.nunique())
    return df[["horse_id", "trial_date", "trial_id", "fig", "lbw", "won", "easy", "pushed", "trouble"]]


def add_trial_features(df: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """`df` needs race_id, horse_id, race_date, days (since last run, NaN for a debut), runs and first_up."""
    base = df[["horse_id", "race_date", "days"]].reset_index().rename(columns={"index": "row"})
    base["since"] = base.race_date - pd.to_timedelta(base.days, unit="D")
    m = base.merge(runs, on="horse_id")
    m = m[(m.trial_date < m.race_date) & (m.trial_date >= m.race_date - pd.Timedelta(days=TRIAL_WINDOW))]
    m["after_last"] = m.since.isna() | (m.trial_date > m.since)
    m = m.sort_values(["row", "trial_date"])
    last = m.groupby("row").tail(1).set_index("row")
    since = m[m.after_last].groupby("row").agg(n=("fig", "size"), best=("fig", "max"))

    out = df.copy()
    idx = out.index
    out["tr_count"] = np.log1p(since.n.reindex(idx).fillna(0))
    out["tr_recent"] = idx.isin(last.index).astype(float)
    gap = (out.race_date - last.trial_date.reindex(idx)).dt.days
    out["tr_days"] = np.log(gap).fillna(0.0)
    out["tr_fig"] = last.fig.reindex(idx)
    out["tr_best"] = since.best.reindex(idx)
    out["tr_lbw"] = last.lbw.reindex(idx).clip(0, 20).fillna(5.0)
    for col in ("won", "easy", "pushed", "trouble"):
        out[f"tr_{col}"] = last[col].reindex(idx).fillna(0.0)
    for col in ("tr_fig", "tr_best"):
        mean = out.groupby("race_id")[col].transform("mean")
        out[f"{col}_rel"] = (out[col] - mean).fillna(0.0)
    fresh = ((out.runs == 0) | (out.first_up > 0)).astype(float)
    out["tr_fig_fresh"] = out.tr_fig_rel * fresh
    return out
