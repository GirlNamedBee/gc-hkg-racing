"""Course preference: how a horse handles Happy Valley against Sha Tin, and the level of its form at each.

    venue_offset(df, fit_before)    -> HV figure minus ST figure for the same horse, in points
    add_course_features(df, delta)  -> df with the COURSE_FEATURES columns added

Happy Valley form is usually a lower level: HV has no Group races and few Class 1 or 2 races, and the best
horses in each class tend to race at Sha Tin. Speed figures share one set of class offsets across both
courses, so an HV figure overstates the horse by about a length. `venue_offset` measures this from horses
switching course between consecutive turf runs (ST to HV figures go up, HV to ST they go down; half the
difference of the two mean changes cancels the horse's own drift) on seasons before `fit_before` only.
The aptitude features below use figures with the offset taken off HV runs.

Aptitude compares HV with Sha Tin turf and is 0 in all-weather races (turf vs AWT is the surface features'
job):
    log_venue_runs     log(1 + earlier turf runs at today's course)
    first_venue        first turf run at today's course, after at least one run elsewhere
    fig_venue_best5_rel  best adjusted figure in the last 5 runs at today's course, against the field
    venue_pref         mean adjusted figure at today's course minus at the other, shrunk toward 0 when
                       either side has few runs
    venue_ae           wins against the market's expectation (sum of p_market) at today's course minus the
                       same at the other, both shrunk; the market expectation already allows for class

Level and switches (every race):
    hv_form_at_st      share of the last 3 runs at HV, when today is at Sha Tin (stepping up a level)
    st_form_at_hv      share of the last 3 runs at Sha Tin, when today is at HV
    venue_switch       last run was at the other course
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COURSE_FEATURES = ["log_venue_runs", "first_venue", "fig_venue_best5_rel", "venue_pref", "venue_ae",
                   "hv_form_at_st", "st_form_at_hv", "venue_switch"]

VENUE_PRIOR = 3.0   # venue_pref shrinkage, in runs (harmonic mean of the runs at each course)
AE_PRIOR = 1.0      # venue_ae shrinkage, in expected wins


def venue_offset(df: pd.DataFrame, fit_before: str) -> float:
    """HV minus ST figure for the same horse from consecutive turf runs at different courses."""
    t = df[(df.surface == "TURF") & (df.season < fit_before)].sort_values(["horse_id", "race_date"])
    g = t.groupby("horse_id", sort=False)
    d = t.fig - g.fig.shift()
    prev = g.venue.shift()
    up = d[(prev == "ST") & (t.venue == "HV")].mean()     # ST -> HV
    down = d[(prev == "HV") & (t.venue == "ST")].mean()   # HV -> ST
    return float((up - down) / 2)


def _course_form(df: pd.DataFrame, delta: float, best_of: int = 5) -> dict[str, np.ndarray]:
    n = len(df)
    out = {k: np.zeros(n) for k in ("venue_runs", "venue_pref", "venue_ae")}
    out["fig_venue_best5"] = np.full(n, np.nan)
    fig = (df.fig - np.where(df.venue == "HV", delta, 0.0)).to_numpy()
    venue, turf = df.venue.to_numpy(), (df.surface == "TURF").to_numpy()
    won, pm = df.won.to_numpy(), df.p_market.to_numpy()
    for _, idx in df.groupby("horse_id", sort=False).indices.items():
        idx = idx[turf[idx]]
        v, f, w, p = venue[idx], fig[idx], won[idx], pm[idx]
        for j in range(1, len(idx)):
            same = v[:j] == v[j]
            out["venue_runs"][idx[j]] = same.sum()
            f_same, f_other = f[:j][same], f[:j][~same]
            f_same, f_other = f_same[~np.isnan(f_same)], f_other[~np.isnan(f_other)]
            if len(f_same):
                out["fig_venue_best5"][idx[j]] = f_same[-best_of:].max()
            if len(f_same) and len(f_other):
                h = len(f_same) * len(f_other) / (len(f_same) + len(f_other))
                out["venue_pref"][idx[j]] = (f_same.mean() - f_other.mean()) * h / (h + VENUE_PRIOR)
            if same.any() and (~same).any():
                ae = lambda m: np.log((w[:j][m].sum() + AE_PRIOR) / (p[:j][m].sum() + AE_PRIOR))
                out["venue_ae"][idx[j]] = ae(same) - ae(~same)
    return out


def add_course_features(df: pd.DataFrame, delta: float) -> pd.DataFrame:
    """`df` must be in date order with runs, won and p_market (see hkracing.features)."""
    form = _course_form(df, delta)
    turf = df.surface == "TURF"
    df["log_venue_runs"] = np.log1p(form["venue_runs"])
    df["first_venue"] = ((form["venue_runs"] == 0) & turf & df.runs.gt(0)).astype(float)
    df["fig_venue_best5"] = form["fig_venue_best5"]
    best_mean = df.groupby("race_id").fig_venue_best5.transform("mean")
    df["fig_venue_best5_rel"] = (df.fig_venue_best5 - best_mean).fillna(0)
    df["venue_pref"] = form["venue_pref"]
    df["venue_ae"] = form["venue_ae"]

    h = df.groupby("horse_id", sort=False)
    at_hv = (df.venue == "HV").astype(float)
    hv_share3 = at_hv.groupby(df.horse_id).transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    has_runs = df.runs.gt(0)
    df["hv_form_at_st"] = np.where((df.venue == "ST") & has_runs, hv_share3, 0.0)
    df["st_form_at_hv"] = np.where((df.venue == "HV") & has_runs, 1 - hv_share3, 0.0)
    df["venue_switch"] = ((df.venue != h.venue.shift()) & has_runs).astype(float)
    return df
