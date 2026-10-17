"""Trackwork form: per-starter features from the horse's morning work before the race.

    track_work(conn)                       -> DataFrame, one row per piece of work (hkracing.trackwork)
    add_trackwork_features(df, work)       -> df with the TRACKWORK_FEATURES columns added

Gallop times are only comparable on the same track with the same number of timed sections, so each gallop
is scored against those: its last section (the fastest, usually the final 400m) as a z-score against the
median and spread of last sections on that track and section count, positive = faster. The pars use times
only, never results, so fitting them on every season leaks nothing about who won.

Per starter, using work dated before the race day only:

    tw_gallops      log(1 + gallops in the last 14 days)
    tw_gallops28    log(1 + gallops in the last 28 days)
    tw_days_gallop  log days since the last gallop (capped at 60)
    tw_work_days    log(1 + days with any work in the last 28 days, races and trials aside)
    tw_swim         log(1 + swims in the last 28 days)
    tw_rehab        log(1 + treadmill and aqua-walker sessions in the last 28 days)
    tw_conghua      share of the last 28 days' work done at Conghua
    tw_jockey       1 if today's jockey rode the horse in a gallop in the last 14 days
    tw_named_rider  1 if any jockey (not the stable's track rider) rode a gallop in the last 14 days
    tw_speed_rel    best gallop z-score in the last 21 days, against the field's runners that have one
    tw_speed_trend  latest gallop z-score minus the horse's mean over the 90 days before it

HKJC only keeps trackwork for horses still in training, so older runs are partly covered. A starter whose
horse has no trackwork page gets the average of the covered runners in its race for every feature, which
leaves the within-race comparison to the covered horses (a race with none covered is unaffected). There is
deliberately no "has trackwork" flag: being covered means the horse is still racing today, which would leak
the future. `tw_covered` is kept as a column for evaluation only, never as a model feature.
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

log = logging.getLogger("hkracing.trackform")

TRACKWORK_FEATURES = ["tw_gallops", "tw_gallops28", "tw_days_gallop", "tw_work_days", "tw_swim", "tw_rehab",
                      "tw_conghua", "tw_jockey", "tw_named_rider", "tw_speed_rel", "tw_speed_trend"]

TRACK_RIDERS = {"R.B.", "A.T.", "Trainer"}   # riding-boy, assistant trainer: not a race jockey
MIN_PAR_GALLOPS = 30


def track_work(conn: sqlite3.Connection) -> pd.DataFrame:
    w = pd.read_sql("SELECT horse_id, work_date, work_type, track, splits, time_s, rider FROM trackwork", conn)
    w["work_date"] = pd.to_datetime(w.work_date)
    w["covered"] = True
    gal = w.work_type.eq("Gallop") & w.splits.notna()
    parts = w.splits.where(gal).str.split()
    w["n_sec"] = parts.str.len()
    w["last_sec"] = pd.to_numeric(parts.str[-1], errors="coerce")
    key = w.track + "|" + w.n_sec.astype("Int64").astype(str)
    g = w[gal].groupby(key[gal]).last_sec
    med, n = g.transform("median"), g.transform("size")
    mad = g.transform(lambda s: (s - s.median()).abs().median() * 1.4826)
    z = (med - w.last_sec[gal]) / mad.clip(lower=0.3)
    w["speed_z"] = z.where(n >= MIN_PAR_GALLOPS).clip(-4, 4)
    log.info("%d trackwork rows, %d horses, %d gallops (%d scored)", len(w), w.horse_id.nunique(), gal.sum(),
             w.speed_z.notna().sum())
    return w


def _count(dates: np.ndarray, race_dates: np.ndarray, days: int) -> np.ndarray:
    """Number of `dates` (sorted) in [race_date - days, race_date) for each race date."""
    hi = np.searchsorted(dates, race_dates, side="left")
    lo = np.searchsorted(dates, race_dates - np.timedelta64(days, "D"), side="left")
    return hi - lo


def add_trackwork_features(df: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
    """`df` needs race_id, horse_id, race_date and jockey."""
    out = df.copy()
    n = len(out)
    feats = {k: np.full(n, np.nan) for k in ("gallops", "gallops28", "days_gallop", "work_days", "swim", "rehab",
                                             "conghua", "jockey", "named", "speed_best", "speed_trend")}
    covered = np.zeros(n, dtype=bool)
    rows_by_horse = out.groupby("horse_id", sort=False).indices
    jockeys = out.jockey.to_numpy()
    race_dates = out.race_date.to_numpy().astype("datetime64[D]")
    for horse_id, w in work.groupby("horse_id", sort=False):
        idx = rows_by_horse.get(horse_id)
        if idx is None:
            continue
        w = w.sort_values("work_date")
        rd = race_dates[idx]
        covered[idx] = True
        wd = w.work_date.to_numpy().astype("datetime64[D]")
        typ = w.work_type.to_numpy()
        gal = typ == "Gallop"
        gd = wd[gal]
        feats["gallops"][idx] = _count(gd, rd, 14)
        feats["gallops28"][idx] = _count(gd, rd, 28)
        last = np.searchsorted(gd, rd, side="left") - 1
        gap = np.where(last >= 0, (rd - gd[np.maximum(last, 0)]).astype(float), 60.0)
        feats["days_gallop"][idx] = np.minimum(gap, 60.0)
        exercise = ~np.isin(typ, ["Race", "Barrier Trial"])
        days = np.unique(wd[exercise])
        feats["work_days"][idx] = _count(days, rd, 28)
        feats["swim"][idx] = _count(wd[typ == "Swimming"], rd, 28)
        feats["rehab"][idx] = _count(wd[np.isin(typ, ["Equine Treadmill", "Aqua Walker"])], rd, 28)
        ch = exercise & w.track.fillna("").str.startswith("Conghua").to_numpy()
        tot = _count(wd[exercise], rd, 28)
        feats["conghua"][idx] = np.where(tot > 0, _count(wd[ch], rd, 28) / np.maximum(tot, 1), 0.0)

        riders = w.rider.to_numpy()[gal]
        z = w.speed_z.to_numpy()[gal]
        for k, i in enumerate(idx):
            r = rd[k]
            lo14 = np.searchsorted(gd, r - np.timedelta64(14, "D"))
            hi = np.searchsorted(gd, r)
            recent = riders[lo14:hi]
            feats["jockey"][i] = float(jockeys[i] is not None and jockeys[i] in set(recent))
            feats["named"][i] = float(any(x and x not in TRACK_RIDERS for x in recent))
            lo21 = np.searchsorted(gd, r - np.timedelta64(21, "D"))
            zz = z[lo21:hi]
            zz = zz[~np.isnan(zz)]
            if len(zz):
                feats["speed_best"][i] = zz.max()
            if hi > 0 and not np.isnan(z[hi - 1]) and gd[hi - 1] >= r - np.timedelta64(21, "D"):
                lo90 = np.searchsorted(gd, gd[hi - 1] - np.timedelta64(90, "D"))
                prev = z[lo90:hi - 1]
                prev = prev[~np.isnan(prev)]
                if len(prev):
                    feats["speed_trend"][i] = z[hi - 1] - prev.mean()

    out["tw_covered"] = covered
    out["tw_gallops"] = np.log1p(feats["gallops"])
    out["tw_gallops28"] = np.log1p(feats["gallops28"])
    out["tw_days_gallop"] = np.log1p(feats["days_gallop"])
    out["tw_work_days"] = np.log1p(feats["work_days"])
    out["tw_swim"] = np.log1p(feats["swim"])
    out["tw_rehab"] = np.log1p(feats["rehab"])
    out["tw_conghua"] = feats["conghua"]
    out["tw_jockey"] = feats["jockey"]
    out["tw_named_rider"] = feats["named"]
    # Speed: relative to covered runners with a scored gallop; 0 (field average) otherwise.
    best = pd.Series(feats["speed_best"], index=out.index)
    out["tw_speed_rel"] = (best - best.groupby(out.race_id).transform("mean")).fillna(0.0)
    out["tw_speed_trend"] = np.nan_to_num(feats["speed_trend"])
    # Uncovered starters take their race's covered average, so they don't differ from it.
    for col in TRACKWORK_FEATURES:
        if col in ("tw_speed_rel", "tw_speed_trend"):
            continue
        s = out[col].where(out.tw_covered)
        fill = s.groupby(out.race_id).transform("mean").fillna(0.0)
        out[col] = s.fillna(fill)
    out["tw_speed_trend"] = out.tw_speed_trend.where(out.tw_covered, 0.0)
    log.info("trackwork coverage: %.1f%% of starters", 100 * covered.mean())
    return out
