"""Sectional-time measures for each past run: finishing speed, early speed and race shape.

    run_measures(conn, fit_before="2019/20") -> DataFrame, one row per runner with sectional times

HKJC times every runner over each section; the last section is always the final 400m, so for each run:

    l400      the horse's own time for the last 400m
    early_s   its time to the 400m mark (finish time − l400)

Times are compared on a log scale after taking off the day's track variant (track_variants, from
hkracing.speed), against a par for the course (venue, surface, distance): the median winner. All
figures here are in lengths (2.4m) at that section's speed, positive = faster than par.

    late_fig    last-400m speed vs the course's winning par
    early_fig   speed to the 400m mark vs par
    close_rel   last-400m speed vs the median finisher in the same race (no par or variant needed)
    race_shape  race shape: how much faster the first four finishers came home than the leader went
                early, against the course's usual shape. High = a slow early pace and a sprint home,
                low = a fast early pace that tired the leaders.
    fig_pa      the horse's speed figure with the effect of race shape on its running position taken off:
                in a slow-run race horses near the lead are flattered and those held up behind are
                not, and the other way round when the pace is hot.

Pars, course shapes and the race-shape coefficient are fitted only on seasons before `fit_before`, so
measures in the walk-forward test seasons use nothing learned from those seasons.
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

log = logging.getLogger("hkracing.sectionals")

LENGTH_M = 2.4
CLOSE_CAP = 8.0       # lengths; eased horses otherwise dominate the averages
LATE_CAP = 12.0


def _load(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("""
        WITH n AS (SELECT race_id, MAX(section_no) AS nsec FROM sectionals GROUP BY race_id)
        SELECT s.race_id, s.horse_id, r.meeting_id, m.season, m.venue, r.surface, r.distance_m, r.going,
               u.finish_pos, u.finish_time_s, u.running_pos, s.time_s AS l400, f.fig, v.variant,
               (SELECT COUNT(*) FROM runners x WHERE x.race_id = s.race_id AND x.finish_pos IS NOT NULL)
                   AS finishers
        FROM sectionals s JOIN n ON s.race_id = n.race_id AND s.section_no = n.nsec
        JOIN runners u ON u.race_id = s.race_id AND u.horse_id = s.horse_id
        JOIN races r ON r.race_id = s.race_id JOIN meetings m USING (meeting_id)
        LEFT JOIN speed_figures f ON f.race_id = s.race_id AND f.horse_id = s.horse_id
        LEFT JOIN track_variants v ON v.meeting_id = r.meeting_id AND v.surface = r.surface
                                   AND v.going = r.going
        WHERE u.finish_pos IS NOT NULL AND u.finish_time_s > 0 AND s.time_s BETWEEN 18 AND 40""", conn)
    df["variant"] = df.variant.fillna(0.0)
    df["early_s"] = df.finish_time_s - df.l400
    df["course"] = df.venue + "|" + df.surface + "|" + df.distance_m.astype(str)
    df["y_late"] = 1000 * np.log(df.l400) - df.variant
    df["y_early"] = 1000 * np.log(df.early_s) - df.variant
    first_call = pd.to_numeric(df.running_pos.str.split().str[0], errors="coerce")
    df["pos_early"] = ((first_call - 1) / (df.finishers - 1).clip(lower=1)).clip(0, 1)
    return df


def _course_par(df: pd.DataFrame, col: str, fit: pd.Series) -> pd.Series:
    """Median of `col` per course over the fit rows, falling back to all rows for courses fit lacks."""
    par = df[fit].groupby("course")[col].median()
    par = par.reindex(df.course.unique()).fillna(df.groupby("course")[col].median())
    return df.course.map(par)


def run_measures(conn: sqlite3.Connection, fit_before: str = "2019/20") -> pd.DataFrame:
    df = _load(conn)
    fit = df.season < fit_before
    won = df.finish_pos == 1

    late_len = 400 / LENGTH_M / 1000          # lengths per per-mille over 400m
    early_len = (df.distance_m - 400) / LENGTH_M / 1000
    df["late_fig"] = ((_course_par(df.assign(y=df.y_late.where(won)), "y", fit) - df.y_late) * late_len) \
        .clip(-LATE_CAP, LATE_CAP)
    df["early_fig"] = ((_course_par(df.assign(y=df.y_early.where(won)), "y", fit) - df.y_early) * early_len) \
        .clip(-LATE_CAP, LATE_CAP)

    g = df.groupby("race_id")
    df["close_rel"] = ((g.y_late.transform("median") - df.y_late) * late_len).clip(-CLOSE_CAP, CLOSE_CAP)

    # Race shape: first four finishers' late figure minus the early figure of the horse at the 400m mark
    # first (lowest early time), against the course's typical shape.
    top4 = df.late_fig.where(df.finish_pos <= 4).groupby(df.race_id).transform("mean")
    lead_early = g.early_fig.transform("max")
    df["shape_raw"] = top4 - lead_early
    df["race_shape"] = df.shape_raw - _course_par(df, "shape_raw", fit)

    # How much race shape moves a horse's figure through its early position, from a within-race fit of
    # fig on early position and early position × shape. Only the interaction is taken off: the plain
    # position effect is mostly better horses getting closer to the lead.
    fig_rel = df.fig - g.fig.transform("mean")
    pos = df.pos_early - 0.5
    X = np.c_[pos, pos * df.race_shape]
    ok = fit & fig_rel.notna() & df.pos_early.notna() & df.race_shape.notna()
    coef, *_ = np.linalg.lstsq(X[ok.to_numpy()], fig_rel[ok].to_numpy(), rcond=None)
    df["shape_effect"] = (coef[1] * pos * df.race_shape).fillna(0.0)
    df["fig_pa"] = df.fig - df.shape_effect
    log.info("%d runs with sectionals; position x shape coefficient %.3f lengths per length of shape",
             len(df), coef[1])
    return df[["race_id", "horse_id", "l400", "late_fig", "early_fig", "close_rel", "race_shape", "shape_effect",
               "fig_pa"]]
