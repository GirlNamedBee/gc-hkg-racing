"""Speed figures: each runner's finishing time against par, corrected by a daily track variant.

    python -m hkracing.speed            # (re)build speed_figures, track_variants and par_times

Winning times are modelled on a log scale, in per-mille (1 = 0.1% of the race time):

    1000·log(winning time) = course par[venue, surface, distance] + class offset[class] + variant[day] + noise

where "day" is one meeting's races on one surface and going. Pars and class offsets start from the HKJC
standard times (standard_times table) and are refined from our own history, which also covers distances and
classes HKJC leaves blank. The variant is the day's shrunk average of how much slower than par its winners
ran, so a fast or slow track doesn't read as a good or bad horse.

A runner's figure is then how much faster than the course's Class 4 par it ran, after the variant:

    figure = BASE + (par[venue, surface, distance] + variant − 1000·log(finish time)) / PERMILLE_PER_LENGTH

PERMILLE_PER_LENGTH = 2.0 means one point is about one length at 1200m (a length is ~2.4m), and
BASE = 80 puts a typical Class 4 winner at 80. Class offsets are only used to estimate the variant; they are
not taken off the figure, so a Class 1 winner earns a higher figure than a Class 4 winner on the same day.
Finishing times beaten more than CAP_LENGTHS behind the winner are capped there (eased horses).

Pars and class offsets are fitted on all seasons at once, so a figure in an early season uses a par that
saw later races. The leak is tiny (a par is an average of hundreds of races) and is accepted for now.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3

import numpy as np
import pandas as pd

from .db import connect

log = logging.getLogger("hkracing.speed")

BASE = 80.0
PERMILLE_PER_LENGTH = 2.0
CAP_LENGTHS = 15.0
HUBER = 15.0          # per-mille; residuals beyond this (≈1s at 1200m) are down-weighted
PRIOR_RACES = 30.0    # how many races of evidence the HKJC standard counts for
VARIANT_PRIOR = 1.5   # shrink each day's variant toward 0 as if it had this many par races
ITERATIONS = 25

CLASS_KEY = {"Group One": "Group", "Group Two": "Group", "Group Three": "Group", "Griffin Race": "Griffin",
             "Class 3 (Restricted)": "Class 3", "Class 4 (Restricted)": "Class 4", "Restricted Race": "Class 4",
             "4 Year Olds": "Class 2"}
REF_CLASS = "Class 4"

SCHEMA = """
CREATE TABLE IF NOT EXISTS par_times (
    venue TEXT, surface TEXT, distance_m INTEGER, class TEXT,
    par_s REAL,              -- fitted winning-time par
    hkjc_s REAL,             -- HKJC standard time, where published
    n_races INTEGER,
    PRIMARY KEY (venue, surface, distance_m, class)
);
CREATE TABLE IF NOT EXISTS track_variants (
    meeting_id TEXT, surface TEXT, going TEXT,
    variant REAL,            -- per-mille slower than par (positive = slow track)
    variant_s_1200 REAL,     -- the same as seconds over a 1200m race
    n_races INTEGER,
    PRIMARY KEY (meeting_id, surface, going)
);
CREATE TABLE IF NOT EXISTS speed_figures (
    race_id TEXT, horse_id TEXT,
    fig REAL,                -- capped at CAP_LENGTHS behind the winner
    fig_raw REAL,
    PRIMARY KEY (race_id, horse_id)
);
"""


def class_key(cls: str | None) -> str:
    return CLASS_KEY.get(cls, cls or REF_CLASS)


def load(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    races = pd.read_sql("""
        SELECT r.race_id, r.meeting_id, m.race_date, m.venue, m.season, r.class, r.distance_m, r.surface,
               r.going, r.winning_time_s
        FROM races r JOIN meetings m USING (meeting_id)
        WHERE r.winning_time_s > 0""", conn)
    runners = pd.read_sql("""
        SELECT race_id, horse_id, finish_time_s, lbw FROM runners
        WHERE finish_pos IS NOT NULL AND finish_time_s > 0""", conn)
    std = pd.read_sql("SELECT venue, surface, distance_m, class, time_s, season FROM standard_times", conn)
    return races, runners, std


def _huber_w(r: pd.Series) -> pd.Series:
    return (HUBER / r.abs().clip(lower=HUBER)).astype(float)


def fit_pars(races: pd.DataFrame, std: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Fit course pars, class offsets and day variants. Returns (races with columns, mu, b, variant)."""
    races = races.copy()
    races["ckey"] = races["class"].map(class_key)
    races["course"] = races.venue + "|" + races.surface + "|" + races.distance_m.astype(str)
    races["day"] = races.meeting_id + "|" + races.surface + "|" + races.going.fillna("")
    races["y"] = 1000 * np.log(races.winning_time_s)

    # Priors from the latest HKJC standards: course par = Class 4 standard, class offset = average gap to it.
    std = std[std.season == std.season.max()] if len(std) else std
    std = std.assign(course=std.venue + "|" + std.surface + "|" + std.distance_m.astype(str),
                     y=1000 * np.log(std.time_s))
    ref = std[std["class"] == REF_CLASS].set_index("course").y
    gaps = std.join(ref.rename("ref"), on="course").dropna(subset=["ref"])
    b_prior = (gaps.y - gaps.ref).groupby(gaps["class"]).mean()
    b_prior[REF_CLASS] = 0.0
    mu_prior = ref.copy()
    for course, grp in std.groupby("course"):  # courses with no Class 4 standard (e.g. ST 2400)
        if course not in mu_prior.index:
            mu_prior[course] = (grp.y - grp["class"].map(b_prior).fillna(0)).mean()

    courses = races.course.unique()
    classes = races.ckey.unique()
    mu_w = pd.Series([PRIOR_RACES if c in mu_prior.index else 0.0 for c in courses], index=courses)
    mu0 = pd.Series([mu_prior.get(c, np.nan) for c in courses], index=courses)
    mu0 = mu0.fillna(races.groupby("course").y.median())
    b_w = pd.Series([PRIOR_RACES if c in b_prior.index else 0.0 for c in classes], index=classes)
    b0 = pd.Series([b_prior.get(c, 0.0) for c in classes], index=classes)

    mu, b = mu0.copy(), b0.copy()
    v = pd.Series(0.0, index=races.day.unique())
    for _ in range(ITERATIONS):
        def resid(exclude):
            parts = {"mu": races.course.map(mu), "b": races.ckey.map(b), "v": races.day.map(v)}
            return races.y - sum(p for k, p in parts.items() if k != exclude)

        r = resid("v")
        w = _huber_w(r - races.day.map(v))
        v = (r * w).groupby(races.day).sum() / (w.groupby(races.day).sum() + VARIANT_PRIOR)
        r = resid("mu")
        w = _huber_w(r - races.course.map(mu))
        mu = ((r * w).groupby(races.course).sum() + mu0 * mu_w) / (w.groupby(races.course).sum() + mu_w)
        r = resid("b")
        w = _huber_w(r - races.ckey.map(b))
        b = ((r * w).groupby(races.ckey).sum() + b0 * b_w) / (w.groupby(races.ckey).sum() + b_w)
        b = b - b.get(REF_CLASS, 0.0)  # keep Class 4 as the zero point
    races["mu"] = races.course.map(mu)
    races["variant"] = races.day.map(v)
    return races, mu, b, v


def figures(races: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    df = runners.merge(races[["race_id", "mu", "variant", "winning_time_s"]], on="race_id")
    fast = df.mu + df.variant - 1000 * np.log(df.finish_time_s)
    df["fig_raw"] = BASE + fast / PERMILLE_PER_LENGTH
    win = BASE + (df.mu + df.variant - 1000 * np.log(df.winning_time_s)) / PERMILLE_PER_LENGTH
    df["fig"] = np.maximum(df.fig_raw, win - CAP_LENGTHS)
    return df[["race_id", "horse_id", "fig", "fig_raw"]]


def build(conn: sqlite3.Connection) -> pd.DataFrame:
    conn.executescript(SCHEMA)
    races, runners, std = load(conn)
    races, mu, b, v = fit_pars(races, std)
    figs = figures(races, runners)

    std_now = std[std.season == std.season.max()] if len(std) else std
    hkjc = {(r.venue, r.surface, r.distance_m, r["class"]): r.time_s for _, r in std_now.iterrows()}
    counts = races.groupby(["course", "ckey"]).size()
    pars = []
    for course in mu.index:
        venue, surface, dist = course.split("|")
        for cls in b.index:
            key = (venue, surface, int(dist), cls)
            n = int(counts.get((course, cls), 0))
            if n or key in hkjc:
                pars.append((*key, round(float(np.exp((mu[course] + b[cls]) / 1000)), 2), hkjc.get(key), n))
    days = races.groupby("day").agg(meeting_id=("meeting_id", "first"), surface=("surface", "first"),
                                    going=("going", "first"), n=("race_id", "size"))
    days["variant"] = v.reindex(days.index)

    conn.execute("DELETE FROM par_times")
    conn.execute("DELETE FROM track_variants")
    conn.execute("DELETE FROM speed_figures")
    conn.executemany("INSERT INTO par_times VALUES (?,?,?,?,?,?,?)", pars)
    conn.executemany("INSERT INTO track_variants VALUES (?,?,?,?,?,?)",
                     [(d.meeting_id, d.surface, d.going, round(d.variant, 3),
                       round(69.0 * d.variant / 1000, 3), int(d.n)) for d in days.itertuples()])
    conn.executemany("INSERT INTO speed_figures VALUES (?,?,?,?)",
                     figs.round({"fig": 2, "fig_raw": 2}).itertuples(index=False, name=None))
    conn.commit()
    log.info("%d figures, %d track variants (sd %.2fs at 1200m), %d pars", len(figs), len(days),
             69.0 * days.variant.std() / 1000, len(pars))
    return figs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/hkracing.db")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(connect(args.db))


if __name__ == "__main__":
    main()
