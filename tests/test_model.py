import gzip
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hkracing.db import connect
from hkracing.model import ConditionalLogit, race_log_loss
from hkracing.speed import build as build_speed
from hkracing.standards import parse_standard_time, parse_standards, store_standards

PAGES = Path(__file__).parent / "pages"


def test_parse_standards():
    season, rows = parse_standards(gzip.decompress((PAGES / "standard_times_2026-08.html.gz").read_bytes()).decode())
    assert season == "2026/27"
    assert len(rows) == 71
    std = {(r["venue"], r["surface"], r["distance_m"], r["class"]): r["time_s"] for r in rows}
    assert std[("ST", "TURF", 1200, "Class 4")] == 69.25
    assert std[("ST", "TURF", 2400, "Group")] == 147.10
    assert std[("ST", "AWT", 1650, "Class 1")] == 97.80
    assert ("ST", "TURF", 1800, "Class 1") not in std  # "-" on the page
    assert parse_standard_time("0.55.95") == 55.95 and parse_standard_time("-") is None


def _toy_db(slow_day_factor=1.01):
    """Two Sha Tin meetings, 6 races each; the second day's times are all 1% slower."""
    conn = connect(":memory:")
    store_standards(conn, "2026/27", [{"venue": "ST", "surface": "TURF", "distance_m": 1200,
                                       "class": "Class 4", "time_s": 70.0}])
    for d, (date, factor) in enumerate([("2025-09-07", 1.0), ("2025-09-14", slow_day_factor)]):
        mid = f"{date}_ST"
        conn.execute("INSERT INTO meetings VALUES (?, ?, 'ST', '2025/26', 6)", (mid, date))
        for n in range(1, 7):
            rid = f"{mid}_R{n:02d}"
            conn.execute("INSERT INTO races (race_id, meeting_id, race_no, class, distance_m, surface, going, "
                         "winning_time_s) VALUES (?, ?, ?, 'Class 4', 1200, 'TURF', 'GOOD', ?)",
                         (rid, mid, n, 70.0 * factor))
            for k in range(3):
                hid = f"H{d}{n}{k}"
                conn.execute("INSERT INTO horses (horse_id) VALUES (?)", (hid,))
                conn.execute("INSERT INTO runners (race_id, horse_id, finish_pos, finish_time_s, win_odds) "
                             "VALUES (?, ?, ?, ?, 3.0)", (rid, hid, k + 1, (70.0 + 0.2 * k) * factor))
    return conn


def test_speed_figures_remove_track_variant():
    conn = _toy_db()
    build_speed(conn)
    variants = dict(conn.execute("SELECT meeting_id, variant FROM track_variants"))
    assert variants["2025-09-07_ST"] < 1 < variants["2025-09-14_ST"]  # per-mille; slow day ≈ +10, shrunk
    figs = dict(conn.execute("SELECT horse_id, fig FROM speed_figures"))
    # Same-placed horses on the fast and slow day: 1% in time is 5 points unadjusted; the variant (shrunk
    # toward 0 with only 6 races a day) takes most of it out.
    assert abs(figs["H010"] - figs["H110"]) < 1.5
    assert figs["H010"] > figs["H011"] > figs["H012"]
    # A 2-length (0.4s) gap at 1200m is about 3 points with PERMILLE_PER_LENGTH = 2.
    assert 2.5 < figs["H010"] - figs["H012"] < 3.5


def test_conditional_logit_recovers_weight():
    rng = np.random.default_rng(0)
    n_races, field, beta = 3000, 10, 1.5
    x = rng.normal(size=(n_races * field, 1))
    race = np.repeat(np.arange(n_races), field)
    s = (beta * x[:, 0]).reshape(n_races, field)
    p = np.exp(s) / np.exp(s).sum(1, keepdims=True)
    winner = np.array([rng.choice(field, p=row) for row in p])
    won = np.zeros(n_races * field)
    won[np.arange(n_races) * field + winner] = 1
    m = ConditionalLogit(l2=0.0, standardize=False).fit(x, race, won)
    assert abs(m.coef_[0] - beta) < 3 * m.se_[0]
    probs = m.predict(x, race)
    assert np.allclose(pd.Series(probs).groupby(race).sum(), 1.0)


def test_race_log_loss_uniform():
    race = np.repeat([0, 1], [4, 5])
    won = np.array([1, 0, 0, 0, 0, 0.5, 0.5, 0, 0])  # second race is a dead heat
    p = np.r_[np.full(4, 1 / 4), np.full(5, 1 / 5)]
    assert math.isclose(race_log_loss(p, won, race), (math.log(4) + math.log(5)) / 2)
