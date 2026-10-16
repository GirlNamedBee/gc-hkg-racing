import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from hkracing.db import connect
from hkracing.trackform import TRACKWORK_FEATURES, add_trackwork_features, track_work
from hkracing.trackwork import parse_gallop, parse_trackwork, store_trackwork

PAGES = Path(__file__).parent / "pages"


def _page():
    return gzip.decompress((PAGES / "trackwork_HK_2022_H108.html.gz").read_bytes()).decode("utf-8")


def test_parse_trackwork():
    rows = parse_trackwork(_page())
    assert len(rows) == 2505
    assert rows[0] == {"work_date": "2026-10-15", "work_type": "Equine Treadmill", "track": "Sha Tin",
                       "workout": "Treadmill - Walk With Incline", "gear": None}
    gallop = next(r for r in rows if r["work_type"] == "Gallop")
    assert (gallop["work_date"], gallop["track"], gallop["splits"], gallop["time_s"], gallop["rider"]) == \
        ("2026-10-10", "Sha Tin AWT", "30.2 25.4", 55.6, "K Teetan")
    assert all("time_s" in r for r in rows if r["work_type"] == "Gallop")
    assert rows[-1]["work_date"] == "2022-09-01"
    assert parse_trackwork("<html><body>retired horse profile</body></html>") is None


def test_parse_gallop_formats():
    assert parse_gallop("31.0 29.1 (1.00.1) (R.B.)") == {"splits": "31.0 29.1", "time_s": 60.1, "rider": "R.B."}
    assert parse_gallop("30.1 22.9 (53.0) (C L Chau) (Partner Horse)")["rider"] == "C L Chau"
    assert parse_gallop("SmT 1 Round - Fast (R.B.)") == {}


def test_trackwork_features():
    conn = connect(":memory:")
    store_trackwork(conn, "HK_2022_H108", parse_trackwork(_page()))
    work = track_work(conn)
    assert len(work) == 2505
    # Race on 2026-10-14 at Happy Valley, ridden by K Teetan; plus a horse with no trackwork page.
    df = pd.DataFrame({"race_id": "R1", "horse_id": ["HK_2022_H108", "X"],
                       "race_date": pd.Timestamp("2026-10-14"), "jockey": ["K Teetan", "Z Purton"]})
    out = add_trackwork_features(df, work).set_index("horse_id")
    h = out.loc["HK_2022_H108"]
    # Gallops on 10/10, 07/10, 03/10 in the 14 days before (the race day itself excluded).
    assert h.tw_gallops == np.log1p(3)
    assert h.tw_days_gallop == np.log1p(4)
    assert h.tw_jockey == 1.0 and h.tw_named_rider == 1.0
    assert 0 < h.tw_conghua < 1
    assert bool(h.tw_covered) and not bool(out.loc["X", "tw_covered"])
    # The uncovered horse takes the covered average, so nothing separates the two within the race.
    for col in TRACKWORK_FEATURES:
        if col not in ("tw_speed_rel", "tw_speed_trend"):
            assert out.loc["X", col] == h[col], col
    assert out.loc["X", "tw_speed_trend"] == 0.0
