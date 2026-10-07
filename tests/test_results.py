import gzip
from datetime import date
from pathlib import Path

from hkracing.db import connect
from hkracing.results import (parse_fixture, parse_margin, parse_results, parse_sectionals, parse_time,
                              season_of)
from hkracing.scrape import store_race, store_sectionals

PAGES = Path(__file__).parent / "pages"


def page(name):
    return gzip.decompress((PAGES / name).read_bytes()).decode("utf-8")


def test_small_parsers():
    assert parse_time("1:09.96") == 69.96
    assert parse_time("24.00") == 24.0
    assert parse_margin("1-3/4") == 1.75
    assert parse_margin("1/2") == 0.5
    assert parse_margin("HD") == 0.2
    assert parse_margin("10") == 10
    assert parse_margin("") is None
    assert season_of(date(2025, 7, 16)) == "2024/25"
    assert season_of(date(2025, 9, 7)) == "2025/26"


def test_parse_fixture():
    assert parse_fixture(page("fixture_2017-02.html.gz"))[:3] == [(2, "HV"), (5, "ST"), (8, "HV")]


def test_parse_results():
    r = parse_results(page("results_2025-07-16_HV_R1.html.gz"))
    assert r["venue"] == "HV" and r["race_no"] == 1 and r["race_index"] == 839
    assert r["race_nos"] == list(range(1, 10))
    assert (r["class"], r["distance_m"], r["rating_band"]) == ("Class 5", 1200, "40-0")
    assert (r["surface"], r["course"], r["going"]) == ("TURF", "B", "GOOD TO FIRM")
    assert r["prize_hkd"] == 875000 and r["winning_time_s"] == 69.96
    first = r["runners"][0]
    assert first["horse_id"] == "HK_2022_H108" and first["brand"] == "H108"
    assert (first["finish_pos"], first["draw"], first["weight_lb"], first["body_weight_lb"]) == (1, 7, 123, 1125)
    assert (first["running_pos"], first["win_odds"], first["lbw"]) == ("1 1 1", 7.7, 0.0)
    assert len(r["runners"]) == 12


def test_parse_sectionals():
    s = parse_sectionals(page("sectional_2025-07-16_R1.html.gz"))
    assert len(s) == 36
    assert s[1] == {"horse_id": "HK_2022_H108", "section_no": 2, "position": 1, "margin_l": 0.5,
                    "time_s": 23.13, "splits": "11.42 11.71"}
    # Withdrawn horse (no. 3) has blank cells and must produce no rows.
    s = parse_sectionals(page("sectional_2025-12-14_R2_wv.html.gz"))
    assert len({x["horse_id"] for x in s}) == 13 and "HK_2023_J114" not in {x["horse_id"] for x in s}


def test_store_round_trip():
    conn = connect(":memory:")
    conn.execute("INSERT INTO meetings VALUES ('2025-07-16_HV', '2025-07-16', 'HV', '2024/25', 9)")
    store_race(conn, "2025-07-16_HV", parse_results(page("results_2025-07-16_HV_R1.html.gz")))
    store_sectionals(conn, "2025-07-16_HV_R01", parse_sectionals(page("sectional_2025-07-16_R1.html.gz")))
    assert conn.execute("SELECT COUNT(*) FROM runners").fetchone()[0] == 12
    assert conn.execute("SELECT COUNT(*) FROM sectionals").fetchone()[0] == 36
