import gzip
import json
from pathlib import Path

import pandas as pd

from hkracing.db import connect
from hkracing.predict import add_meeting, card_rows, current_win_odds, scorecard

PAGES = Path(__file__).parent / "pages"


def _fixture():
    return json.loads(gzip.decompress((PAGES / "odds_2026-10-24_S1.json.gz").read_bytes()))


class FakeClient:
    def __init__(self, pools):
        self.pools_ = pools

    def pools(self, date, venue):
        return self.pools_


def test_card_rows_and_add_meeting():
    d = _fixture()
    odds = current_win_odds(FakeClient(d["pools"]), "2026-10-24", "S1")
    assert odds[(1, 4)] == 2.7 and (2, 1) not in odds  # scratched: no odds
    races, runners = card_rows(d["meeting"], odds)
    assert [r["race_no"] for r in races] == [1, 2]
    assert all(u["race_id"].startswith("2026-10-24_S1_R0") for u in runners)
    assert len(runners) < 21  # scratched runners are left out
    conn = connect(":memory:")
    add_meeting(conn, "2026-10-24", "S1", races, runners)
    add_meeting(conn, "2026-10-24", "S1", races, runners)  # re-running replaces the card
    assert conn.execute("SELECT COUNT(*) FROM runners WHERE finish_pos IS NULL").fetchone()[0] == len(runners)
    assert conn.execute("SELECT season, n_races FROM meetings").fetchone() == ("2026/27", 0)


def test_scorecard_with_no_results_yet():
    d = _fixture()
    card = pd.DataFrame({"race_id": ["2026-10-24_S1_R01"], "horse_id": ["x"]})
    _, text = scorecard(card, d["meeting"])
    assert text == "No results yet."


def test_model_only_report():
    from hkracing.predict import card_report
    card = pd.DataFrame({"race_no": [1, 1], "class": "Class 4", "distance_m": 1200, "horse_no": [1, 2],
                         "horse_name": ["A", "B"], "draw": [1, 2], "jockey": "J", "trainer": "T", "rating": 50,
                         "win_odds": float("nan"), "p_market": float("nan"), "p_model": [0.75, 0.25],
                         "p_combined": float("nan"), "debut": 0.0, "no_fig": 0.0})
    text = card_report(card, {"train_races": 1, "last_train_date": "2026-10-28"}, {"date": "2026-11-01",
                                                                                     "venueCode": "HV"}, "now")
    assert text.startswith("# Model-only prices") and "|    1.3 |" in text and "Market %" not in text
