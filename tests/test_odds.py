import gzip
import json
from datetime import datetime
from pathlib import Path

from hkracing.odds import HKT, _query, connect, next_interval, publish, store_snapshot, watch

PAGES = Path(__file__).parent / "pages"


def _fixture():
    """A simulcast meeting (S1) on 2026-10-24, races 1-2, as the odds API returned it while selling."""
    return json.loads(gzip.decompress((PAGES / "odds_2026-10-24_S1.json.gz").read_bytes()))


def test_queries_are_packaged():
    assert _query("pm_pools").startswith("query sp4e_racing_pmPool(")
    assert "fragment raceFragment on Race" in _query("race_meetings")


def test_store_snapshot():
    d = _fixture()
    conn = connect(":memory:")
    n = store_snapshot(conn, "2026-10-24T08:20:00+08:00", d["meeting"], d["pools"])
    assert n == 2 * (11 + 10)
    assert conn.execute("SELECT COUNT(*) FROM odds_runners").fetchone()[0] == 21
    assert conn.execute("SELECT odds, hot_fav FROM odds_snapshots WHERE race_no=1 AND pool='WIN' AND runner_no=4"
                        ).fetchone() == (2.7, 1)
    # scratched runners keep the raw "SCR" with no odds
    assert conn.execute("SELECT odds, odds_raw FROM odds_snapshots WHERE race_no=2 AND pool='WIN' AND runner_no=1"
                        ).fetchone() == (None, "SCR")
    assert conn.execute("SELECT investment FROM pool_snapshots WHERE race_no=1 AND pool='PLA'").fetchone() == (593981,)


def test_next_interval():
    def meeting(*posts, status="DECLARED"):
        return [{"races": [{"no": i + 1, "postTime": p, "status": status} for i, p in enumerate(posts)]}]
    now = datetime(2026, 10, 25, 12, 0, tzinfo=HKT)
    assert next_interval(meeting("2026-10-25T12:10:00+08:00"), now) == 60
    assert next_interval(meeting("2026-10-25T12:40:00+08:00"), now) == 180
    assert next_interval(meeting("2026-10-25T14:00:00+08:00"), now) == 600
    assert next_interval(meeting("2026-10-26T12:40:00+08:00"), now) == 1800
    assert next_interval(meeting("2026-10-25T11:00:00+08:00"), now) is None
    assert next_interval(meeting("2026-10-25T12:10:00+08:00", status="RESULT"), now) is None
    assert next_interval([], now) is None


def test_publish_merges_rows_from_another_run(tmp_path):
    d = _fixture()
    a, b, target = tmp_path / "a.db", tmp_path / "b.db", tmp_path / "pub" / "odds.db"
    for path, t in ((a, "2026-10-24T08:20:00+08:00"), (b, "2026-10-24T08:30:00+08:00")):
        conn = connect(path)
        store_snapshot(conn, t, d["meeting"], d["pools"])
        conn.close()
    publish(a, target)
    publish(b, target)
    conn = connect(target)
    assert conn.execute("SELECT COUNT(DISTINCT taken_at) FROM odds_snapshots").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM odds_runners").fetchone()[0] == 21


class FakeClient:
    def __init__(self, meeting, pools):
        self.meeting, self.pools_ = meeting, pools

    def meetings(self):
        return [self.meeting]

    def pools(self, date, venue):
        return self.pools_


def test_watch_stops_when_races_are_over(tmp_path):
    d = _fixture()
    for race in d["meeting"]["races"]:
        race["status"] = "RESULT"
    n = watch(tmp_path / "odds.db", 5, tmp_path / "pub.db", FakeClient(d["meeting"], d["pools"]))
    assert n == 1
    conn = connect(tmp_path / "pub.db")
    assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 42
