from hkracing.db import connect
from hkracing.fetch import Fetcher, page_url


class FakeResp:
    text = "<html>ok</html>"
    encoding = "utf-8"

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.calls = 0
        self.headers = {}

    def get(self, url, timeout):
        self.calls += 1
        return FakeResp()


def test_fetch_is_cached(tmp_path):
    s = FakeSession()
    f = Fetcher(tmp_path, delay=0, session=s)
    url = page_url("Racing/LocalResults.aspx", RaceDate="2025/09/07")
    assert f.get(url) == "<html>ok</html>"
    assert f.get(url) == "<html>ok</html>"
    assert s.calls == 1


def test_schema_creates_tables():
    conn = connect(":memory:")
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"meetings", "races", "runners", "sectionals", "horses", "standard_times"} <= names
