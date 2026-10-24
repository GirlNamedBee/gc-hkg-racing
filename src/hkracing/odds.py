"""Record HKJC win and place odds through each race day, so pre-race odds movement can be modelled later.

    python -m hkracing.odds snapshot                         # one snapshot of every local meeting on sale
    python -m hkracing.odds watch --minutes 55 --publish /mnt/project-files/gc-hkg-racing/data/odds.db

HKJC keeps no odds history: results pages show only the final win odds. The live odds behind bet.hkjc.com
come from a GraphQL API (info.cld.hkjc.com) that only answers the exact queries the betting site sends (any
other query text gets WHITELIST_ERROR), so the two queries are copied verbatim from the site's bundle into
graphql/. If HKJC changes them, copy the new text from bet.hkjc.com's main.*.js.

Only local meetings (ST, HV) are recorded; simulcast meetings (S1, S2...) are skipped. Snapshots go to their own
database (data/odds.db) so the collector never writes to hkracing.db while other jobs use it:
- odds_runners: each declared runner (horse_id matches hkracing.db), its draw, jockey, trainer and status.
- pool_snapshots: per snapshot, each race's WIN/PLA pool status, investment and odds update time.
- odds_snapshots: per snapshot, each runner's WIN and PLA odds.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path

import requests

log = logging.getLogger("hkracing.odds")

GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base-get/"
LOCAL_VENUES = ("ST", "HV")
HKT = timezone(timedelta(hours=8))

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_runners (
    race_date TEXT NOT NULL, venue TEXT NOT NULL, race_no INTEGER NOT NULL, runner_no INTEGER NOT NULL,
    horse_id TEXT, horse_name TEXT, status TEXT, post_time TEXT, draw INTEGER, weight INTEGER,
    jockey TEXT, trainer TEXT, updated_at TEXT,
    PRIMARY KEY (race_date, venue, race_no, runner_no)
);
CREATE TABLE IF NOT EXISTS pool_snapshots (
    taken_at TEXT NOT NULL, race_date TEXT NOT NULL, venue TEXT NOT NULL, race_no INTEGER NOT NULL,
    pool TEXT NOT NULL, status TEXT, sell_status TEXT, investment REAL, odds_updated TEXT,
    PRIMARY KEY (taken_at, race_date, venue, race_no, pool)
);
CREATE TABLE IF NOT EXISTS odds_snapshots (
    taken_at TEXT NOT NULL, race_date TEXT NOT NULL, venue TEXT NOT NULL, race_no INTEGER NOT NULL,
    pool TEXT NOT NULL, runner_no INTEGER NOT NULL, odds REAL, odds_raw TEXT, hot_fav INTEGER,
    PRIMARY KEY (taken_at, race_date, venue, race_no, pool, runner_no)
);
"""
TABLES = ("odds_runners", "pool_snapshots", "odds_snapshots")


def connect(path: str | Path = "data/odds.db") -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _query(name: str) -> str:
    return files("hkracing").joinpath("graphql", f"{name}.graphql").read_text()


class OddsClient:
    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def post(self, name: str, variables: dict) -> dict:
        for attempt in range(4):
            try:
                resp = self.session.post(GRAPHQL_URL, json={"query": _query(name), "variables": variables},
                                         timeout=30)
                resp.raise_for_status()
                body = resp.json()
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        if body.get("errors"):
            raise RuntimeError(f"{name}: {body['errors'][0].get('message')}")
        return body["data"]

    def meetings(self) -> list[dict]:
        """Local meetings HKJC currently lists (today's, or the next one), with races and runners."""
        out = []
        active = self.post("race_meetings", {})["activeMeetings"] or []
        for m in active:
            if m["venueCode"] not in LOCAL_VENUES:
                continue
            full = self.post("race_meetings", {"date": m["date"], "venueCode": m["venueCode"]})["raceMeetings"]
            out.extend(full or [])
        return out

    def pools(self, date: str, venue: str) -> list[dict]:
        data = self.post("pm_pools", {"date": date, "venueCode": venue, "oddsTypes": ["WIN", "PLA"], "raceNo": 0})
        meetings = data["raceMeetings"] or []
        return meetings[0]["pmPools"] if meetings else []


def _int(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _odds(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None  # "SCR" (scratched) or "" before the pool opens


def store_snapshot(conn: sqlite3.Connection, taken_at: str, meeting: dict, pools: list[dict]) -> int:
    """Store one meeting's runners, pool states and odds. Returns the number of odds rows written."""
    date, venue = meeting["date"], meeting["venueCode"]
    invest = {(p["oddsType"], p["leg"]["races"][0]): p for p in meeting.get("poolInvs") or []
              if p["oddsType"] in ("WIN", "PLA")}
    n = 0
    with conn:
        for race in meeting["races"]:
            for r in race.get("runners") or []:
                conn.execute(
                    "INSERT OR REPLACE INTO odds_runners VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (date, venue, race["no"], _int(r["no"]), (r.get("horse") or {}).get("id"), r.get("name_en"),
                     r.get("status"), race.get("postTime"), _int(r.get("barrierDrawNumber")),
                     _int(r.get("handicapWeight")), (r.get("jockey") or {}).get("code"),
                     (r.get("trainer") or {}).get("code"), taken_at))
        for p in pools:
            race_no, pool = p["leg"]["races"][0], p["oddsType"]
            inv = invest.get((pool, race_no), {})
            conn.execute("INSERT OR REPLACE INTO pool_snapshots VALUES (?,?,?,?,?,?,?,?,?)",
                         (taken_at, date, venue, race_no, pool, p.get("status"), p.get("sellStatus"),
                          _odds(inv.get("investment")), p.get("oddsLastUpdateTime") or p.get("lastUpdateTime")))
            for node in p.get("oddsNodes") or []:
                conn.execute("INSERT OR REPLACE INTO odds_snapshots VALUES (?,?,?,?,?,?,?,?,?)",
                             (taken_at, date, venue, race_no, pool, _int(node["combString"]),
                              _odds(node["oddsValue"]), node["oddsValue"], int(bool(node.get("hotFavourite")))))
                n += 1
    return n


def snapshot(conn: sqlite3.Connection, client: OddsClient, now: datetime | None = None) -> list[dict]:
    """Snapshot every local meeting on sale. Returns the meetings seen (empty when none is listed)."""
    taken_at = (now or datetime.now(HKT)).astimezone(HKT).isoformat(timespec="seconds")
    meetings = client.meetings()
    for m in meetings:
        pools = client.pools(m["date"], m["venueCode"])
        selling = [p for p in pools if any(_odds(n["oddsValue"]) for n in p.get("oddsNodes") or [])]
        n = store_snapshot(conn, taken_at, m, selling)
        log.info("%s %s %s: %d pools with odds, %d odds rows", taken_at, m["date"], m["venueCode"], len(selling), n)
    return meetings


def next_interval(meetings: list[dict], now: datetime) -> float | None:
    """Seconds to the next snapshot: dense near the off, sparse the day before. None when nothing is left to record."""
    posts = []
    for m in meetings:
        for race in m["races"]:
            if (race.get("status") or "").upper() in ("RESULT", "CLOSED", "VOIDED", "ABANDONED"):
                continue
            if race.get("postTime"):
                posts.append(datetime.fromisoformat(race["postTime"]))
    upcoming = [(p - now).total_seconds() for p in posts if p > now - timedelta(minutes=10)]
    if not upcoming:
        return None
    gap = min(upcoming)
    if gap <= 15 * 60:
        return 60
    if gap <= 60 * 60:
        return 3 * 60
    if gap <= 4 * 3600:
        return 10 * 60
    return 30 * 60


def publish(local: Path, target: Path) -> None:
    """Merge rows already at target (another run may have written them) into local, then copy local over it."""
    conn = sqlite3.connect(local)
    if target.exists():
        conn.execute("ATTACH ? AS pub", (str(target),))
        with conn:
            for t in TABLES:  # local odds_runners rows are newer, so existing keys keep the local row
                conn.execute(f"INSERT OR IGNORE INTO {t} SELECT * FROM pub.{t}")
        conn.execute("DETACH pub")
    conn.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp{os.getpid()}")
    shutil.copyfile(local, tmp)
    tmp.replace(target)


def watch(db: Path, minutes: float, publish_to: Path | None, client: OddsClient | None = None) -> int:
    """Snapshot repeatedly for up to `minutes`, publishing after each one. Returns the number of snapshots."""
    client = client or OddsClient()
    if publish_to and publish_to.exists() and not db.exists():
        shutil.copyfile(publish_to, db)
    conn = connect(db)
    conn.close()
    deadline = time.monotonic() + minutes * 60
    count = 0
    while True:
        conn = connect(db)
        try:
            meetings = snapshot(conn, client)
        finally:
            conn.close()
        count += 1
        if publish_to:
            publish(db, publish_to)
        wait = next_interval(meetings, datetime.now(HKT))
        if wait is None:
            log.info("no local races left to record; stopping")
            break
        if time.monotonic() + wait > deadline:
            break
        time.sleep(wait)
    return count


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "watch"])
    ap.add_argument("--db", default="data/odds.db", type=Path)
    ap.add_argument("--minutes", type=float, default=55, help="how long watch runs")
    ap.add_argument("--publish", type=Path, help="copy the database here after each snapshot (merging)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "snapshot":
        conn = connect(args.db)
        snapshot(conn, OddsClient())
        conn.close()
        if args.publish:
            publish(args.db, args.publish)
    else:
        n = watch(args.db, args.minutes, args.publish)
        log.info("%d snapshots", n)


if __name__ == "__main__":
    main()
