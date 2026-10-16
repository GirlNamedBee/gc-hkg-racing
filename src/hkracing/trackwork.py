"""Scrape HKJC trackwork records (morning work) into SQLite.

    python -m hkracing.trackwork --active-since 2023-09-01    # horses that ran on or after this date
    python -m hkracing.trackwork --horse HK_2022_H108

trackworkresult?horseid=... lists every piece of work a horse has done since it arrived in Hong Kong, newest
first: date, type (Gallop, Trotting, Swimming, Barrier Trial, Race, Equine Treadmill, Aqua Walker),
racecourse/track, the workout text and gear. A gallop's workout reads "30.2 25.4 (55.6) (K Teetan)": the
times of the last sections (each about 400m), their total, and the rider ("R.B." is the trainer's track
rider). HKJC only keeps these pages for horses still in training; a retired horse's page redirects to its
"otherhorse" profile, which has no trackwork, so history comes only from horses active when it is scraped.
The daily trackwork search goes back only three weeks.
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from datetime import datetime

from bs4 import BeautifulSoup

from .db import connect
from .fetch import Fetcher, page_url
from .results import _text

log = logging.getLogger("hkracing.trackwork")

_GALLOP = re.compile(r"^((?:\d+\.\d\s+)*\d+\.\d)\s*\((?:(\d+)\.)?(\d+\.\d)\)\s*(?:\(([^)]+)\))?")


def parse_trackwork(html: str) -> list[dict] | None:
    """Every trackwork row, newest first. None when the page has no trackwork table (retired horse)."""
    soup = BeautifulSoup(html, "lxml")
    table = None
    for t in soup.select("table.table_bd"):
        head = t.find("tr")
        if head and "Workouts" in _text(head):
            table = t
            break
    if table is None:
        return None
    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = [_text(td) for td in tr.find_all("td", recursive=False)]
        if len(tds) < 5:
            continue
        try:
            d = datetime.strptime(tds[0], "%d/%m/%Y").date()
        except ValueError:
            continue
        row = {"work_date": d.isoformat(), "work_type": tds[1], "track": tds[2] or None,
               "workout": tds[3] or None, "gear": tds[4] or None}
        row.update(parse_gallop(tds[3]) if tds[1] == "Gallop" else {})
        rows.append(row)
    return rows


def parse_gallop(text: str) -> dict:
    """'30.2 25.4 (55.6) (K Teetan)' -> splits, total time and rider. Unknown formats give {}."""
    m = _GALLOP.match((text or "").strip())
    if not m:
        return {}
    minutes = int(m.group(2) or 0)  # totals over a minute read 1.00.1
    return {"splits": m.group(1), "time_s": round(60 * minutes + float(m.group(3)), 2), "rider": m.group(4)}


def store_trackwork(conn: sqlite3.Connection, horse_id: str, rows: list[dict]) -> None:
    with conn:
        conn.execute("DELETE FROM trackwork WHERE horse_id = ?", (horse_id,))
        conn.executemany(
            "INSERT INTO trackwork (horse_id, work_date, seq, work_type, track, workout, gear, splits, time_s,"
            " rider) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(horse_id, r["work_date"], i, r["work_type"], r["track"], r["workout"], r["gear"],
              r.get("splits"), r.get("time_s"), r.get("rider")) for i, r in enumerate(rows)])
        conn.execute("INSERT OR REPLACE INTO trackwork_fetched (horse_id, fetched, n_rows) VALUES (?,?,?)",
                     (horse_id, datetime.now().date().isoformat(), len(rows)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--active-since", help="horses that ran on or after this date (YYYY-MM-DD)")
    g.add_argument("--horse", help="one horse id, e.g. HK_2022_H108")
    ap.add_argument("--part", help="K/N: only every Nth horse starting at the Kth, to split a run in two")
    ap.add_argument("--refresh", action="store_true", help="refetch pages already in the cache")
    ap.add_argument("--db", default="data/hkracing.db")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between live requests")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fetcher = Fetcher(args.cache, delay=args.delay)
    conn = connect(args.db)
    if args.horse:
        horse_ids = [args.horse]
    else:
        horse_ids = [r[0] for r in conn.execute(
            "SELECT horse_id FROM runners GROUP BY horse_id HAVING MAX(substr(race_id, 1, 10)) >= ?"
            " ORDER BY MAX(substr(race_id, 1, 10)) DESC", (args.active_since,))]
    if args.part:
        k, n = map(int, args.part.split("/"))
        horse_ids = horse_ids[k - 1::n]
    log.info("%d horses to fetch", len(horse_ids))
    found = 0
    for i, horse_id in enumerate(horse_ids, 1):
        rows = parse_trackwork(fetcher.get(page_url("trackworkresult", horseid=horse_id), refresh=args.refresh))
        if rows is not None:
            found += 1
            store_trackwork(conn, horse_id, rows)
        if i % 100 == 0:
            log.info("[%d/%d] %d with trackwork (live requests: %d)", i, len(horse_ids), found,
                     fetcher.live_requests)
    log.info("done: %d horses, %d with trackwork", len(horse_ids), found)


if __name__ == "__main__":
    main()
