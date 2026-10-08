"""Scrape HKJC results and sectional times into SQLite.

    python -m hkracing.scrape --date 2025-07-16            # one meeting
    python -m hkracing.scrape --season 2024/25             # one season
    python -m hkracing.scrape --seasons 2016/17 2025/26    # a range of seasons
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import date, datetime, timedelta

from .db import connect
from .fetch import Fetcher, page_url
from .results import parse_fixture, parse_results, parse_sectionals, season_of

log = logging.getLogger("hkracing.scrape")


def season_months(season: str) -> list[tuple[int, int]]:
    """'2024/25' -> [(2024, 8), ..., (2025, 7)]."""
    start = int(season[:4])
    return [(start, m) for m in range(8, 13)] + [(start + 1, m) for m in range(1, 8)]


def season_meetings(fetcher: Fetcher, season: str, today: date | None = None) -> list[tuple[date, str]]:
    """Past meeting dates and venues for a season, read from the fixture calendar."""
    today = today or date.today()
    out = []
    for y, m in season_months(season):
        if date(y, m, 1) > today:
            break
        url = page_url("fixture", calyear=y, calmonth=f"{m:02d}")
        current = (y, m) == (today.year, today.month)
        for day, venue in parse_fixture(fetcher.get(url, refresh=current)):
            d = date(y, m, day)
            if d < today:
                out.append((d, venue))
    return out


def scrape_meeting(fetcher: Fetcher, conn: sqlite3.Connection, d: date, venue: str) -> int:
    """Fetch and store every race of one meeting. Returns the number of races stored."""
    meeting_id = f"{d.isoformat()}_{venue}"
    racedate = d.strftime("%Y/%m/%d")

    # HKJC occasionally serves an empty "no information" page for a race that exists; refetch once before giving up.
    def results_page(n):
        url = page_url("localresults", racedate=racedate, Racecourse=venue, RaceNo=n)
        return parse_results(fetcher.get(url)) or parse_results(fetcher.get(url, refresh=True))

    def sectionals_page(n):
        url = page_url("displaysectionaltime", racedate=d.strftime("%d/%m/%Y"), RaceNo=n)
        return parse_sectionals(fetcher.get(url)) or parse_sectionals(fetcher.get(url, refresh=True))

    first = results_page(1)
    races = []
    if first is not None:
        if first["venue"] and first["venue"] != venue:
            log.warning("%s: fixture says %s, results page says %s", d, venue, first["venue"])
        races.append(first)
        for n in first["race_nos"][1:]:
            r = results_page(n)
            if r is None:
                log.warning("%s %s: race %d is listed but its results page is empty", d, venue, n)
            else:
                races.append(r)
    # Abandoned meetings (race number 0) and voided races list runners but have no finishers; skip them.
    skipped = [r["race_no"] for r in races if not any(h["finish_pos"] for h in r["runners"])]
    if skipped:
        log.info("%s %s: no finishers in races %s (abandoned or void), skipped", d, venue, skipped)
    races = [r for r in races if r["race_no"] not in skipped]

    with conn:
        conn.execute("INSERT OR REPLACE INTO meetings VALUES (?,?,?,?,?)",
                     (meeting_id, d.isoformat(), venue, season_of(d), len(races)))
        for r in races:
            store_race(conn, meeting_id, r)
            store_sectionals(conn, race_id(meeting_id, r["race_no"]), sectionals_page(r["race_no"]))
    return len(races)


def race_id(meeting_id: str, race_no: int) -> str:
    return f"{meeting_id}_R{race_no:02d}"


def store_race(conn: sqlite3.Connection, meeting_id: str, r: dict) -> None:
    rid = race_id(meeting_id, r["race_no"])
    conn.execute(
        "INSERT OR REPLACE INTO races (race_id, meeting_id, race_no, race_index, race_name, class, rating_band,"
        " distance_m, surface, course, going, prize_hkd, winning_time_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rid, meeting_id, r["race_no"], r["race_index"], r["race_name"], r["class"], r["rating_band"],
         r["distance_m"], r["surface"], r["course"], r["going"], r["prize_hkd"], r["winning_time_s"]))
    for h in r["runners"]:
        conn.execute("INSERT INTO horses (horse_id, brand, name) VALUES (?,?,?) "
                     "ON CONFLICT(horse_id) DO UPDATE SET name = excluded.name",
                     (h["horse_id"], h["brand"], h["name"]))
        conn.execute(
            "INSERT OR REPLACE INTO runners (race_id, horse_id, horse_no, draw, weight_lb, body_weight_lb, jockey,"
            " trainer, finish_pos, finish_status, lbw, running_pos, finish_time_s, win_odds)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, h["horse_id"], h["horse_no"], h["draw"], h["weight_lb"], h["body_weight_lb"], h["jockey"],
             h["trainer"], h["finish_pos"], h["finish_status"], h["lbw"], h["running_pos"], h["finish_time_s"],
             h["win_odds"]))


def store_sectionals(conn: sqlite3.Connection, rid: str, secs: list[dict]) -> None:
    for s in secs:
        conn.execute(
            "INSERT OR REPLACE INTO sectionals (race_id, horse_id, section_no, position, margin_l, time_s, splits)"
            " VALUES (?,?,?,?,?,?,?)",
            (rid, s["horse_id"], s["section_no"], s["position"], s["margin_l"], s["time_s"], s["splits"]))


def all_seasons(first: str, last: str) -> list[str]:
    return [f"{y}/{(y + 1) % 100:02d}" for y in range(int(first[:4]), int(last[:4]) + 1)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="one meeting, YYYY-MM-DD")
    g.add_argument("--season", help="one season, e.g. 2024/25")
    g.add_argument("--seasons", nargs=2, metavar=("FIRST", "LAST"), help="season range, inclusive")
    ap.add_argument("--venue", choices=["ST", "HV"], help="with --date; default: read from the fixture")
    ap.add_argument("--db", default="data/hkracing.db")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between live requests")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    fetcher = Fetcher(args.cache, delay=args.delay)
    conn = connect(args.db)
    if args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d").date()
        venue = args.venue or dict(season_meetings(fetcher, season_of(d), today=d + timedelta(days=1))).get(d)
        if venue is None:
            ap.error(f"no meeting found for {d}; pass --venue")
        n = scrape_meeting(fetcher, conn, d, venue)
        log.info("%s %s: %d races", d, venue, n)
        return

    seasons = [args.season] if args.season else all_seasons(*args.seasons)
    for season in seasons:
        meetings = season_meetings(fetcher, season)
        log.info("season %s: %d meetings in fixture", season, len(meetings))
        total = 0
        for i, (d, venue) in enumerate(meetings, 1):
            n = scrape_meeting(fetcher, conn, d, venue)
            total += n
            log.info("season %s [%d/%d] %s %s: %d races (live requests so far: %d)",
                     season, i, len(meetings), d, venue, n, fetcher.live_requests)
        log.info("season %s done: %d meetings, %d races", season, len(meetings), total)


if __name__ == "__main__":
    main()
