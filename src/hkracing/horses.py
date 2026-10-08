"""Parse and scrape HKJC horse profile pages (full form record, pedigree).

    python -m hkracing.horses                  # every horse in the runners table
    python -m hkracing.horses --active-since 2026-01-01

The profile's form record carries each run's pre-race rating and gear, which the results pages lack.
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
from .results import _float, _int, _text, parse_margin, parse_time, season_of

log = logging.getLogger("hkracing.horses")

_INFO = {"Country of Origin": "origin", "Colour / Sex": "colour_sex", "Import Type": "import_type",
         "Sire": "sire", "Dam": "dam", "Dam's Sire": "dam_sire"}


def parse_horse(html: str) -> tuple[dict, list[dict]] | None:
    """Return (profile info, form rows) or None when the page has no profile."""
    soup = BeautifulSoup(html, "lxml")
    form = soup.select_one("table.bigborder")
    if form is None:
        return None
    info = {}
    for tr in soup.select("table.table_eng_text tr"):
        tds = [_text(td) for td in tr.find_all("td", recursive=False)]
        if len(tds) >= 3:
            label = tds[0].split(" / Age")[0]  # "Country of Origin / Age" on active horses
            if label in _INFO:
                info[_INFO[label]] = tds[2].split(" / ")[0] if label == "Country of Origin" else tds[2]
    colour_sex = info.pop("colour_sex", "")
    info["colour"], _, info["sex"] = colour_sex.partition(" / ")

    cols = [_text(td) for td in form.find("tr").find_all("td", recursive=False)]
    idx = {name: i for i, name in enumerate(cols)}
    rows = []
    for tr in form.find_all("tr"):
        link = tr.find("a", href=re.compile("localresults", re.I))
        if link is None:
            continue  # season headers, spacers, overseas runs
        tds = [_text(td) for td in tr.find_all("td", recursive=False)]

        def cell(name):
            i = idx.get(name)
            return tds[i] if i is not None and i < len(tds) else ""

        m = re.search(r"racedate=(\d{4})/(\d{2})/(\d{2}).*?Racecourse=(\w+).*?RaceNo=(\d+)", link["href"], re.I)
        if not m:
            continue
        race_date = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        place = cell("Pla.")
        gear = cell("Gear")
        rows.append({
            "race_date": race_date.isoformat(),
            "venue": m.group(4).upper(),
            "race_no": int(m.group(5)),
            "race_index": _int(cell("Race Index")),
            "season": season_of(race_date),
            "finish_status": place or None,
            "finish_pos": _int(place) if re.match(r"\d", place) else None,
            "track": cell("RC /Track/ Course").partition(" / ")[2] or None,
            "distance_m": _int(cell("Dist.")),
            "going": cell("G") or None,
            "class": cell("Race Class") or None,
            "draw": _int(cell("Dr.")),
            "rating": _int(cell("Rtg.")),
            "trainer": cell("Trainer") or None,
            "jockey": cell("Jockey") or None,
            "lbw": parse_margin(cell("LBW")),
            "win_odds": _float(cell("Win Odds")),
            "weight_lb": _int(cell("Act. Wt.")),
            "running_pos": cell("Running Position") or None,
            "finish_time_s": parse_time(cell("Finish Time")),
            "body_weight_lb": _int(cell("Declar. Horse Wt.")),
            "gear": None if gear in ("", "--") else gear,
        })
    return info, rows


_FORM_COLS = ["race_date", "venue", "race_no", "race_index", "season", "finish_status", "finish_pos", "track",
              "distance_m", "going", "class", "draw", "rating", "trainer", "jockey", "lbw", "win_odds",
              "weight_lb", "running_pos", "finish_time_s", "body_weight_lb", "gear"]


def store_horse(conn: sqlite3.Connection, horse_id: str, info: dict, rows: list[dict]) -> None:
    conn.execute("UPDATE horses SET origin=?, sex=?, colour=?, import_type=?, sire=?, dam=?, dam_sire=?"
                 " WHERE horse_id=?",
                 (info.get("origin"), info.get("sex") or None, info.get("colour") or None, info.get("import_type"),
                  info.get("sire"), info.get("dam"), info.get("dam_sire"), horse_id))
    conn.executemany(
        f"INSERT OR REPLACE INTO horse_form (horse_id, {', '.join(_FORM_COLS)})"
        f" VALUES (?, {', '.join('?' * len(_FORM_COLS))})",
        [(horse_id, *(r[c] for c in _FORM_COLS)) for r in rows])


def fill_runner_ratings(conn: sqlite3.Connection) -> int:
    """Copy pre-race rating and gear from horse_form onto matching runners rows."""
    with conn:
        cur = conn.execute(
            "UPDATE runners SET rating = f.rating, gear = f.gear FROM horse_form f"
            " WHERE runners.horse_id = f.horse_id"
            " AND runners.race_id = f.race_date || '_' || f.venue || '_R' || printf('%02d', f.race_no)")
    return cur.rowcount


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--active-since", help="only horses that ran on or after this date (YYYY-MM-DD)")
    ap.add_argument("--db", default="data/hkracing.db")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between live requests")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    fetcher = Fetcher(args.cache, delay=args.delay)
    conn = connect(args.db)
    # Most recently active horses first, so a partial run covers the horses that matter for upcoming races.
    q = ("SELECT u.horse_id FROM runners u GROUP BY u.horse_id"
         + (" HAVING MAX(substr(u.race_id, 1, 10)) >= ?" if args.active_since else "")
         + " ORDER BY MAX(substr(u.race_id, 1, 10)) DESC")
    horse_ids = [r[0] for r in conn.execute(q, (args.active_since,) if args.active_since else ())]
    log.info("%d horses to fetch", len(horse_ids))
    for i, horse_id in enumerate(horse_ids, 1):
        parsed = parse_horse(fetcher.get(page_url("horse", horseid=horse_id, Option=1)))
        if parsed is None:
            log.warning("%s: no profile", horse_id)
            continue
        with conn:
            store_horse(conn, horse_id, *parsed)
        if i % 100 == 0:
            log.info("[%d/%d] horses (live requests so far: %d)", i, len(horse_ids), fetcher.live_requests)
    log.info("runners updated with rating/gear: %d", fill_runner_ratings(conn))


if __name__ == "__main__":
    main()
