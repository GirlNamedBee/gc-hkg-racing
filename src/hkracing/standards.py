"""Scrape HKJC standard times (page "racing-course-time") into the standard_times table.

    python -m hkracing.standards

The page is a Next.js app: the table is not in the HTML but in the JSON the page streams into
``self.__next_f.push(...)`` script tags, under a "sectionalData" object. HKJC only publishes the current
season's standards, so each scrape is stored under the season of the page's update date.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone

from .db import connect
from .fetch import Fetcher
from .results import season_of

log = logging.getLogger("hkracing.standards")

HKT = timezone(timedelta(hours=8))
URL = "https://racing.hkjc.com/en-us/local/page/racing-course-time"

# Section anchor -> (venue, surface)
_SECTIONS = {"StandardSectionalST": ("ST", "TURF"), "StandardSectionalHV": ("HV", "TURF"),
             "StandardSectionalSTAW": ("ST", "AWT")}
# HKJC class option value -> our class label (the races.class values these standards apply to)
_CLASSES = {"GroupRace": "Group", "GriffinRace": "Griffin", **{f"Class{i}": f"Class {i}" for i in range(1, 6)}}


def parse_standard_time(s: str) -> float | None:
    """'1.09.25' -> 69.25, '0.55.95' -> 55.95, '-' -> None."""
    m = re.fullmatch(r"(\d+)\.(\d{2})\.(\d{2})", s.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100


def _sectional_data(html: str) -> dict:
    for chunk in re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', html, re.S):
        if "standardTimes" not in chunk:
            continue
        text = json.loads(f'"{chunk}"')
        start = text.index('{"sectionalData"')
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return obj["sectionalData"]
    raise ValueError("no standard times data on page")


def parse_standards(html: str) -> tuple[str, list[dict]]:
    """Return (season, rows) with rows of venue, surface, distance_m, class, time_s."""
    data = _sectional_data(html)
    updated = datetime.fromtimestamp(data["updateDate"]["dateValue"] / 1000, HKT).date()
    season = season_of(updated)
    rows = []
    for section in data["children"]:
        key = _SECTIONS.get(section["anchor"]["value"])
        if key is None:
            log.warning("skipping unknown section %s", section["anchor"]["value"])
            continue
        venue, surface = key
        for dist in section["children"]:
            for cell in dist["children"]:
                cls = _CLASSES.get(cell["class"]["targetItem"]["optionValue"]["value"])
                t = parse_standard_time(cell["standardTimes"]["value"])
                if cls and t:
                    rows.append({"venue": venue, "surface": surface, "distance_m": int(dist["distance"]["value"]),
                                 "class": cls, "time_s": t})
    return season, rows


def store_standards(conn: sqlite3.Connection, season: str, rows: list[dict]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO standard_times (venue, surface, distance_m, class, time_s, season) "
        "VALUES (:venue, :surface, :distance_m, :class, :time_s, :season)",
        [{**r, "season": season} for r in rows])
    conn.commit()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/hkracing.db")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--refresh", action="store_true", help="refetch the page even if cached")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    season, rows = parse_standards(Fetcher(args.cache).get(URL, refresh=args.refresh))
    store_standards(connect(args.db), season, rows)
    log.info("stored %d standard times for %s", len(rows), season)


if __name__ == "__main__":
    main()
