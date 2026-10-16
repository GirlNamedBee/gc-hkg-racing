"""Scrape HKJC barrier trial results into SQLite.

    python -m hkracing.trials --since 2016-08-01      # every day from then to yesterday
    python -m hkracing.trials --date 2025-09-02       # one day

HKJC has no list of old trial days (the page's date menu only covers about a year), so every date is
probed: btresult?date=YYYY/MM/DD returns an empty page on days without trials. Wednesday and Sunday race
days are skipped (trials there are very rare); other race days are probed, since Sha Tin and Conghua
sometimes trial on a Happy Valley race day. Each batch is one heat: a venue/track/distance, going, time
and 400m sectionals, then every runner in finishing order with lengths behind, running positions, time
and the stewards' comment ("Passed" in the Result column marks a horse that had to pass the trial).
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from datetime import date, datetime, timedelta

from bs4 import BeautifulSoup

from .db import connect
from .fetch import Fetcher, page_url
from .results import _int, _text, parse_margin, parse_time

log = logging.getLogger("hkracing.trials")

_BATCH = re.compile(r"Batch\s+(\d+)\s*-\s*(.+?)\s*-\s*(\d+)m", re.I)


def parse_trials(html: str) -> list[dict]:
    """Every batch on a trial-results page, with its runners in finishing order. [] on a day without trials."""
    soup = BeautifulSoup(html, "lxml")
    root = soup.find(id="divBtresult")
    if root is None:
        return []
    batches = []
    venue = None
    for el in root.find_all(["div", "table"]):
        if el.name == "div" and "btrcheader" in (el.get("class") or []):
            venue = _text(el).replace("Barrier Trial", "").strip()  # "Sha Tin", "Conghua"
            continue
        if el.name != "table" or "bigborder" in (el.get("class") or []):
            continue
        head = el.find("td", class_="subheader")
        m = _BATCH.search(_text(head).replace("\xa0", " ")) if head else None
        if not m:
            continue
        info = _text(el)
        going = re.search(r"Going:\s*([A-Z /]+?)\s*(?:Time|$)", info)
        t = re.search(r"Time:\s*([\d.:]+)", info)
        secs = re.search(r"Sectional Time:\s*([\d. ]+)", info)
        track = m.group(2).upper()
        batch = {
            "venue": venue, "batch_no": int(m.group(1)), "track": track,
            "surface": "AWT" if "ALL WEATHER" in track else "TURF", "distance_m": int(m.group(3)),
            "going": going.group(1).strip() if going else None,
            "time_s": parse_time(_dotted(t.group(1))) if t else None,
            "sectionals": " ".join(secs.group(1).split()) if secs else None, "runners": [],
        }
        grid = el.find_next_sibling("table", class_="bigborder")
        for i, tr in enumerate(grid.find_all("tr")[1:] if grid else [], 1):
            tds = tr.find_all("td")
            link = tr.find("a", href=re.compile(r"horseid="))
            if len(tds) < 10 or link is None:
                continue
            cells = [_text(td) for td in tds]
            lbw = cells[5].rstrip("L").strip()
            batch["runners"].append({
                "horse_id": re.search(r"horseid=([\w]+)", link["href"]).group(1),
                "name": _text(link), "jockey": cells[1] or None, "trainer": cells[2] or None,
                "draw": _int(cells[3]), "gear": cells[4] or None,
                "lbw": 0.0 if i == 1 and not lbw else parse_margin(lbw),
                "running_pos": cells[6] or None, "place": i,
                "time_s": parse_time(_dotted(cells[7])), "result": cells[8] or None, "comment": cells[9] or None,
            })
        batches.append(batch)
    return batches


def _dotted(s: str) -> str:
    """Trial times are written 1.10.24; parse_time wants 1:10.24."""
    s = (s or "").strip()
    return re.sub(r"^(\d+)\.(\d+\.\d+)$", r"\1:\2", s)


def trial_id(d: date, venue: str | None, batch_no: int) -> str:
    v = {"Sha Tin": "ST", "Happy Valley": "HV", "Conghua": "CH"}.get(venue or "", (venue or "XX")[:2].upper())
    return f"{d.isoformat()}_{v}_B{batch_no:02d}"


def store_trials(conn: sqlite3.Connection, d: date, batches: list[dict]) -> int:
    with conn:
        for b in batches:
            tid = trial_id(d, b["venue"], b["batch_no"])
            conn.execute(
                "INSERT OR REPLACE INTO trials (trial_id, trial_date, venue, batch_no, track, surface, distance_m,"
                " going, time_s, sectionals) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (tid, d.isoformat(), b["venue"], b["batch_no"], b["track"], b["surface"], b["distance_m"],
                 b["going"], b["time_s"], b["sectionals"]))
            conn.execute("DELETE FROM trial_runners WHERE trial_id = ?", (tid,))
            for h in b["runners"]:
                conn.execute(
                    "INSERT OR REPLACE INTO trial_runners (trial_id, horse_id, name, jockey, trainer, draw, gear,"
                    " lbw, running_pos, place, time_s, result, comment) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tid, h["horse_id"], h["name"], h["jockey"], h["trainer"], h["draw"], h["gear"], h["lbw"],
                     h["running_pos"], h["place"], h["time_s"], h["result"], h["comment"]))
    return len(batches)


def probe_dates(conn: sqlite3.Connection, first: date, last: date) -> list[date]:
    race_days = {r[0] for r in conn.execute("SELECT race_date FROM meetings")}
    out = []
    d = first
    while d <= last:
        if not (d.isoformat() in race_days and d.weekday() in (2, 6)):
            out.append(d)
        d += timedelta(days=1)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="one day, YYYY-MM-DD")
    g.add_argument("--since", help="every day from this date to yesterday, newest first")
    ap.add_argument("--until", help="with --since: last day (default yesterday)")
    ap.add_argument("--db", default="data/hkracing.db")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between live requests")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fetcher = Fetcher(args.cache, delay=args.delay)
    conn = connect(args.db)
    parse = lambda s: datetime.strptime(s, "%Y-%m-%d").date()
    if args.date:
        days = [parse(args.date)]
    else:
        last = parse(args.until) if args.until else date.today() - timedelta(days=1)
        days = probe_dates(conn, parse(args.since), last)[::-1]
    found = 0
    for i, d in enumerate(days, 1):
        batches = parse_trials(fetcher.get(page_url("btresult", date=d.strftime("%Y/%m/%d"))))
        if batches:
            found += 1
            store_trials(conn, d, batches)
            log.info("%s: %d batches, %d runners", d, len(batches), sum(len(b["runners"]) for b in batches))
        if i % 100 == 0:
            log.info("[%d/%d] %d trial days so far (live requests: %d)", i, len(days), found, fetcher.live_requests)
    log.info("done: %d days probed, %d with trials", len(days), found)


if __name__ == "__main__":
    main()
