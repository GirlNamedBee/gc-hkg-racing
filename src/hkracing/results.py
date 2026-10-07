"""Parse HKJC fixture, LocalResults and DisplaySectionalTime pages."""
from __future__ import annotations

import re
from datetime import date

from bs4 import BeautifulSoup

VENUES = {"Sha Tin": "ST", "Happy Valley": "HV"}
_MARGINS = {"NOSE": 0.05, "SH": 0.1, "HD": 0.2, "N": 0.3, "---": 0.0, "DH": 0.0}  # "ML" (many lengths) -> None


def _text(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ")).strip() if el is not None else ""


def season_of(d: date) -> str:
    """HK seasons run September to July: 2025-07-16 -> '2024/25', 2025-09-07 -> '2025/26'."""
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}/{(start + 1) % 100:02d}"


def parse_time(s: str | None) -> float | None:
    """'1:09.96' -> 69.96, '24.00' -> 24.0."""
    if not s:
        return None
    m = re.fullmatch(r"(?:(\d+)[:.])?(\d+\.\d+)", s.strip())
    if not m:
        return None
    return round(int(m.group(1) or 0) * 60 + float(m.group(2)), 2)


def parse_margin(s: str | None) -> float | None:
    """Lengths: '1-3/4' -> 1.75, '1/2' -> 0.5, 'HD' -> 0.2, '---' (winner) -> 0."""
    if not s:
        return None
    s = s.strip().upper()
    if s in _MARGINS:
        return _MARGINS[s]
    m = re.fullmatch(r"(?:(\d+)-?)?(?:(\d+)/(\d+))?", s)
    if not m or not (m.group(1) or m.group(2)):
        return None
    whole = int(m.group(1) or 0)
    frac = int(m.group(2)) / int(m.group(3)) if m.group(2) else 0.0
    return whole + frac


def _int(s: str | None) -> int | None:
    m = re.match(r"\d+", (s or "").replace(",", ""))
    return int(m.group()) if m else None


def _float(s: str | None) -> float | None:
    try:
        return float((s or "").replace(",", ""))
    except ValueError:
        return None


def parse_fixture(html: str) -> list[tuple[int, str]]:
    """Meeting days in a fixture-calendar month page: [(day, 'ST'|'HV'), ...]."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for td in soup.select("table.table_bd td.calendar"):
        day = _int(_text(td.select_one("span.f_fl")))
        venues = [img.get("alt") for img in td.select("span.f_fr img") if img.get("alt") in ("ST", "HV")]
        if day and venues:
            out.append((day, venues[0]))
    return out


def parse_results(html: str) -> dict | None:
    """Parse one LocalResults race page. Returns None when the page has no race result."""
    soup = BeautifulSoup(html, "lxml")
    runners_tbl = soup.select_one("table.draggable")
    header = next((t for t in soup.find_all("table") if re.match(r"RACE \d+", _text(t.find("thead")))), None)
    if runners_tbl is None or header is None:
        return None

    venue_el = soup.select_one("table.js_racecard")
    venue = next((code for name, code in VENUES.items() if name in _text(venue_el)), None)
    race_nos = {int(m) for m in re.findall(r"RaceNo=(\d+)", str(venue_el), re.I)}

    title = _text(header.find("thead"))
    m = re.match(r"RACE (\d+)(?: \((\d+)\))?", title)
    race = {"race_no": int(m.group(1)), "race_index": _int(m.group(2)), "venue": venue}

    rows = [[_text(td) for td in tr.find_all("td")] for tr in header.find("tbody").find_all("tr")]
    rows = [r for r in rows if any(r)]
    cond = rows[0][0] if rows else ""
    race["race_name"] = rows[1][0] if len(rows) > 1 else None
    race["prize_hkd"] = _int(re.sub(r"[^\d]", "", rows[2][0])) if len(rows) > 2 else None
    # cond: "Class 5 - 1200M - (40-0)", "Group One - 2000M", "Griffin Race - 1200M", "4 Year Olds - 1800M - (...)"
    parts = [p.strip() for p in cond.split(" - ")]
    race["class"] = parts[0] or None
    dist = next((p for p in parts if re.fullmatch(r"\d+M", p)), None)
    race["distance_m"] = int(dist[:-1]) if dist else None
    band = next((p for p in parts if p.startswith("(")), None)
    race["rating_band"] = band.strip("()") if band else None

    def labelled(label):
        for r in rows:
            if len(r) > 2 and r[1].rstrip(" :") == label:
                return r[2:]
        return []

    race["going"] = (labelled("Going") or [None])[0]
    course = (labelled("Course") or [""])[0]
    race["surface"] = "AWT" if "ALL WEATHER" in course.upper() else ("TURF" if course else None)
    cm = re.search(r'"([^"]+)"', course)
    race["course"] = cm.group(1) if cm else None
    times = [parse_time(t.strip("()")) for t in labelled("Time")]
    race["winning_time_s"] = times[-1] if times else None

    # Map columns by header text so column order changes don't break parsing.
    cols = [_text(td) for td in runners_tbl.find("thead").find_all("td")]
    idx = {name: i for i, name in enumerate(cols)}

    def cell(tds, name):
        i = idx.get(name)
        return tds[i] if i is not None and i < len(tds) else None

    runners = []
    for tr in runners_tbl.find("tbody").find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < len(cols) - 1:
            continue
        horse_td = cell(tds, "Horse")
        link = horse_td.find("a", href=True) if horse_td else None
        hm = re.search(r"horseid=([\w]+)", link["href"], re.I) if link else None
        brand = re.search(r"\(([A-Z]\d{3})\)", _text(horse_td))
        if not hm and not brand:
            continue
        place = _text(cell(tds, "Pla."))
        rp = cell(tds, "Running Position")
        running = [_text(d) for d in rp.find_all("div") if not d.find("div")] if rp else []
        runners.append({
            "horse_id": hm.group(1) if hm else brand.group(1),
            "brand": brand.group(1) if brand else None,
            "name": _text(link) if link else _text(horse_td).split(" (")[0],
            "horse_no": _int(_text(cell(tds, "Horse No."))),
            "jockey": _text(cell(tds, "Jockey")) or None,
            "trainer": _text(cell(tds, "Trainer")) or None,
            "weight_lb": _int(_text(cell(tds, "Act. Wt."))),
            "body_weight_lb": _int(_text(cell(tds, "Declar. Horse Wt."))),
            "draw": _int(_text(cell(tds, "Dr."))),
            "finish_status": place or None,
            "finish_pos": _int(place) if re.match(r"\d", place) else None,
            "lbw": parse_margin(_text(cell(tds, "LBW"))) if re.match(r"\d", place) else None,
            "running_pos": " ".join(p for p in running if p) or None,
            "finish_time_s": parse_time(_text(cell(tds, "Finish Time"))),
            "win_odds": _float(_text(cell(tds, "Win Odds"))),
        })
    for h in runners:
        if h["finish_pos"] == 1:
            h["lbw"] = 0.0  # winners occasionally show "-HD" etc.
    if race["winning_time_s"] is None:  # a few header blocks are blank; fall back to the winner's time
        race["winning_time_s"] = min((h["finish_time_s"] for h in runners if h["finish_time_s"]), default=None)
    race["runners"] = runners
    race["race_nos"] = sorted(race_nos | {race["race_no"]})
    return race


def parse_sectionals(html: str) -> list[dict]:
    """Parse one DisplaySectionalTime page into per-runner, per-section rows."""
    soup = BeautifulSoup(html, "lxml")
    tbl = soup.select_one("table.race_table")
    if tbl is None:
        return []
    out = []
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        link = tr.find("a", href=re.compile("horseid=", re.I))
        if not link or len(tds) < 4:
            continue
        horse_id = re.search(r"horseid=(\w+)", link["href"], re.I).group(1)
        for n, td in enumerate(tds[3:-1], start=1):
            ps = td.find_all("p", recursive=False)
            if not ps:
                continue
            pos = td.select_one("span.f_fl")
            margin = td.find("i")
            time_p = ps[1] if len(ps) > 1 else None
            splits = time_p.select("span.color_blue2 span") if time_p else []
            own = time_p.find(string=True, recursive=False) if time_p else None
            if not _text(pos) and not (own and own.strip()):
                continue  # blank cell, e.g. a withdrawn horse
            out.append({
                "horse_id": horse_id,
                "section_no": n,
                "position": _int(_text(pos)),
                "margin_l": parse_margin(_text(margin)),
                "time_s": parse_time(own.strip() if own else None),
                "splits": " ".join(_text(s) for s in splits) or None,
            })
    return out
