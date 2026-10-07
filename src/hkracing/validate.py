"""Completeness checks for scraped results: python -m hkracing.validate [--db data/hkracing.db]

HKJC numbers every race in a season (the "(839)" next to "RACE 1"), so a season is complete when its
race numbers run 1..N with no gaps.
"""
from __future__ import annotations

import argparse
import sqlite3


def season_report(conn: sqlite3.Connection) -> list[dict]:
    out = []
    for (season,) in conn.execute("SELECT DISTINCT season FROM meetings ORDER BY season"):
        idx = [r[0] for r in conn.execute(
            "SELECT r.race_index FROM races r JOIN meetings m USING (meeting_id) WHERE m.season = ?"
            " ORDER BY r.race_index", (season,))]
        n_max = max((i for i in idx if i), default=0)
        missing = sorted(set(range(1, n_max + 1)) - set(idx))
        row = conn.execute(
            "SELECT COUNT(DISTINCT m.meeting_id), SUM(m.n_races = 0) FROM meetings m WHERE m.season = ?",
            (season,)).fetchone()
        runners, no_sect = conn.execute(
            "SELECT COUNT(*), SUM(NOT EXISTS (SELECT 1 FROM sectionals s WHERE s.race_id = u.race_id"
            " AND s.horse_id = u.horse_id)) FROM runners u JOIN races r USING (race_id)"
            " JOIN meetings m USING (meeting_id) WHERE m.season = ? AND u.finish_pos IS NOT NULL",
            (season,)).fetchone()
        out.append({"season": season, "meetings": row[0], "empty_meetings": row[1] or 0, "races": len(idx),
                    "last_race_no": n_max, "missing_race_nos": missing, "finishers": runners,
                    "finishers_without_sectionals": no_sect or 0})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/hkracing.db")
    args = ap.parse_args(argv)
    conn = sqlite3.connect(args.db)
    print(f"{'season':8} {'meetings':>8} {'empty':>5} {'races':>5} {'last#':>5} {'finishers':>9} "
          f"{'no sect':>7}  missing race numbers")
    for r in season_report(conn):
        print(f"{r['season']:8} {r['meetings']:8} {r['empty_meetings']:5} {r['races']:5} {r['last_race_no']:5} "
              f"{r['finishers']:9} {r['finishers_without_sectionals']:7}  {r['missing_race_nos'][:20] or '-'}")


if __name__ == "__main__":
    main()
