"""Gear: per-starter features from the gear declared for the race (hkracing.horses copies it onto runners).

    runner_gear(conn)            -> DataFrame (race_id, horse_id, gear, gear_known), one row per runner
    add_gear_features(df, gear)  -> df with the GEAR_FEATURES columns added

HKJC writes gear as items joined by "/", e.g. "B1/TT". A suffix marks a change since the last run:
"1" first time ever worn, "2" put back on after a spell without it, "-" taken off today. The race card
shows the same string before the race, so none of this leaks the result.

Items: B blinkers, BO blinker on one side, V visor, CP cheek pieces, SR shadow roll, PC pacifier with cowls,
P pacifier, E ear plugs, H hood, TT tongue tie, XB crossed nose band, SB sheepskin browband, and a few rarer
ones. "Vision" gear is the group that limits what the horse sees (B, BO, V, CP, SR, PC), the classic
first-time-blinkers angle.

    gear_first       any item worn for the first time
    gear_vis_first   first-time vision gear
    gear_vis_back    vision gear put back on
    gear_vis_off     vision gear taken off
    gear_tt_first    first-time tongue tie
    gear_off         any item taken off
    gear_vis         wearing vision gear today (level, not change)
    gear_n           number of items worn today

The gear string only comes from horse profile pages. A runner with no profile row has gear_known False and
every feature 0, which reads as "no gear, no change"; profiles of retired horses are fetched too, so this is
rare from 2016/17 on.
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

GEAR_FEATURES = ["gear_first", "gear_vis_first", "gear_vis_back", "gear_vis_off", "gear_tt_first", "gear_off",
                 "gear_vis", "gear_n"]

VISION = {"B", "BO", "V", "CP", "SR", "PC"}
_ITEM = re.compile(r"^([A-Z]+)([12-]?)$")


def parse_gear(gear: str | None) -> list[tuple[str, str]]:
    """'B1/TT/CP-' -> [('B', '1'), ('TT', ''), ('CP', '-')]. Unparseable items are dropped."""
    if not isinstance(gear, str) or not gear:  # None / NaN = no gear
        return []
    out = []
    for part in gear.replace(" ", "").split("/"):
        m = _ITEM.match(part.upper())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def gear_row(gear: str | None) -> dict:
    items = parse_gear(gear)
    worn = [(k, s) for k, s in items if s != "-"]
    return {
        "gear_first": float(any(s == "1" for _, s in items)),
        "gear_vis_first": float(any(k in VISION and s == "1" for k, s in items)),
        "gear_vis_back": float(any(k in VISION and s == "2" for k, s in items)),
        "gear_vis_off": float(any(k in VISION and s == "-" for k, s in items)),
        "gear_tt_first": float(any(k == "TT" and s == "1" for k, s in items)),
        "gear_off": float(any(s == "-" for _, s in items)),
        "gear_vis": float(any(k in VISION for k, _ in worn)),
        "gear_n": float(len(worn)),
    }


def runner_gear(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT u.race_id, u.horse_id, u.gear, f.horse_id IS NOT NULL AS gear_known
        FROM runners u JOIN races r USING (race_id) JOIN meetings m USING (meeting_id)
        LEFT JOIN horse_form f ON f.horse_id = u.horse_id AND f.race_date = m.race_date
                              AND f.venue = m.venue AND f.race_no = r.race_no
        WHERE u.win_odds > 0""", conn)


def add_gear_features(df: pd.DataFrame, gear: pd.DataFrame) -> pd.DataFrame:
    g = gear.drop_duplicates(["race_id", "horse_id"])
    feats = pd.DataFrame([gear_row(x) for x in g.gear], index=g.index)
    g = pd.concat([g[["race_id", "horse_id", "gear_known"]], feats], axis=1)
    df = df.drop(columns=[c for c in g.columns if c in df.columns and c not in ("race_id", "horse_id")])
    df = df.merge(g, on=["race_id", "horse_id"], how="left")
    df["gear_known"] = df.gear_known.fillna(0).astype(bool)
    df[GEAR_FEATURES] = df[GEAR_FEATURES].fillna(0.0)
    return df
