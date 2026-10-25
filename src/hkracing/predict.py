"""Race-day predictions: win probabilities and fair prices for a meeting's declared runners, then a scorecard.

    python -m hkracing.predict card    --out data/reports/2026-10-25_ST   # before racing: today's card
    python -m hkracing.predict results --out data/reports/2026-10-25_ST   # after racing: score the predictions

`card` reads the declared runners (draw, weight, rating, gear, jockey, trainer, declared body weight) and the
latest win odds from HKJC's betting API (hkracing.odds), adds them to a scratch copy of the database as an
unfinished meeting, and builds the same features the walk-forward uses. Stage A is fitted on every finished
race from --first-train on; stage B (model + market) is fitted on the out-of-sample predictions the last
walk-forward run wrote (walk_forward_predictions.csv.gz), so it never sees in-sample stage A probabilities.

The market probability is the win odds at the time of the run with the takeout stripped, not the final odds,
so re-run `card` close to the off for the fairest comparison. An overlay is a runner where the combined
probability times the current odds exceeds 1 (positive expected return at those odds); with a model that adds
almost nothing to the market (see walk_forward.md) overlays are mostly noise, not betting advice.

`results` reads the finishing positions and final odds from the same API and scores each probability
(market at prediction time, market at final odds, stage A, combined) by log loss and McFadden R².
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .db import connect
from .features import FEATURES, build_features
from .model import ConditionalLogit, score
from .odds import HKT, OddsClient, _int, _odds
from .results import season_of
from .speed import build as build_speed

log = logging.getLogger("hkracing.predict")


def card_rows(meeting: dict, win_odds: dict[tuple[int, int], float]) -> tuple[list[dict], list[dict]]:
    """Races and declared (not scratched) runners of one API meeting, in the database's column layout."""
    date, venue = meeting["date"], meeting["venueCode"]
    races, runners = [], []
    for race in meeting["races"]:
        rid = f"{date}_{venue}_R{int(race['no']):02d}"
        races.append({"race_id": rid, "race_no": int(race["no"]), "race_name": race.get("raceName_en"),
                      "class": race.get("raceClass_en"), "distance_m": _int(race.get("distance")),
                      "surface": "AWT" if "ALL WEATHER" in (race["raceTrack"]["description_en"] or "").upper()
                      else "TURF",
                      "course": (race.get("raceCourse") or {}).get("displayCode"),
                      "going": (race.get("go_en") or "").upper(), "post_time": race.get("postTime")})
        for r in race.get("runners") or []:
            no = _int(r["no"])
            if no is None or r.get("status") not in ("Declared", "Ran"):
                continue  # standby or scratched; declared runners show as "Ran" once the race is over
            runners.append({"race_id": rid, "horse_id": r["horse"]["id"], "horse_no": no,
                            "horse_name": r.get("name_en"), "draw": _int(r.get("barrierDrawNumber")),
                            "weight_lb": _int(r.get("handicapWeight")), "body_weight_lb": _int(r.get("currentWeight")),
                            "rating": _int(r.get("currentRating")), "jockey": (r.get("jockey") or {}).get("name_en"),
                            "trainer": (r.get("trainer") or {}).get("name_en"),
                            "gear": (r.get("gearInfo") or "").strip() or None,
                            "win_odds": win_odds.get((int(race["no"]), no)),
                            "final_pos": _int(r.get("finalPosition")) or None,
                            "dead_heat": bool(r.get("deadHeat")), "final_odds": _odds(r.get("winOdds"))})
    return races, runners


def current_win_odds(client: OddsClient, date: str, venue: str) -> dict[tuple[int, int], float]:
    odds = {}
    for p in client.pools(date, venue):
        if p["oddsType"] != "WIN":
            continue
        for node in p.get("oddsNodes") or []:
            o = _odds(node["oddsValue"])
            if o:
                odds[(p["leg"]["races"][0], _int(node["combString"]))] = o
    return odds


def add_meeting(conn: sqlite3.Connection, date: str, venue: str, races: list[dict], runners: list[dict]) -> None:
    """Insert the card as an unfinished meeting: no finishing positions or times, odds = current odds."""
    mid = f"{date}_{venue}"
    with conn:
        conn.execute("DELETE FROM runners WHERE race_id LIKE ?", (mid + "%",))
        conn.execute("DELETE FROM races WHERE meeting_id = ?", (mid,))
        conn.execute("DELETE FROM meetings WHERE meeting_id = ?", (mid,))
        conn.execute("INSERT INTO meetings VALUES (?, ?, ?, ?, 0)",
                     (mid, date, venue, season_of(datetime.fromisoformat(date).date())))
        for r in races:
            conn.execute("INSERT INTO races (race_id, meeting_id, race_no, race_name, class, distance_m, surface, "
                         "course, going) VALUES (?,?,?,?,?,?,?,?,?)",
                         (r["race_id"], mid, r["race_no"], r["race_name"], r["class"], r["distance_m"],
                          r["surface"], r["course"], r["going"]))
        for u in runners:
            conn.execute("INSERT OR IGNORE INTO horses (horse_id, brand, name) VALUES (?,?,?)",
                         (u["horse_id"], u["horse_id"].rsplit("_", 1)[-1], u["horse_name"]))
            conn.execute("INSERT INTO runners (race_id, horse_id, horse_no, draw, weight_lb, body_weight_lb, rating, "
                         "jockey, trainer, gear, win_odds) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (u["race_id"], u["horse_id"], u["horse_no"], u["draw"], u["weight_lb"],
                          u["body_weight_lb"], u["rating"], u["jockey"], u["trainer"], u["gear"], u["win_odds"]))


def stage_b(predictions: Path) -> ConditionalLogit:
    """Market + model weights from earlier walk-forward out-of-sample predictions."""
    oos = pd.read_csv(predictions).sort_values(["race_id"], kind="stable")
    oos = oos[(oos.p_model > 0) & (oos.p_market > 0)]
    return ConditionalLogit(l2=0.0, standardize=False).fit(
        np.log(oos[["p_model", "p_market"]].to_numpy()), oos.race_id, oos.won)


def predict(db: Path, meeting: dict, win_odds: dict, predictions: Path, first_train: str = "2017/18",
            l2: float = 1.0) -> tuple[pd.DataFrame, dict]:
    date, venue = meeting["date"], meeting["venueCode"]
    races, runners = card_rows(meeting, win_odds)
    missing = [u["horse_name"] for u in runners if not u["win_odds"]]
    if missing:
        log.warning("no win odds yet for %d runners (%s); they are left out", len(missing), ", ".join(missing[:5]))
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "scratch.db"
        shutil.copyfile(db, scratch)
        conn = connect(scratch)
        add_meeting(conn, date, venue, races, runners)
        build_speed(conn)
        df = build_features(conn)
        conn.close()
    df = df.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    today = df.race_date == pd.Timestamp(date)
    train = df[~today & (df.season >= first_train)]
    card = df[today].copy()
    a = ConditionalLogit(l2=l2).fit(train[FEATURES], train.race_id, train.won)
    b = stage_b(predictions)
    card["p_model"] = a.predict(card[FEATURES], card.race_id)
    card["p_combined"] = b.predict(np.log(card[["p_model", "p_market"]].to_numpy()), card.race_id)
    info = {"train_races": train.race_id.nunique(), "last_train_date": train.race_date.max().date().isoformat(),
            "alpha": b.coef_[0], "gamma": b.coef_[1]}
    extra = pd.DataFrame(runners)[["race_id", "horse_id", "horse_no", "horse_name"]]
    card = card.merge(extra, on=["race_id", "horse_id"], how="left")
    keep = ["race_id", "race_no", "class", "distance_m", "horse_no", "horse_name", "horse_id", "draw", "jockey",
            "trainer", "rating", "win_odds", "p_market", "p_model", "p_combined", "no_fig", "debut"]
    return card[keep].sort_values(["race_no", "horse_no"]).reset_index(drop=True), info


def _price(p: float) -> str:
    return f"{1 / p:.1f}" if p > 0 else "-"


def card_report(card: pd.DataFrame, info: dict, meeting: dict, taken_at: str) -> str:
    lines = [f"# Predictions: {meeting['date']} {meeting['venueCode']}", "",
             f"Odds taken {taken_at}. Stage A fitted on {info['train_races']} races up to {info['last_train_date']}; "
             f"stage B weights: model {info['alpha']:.3f}, market {info['gamma']:.3f}.", "",
             "Probabilities are in percent. Fair = 1 / combined probability (HK odds include the stake). "
             "Value = combined probability × current odds; above 1.00 is an overlay at the current odds. "
             "Model value is the same for stage A alone, which shows where the fundamentals disagree with the "
             "market (the takeout is about 18%, so the market's own value is about 0.82). "
             "This is a test of the model, not betting advice: walk-forward, the model adds almost nothing to the "
             "market, so most overlays are noise.", ""]
    for race_no, r in card.groupby("race_no"):
        first = r.iloc[0]
        lines += [f"## Race {race_no}: {first['class']}, {first.distance_m}m", ""]
        t = pd.DataFrame({
            "No": r.horse_no, "Horse": r.horse_name, "Dr": r.draw, "Jockey": r.jockey,
            "Odds": r.win_odds, "Market %": 100 * r.p_market, "Model %": 100 * r.p_model,
            "Combined %": 100 * r.p_combined, "Fair": [_price(p) for p in r.p_combined],
            "Value": r.p_combined * r.win_odds, "Model value": r.p_model * r.win_odds,
            "Note": np.where(r.debut > 0, "debut", np.where(r.no_fig > 0, "no figure", ""))})
        t = t.sort_values("Combined %", ascending=False)
        lines += [t.to_markdown(index=False, floatfmt=".2f"), ""]
    return "\n".join(lines)


def scorecard(card: pd.DataFrame, meeting: dict) -> tuple[pd.DataFrame, str]:
    _, runners = card_rows(meeting, {})
    res = pd.DataFrame(runners)[["race_id", "horse_id", "final_pos", "dead_heat", "final_odds"]]
    df = card.merge(res, on=["race_id", "horse_id"], how="left")
    done = df.groupby("race_id").final_pos.transform(lambda s: (s == 1).any())
    df = df[done].copy()
    if df.empty:
        return df, "No results yet."
    n_win = (df.final_pos == 1).groupby(df.race_id).transform("sum")
    df["won"] = np.where(df.final_pos == 1, 1 / n_win, 0.0)
    df["field"] = df.groupby("race_id").horse_id.transform("size")
    inv = 1 / df.final_odds
    df["p_final_market"] = inv / inv.groupby(df.race_id).transform("sum")
    rows = [{"probability": name, **score(df, col)} for name, col in
            (("market at prediction time", "p_market"), ("market at final odds", "p_final_market"),
             ("model (stage A)", "p_model"), ("combined", "p_combined"))]
    summary = pd.DataFrame(rows)
    # Per race: winner and the probabilities each source gave it, and how the combined model's overlays did.
    win = df[df.won > 0].sort_values("race_no")
    per_race = pd.DataFrame({"Race": win.race_no, "Winner": win.horse_name, "Final odds": win.final_odds,
                             "Market %": 100 * win.p_market, "Model %": 100 * win.p_model,
                             "Combined %": 100 * win.p_combined,
                             "Combined rank": [int((df[(df.race_id == rid)].p_combined > p).sum()) + 1
                                               for rid, p in zip(win.race_id, win.p_combined)]})
    over = df[df.p_combined * df.win_odds > 1]
    ret = (over.final_odds * over.won).sum()
    fav = df.loc[df.groupby("race_id").p_combined.idxmax()]
    mfav = df.loc[df.groupby("race_id").p_market.idxmax()]
    lines = [f"# Scorecard: {meeting['date']} {meeting['venueCode']} ({df.race_id.nunique()} races)", "",
             "Log loss is the mean over races of minus the log probability given to the winner; lower is better. "
             "R² is McFadden's against 1/field. One meeting is far too small to tell models apart: the walk-forward "
             "over six seasons is the real test.", "",
             summary.to_markdown(index=False, floatfmt=".4f"), "",
             "## Winners", "", per_race.to_markdown(index=False, floatfmt=".1f"), "",
             f"Combined model's top pick won {int(fav.won.gt(0).sum())} of {len(fav)} races; "
             f"the market favourite won {int(mfav.won.gt(0).sum())}.", "",
             f"Overlays at prediction-time odds (value > 1): {len(over)} runners, {int(over.won.gt(0).sum())} won; "
             f"$1 on each at final odds returns ${ret:.2f} for ${len(over)} staked "
             f"({(ret / len(over) - 1) * 100 if len(over) else 0:+.0f}%).", ""]
    return df, "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["card", "results"])
    ap.add_argument("--db", default="data/hkracing.db", type=Path)
    ap.add_argument("--predictions", default="data/reports/walk_forward_predictions.csv.gz", type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output prefix, e.g. data/reports/2026-10-25_ST")
    ap.add_argument("--venue", choices=["ST", "HV"], help="when two local meetings are listed")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = OddsClient()
    meetings = [m for m in client.meetings() if not args.venue or m["venueCode"] == args.venue]
    if not meetings:
        raise SystemExit("HKJC lists no local meeting")
    meeting = meetings[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv = args.out.with_name(args.out.name + "_predictions.csv")
    if args.command == "card":
        taken_at = datetime.now(HKT).isoformat(timespec="minutes")
        odds = current_win_odds(client, meeting["date"], meeting["venueCode"])
        card, info = predict(args.db, meeting, odds, args.predictions)
        card.assign(odds_taken_at=taken_at).to_csv(csv, index=False)
        text = card_report(card, info, meeting, taken_at)
        args.out.with_name(args.out.name + "_card.md").write_text(text)
    else:
        card = pd.read_csv(csv)
        scored, text = scorecard(card, meeting)
        scored.to_csv(args.out.with_name(args.out.name + "_scored.csv"), index=False)
        args.out.with_name(args.out.name + "_scorecard.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
