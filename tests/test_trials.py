import gzip
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hkracing.db import connect
from hkracing.trialform import add_trial_features, trial_runs
from hkracing.trials import parse_trials, store_trials

PAGES = Path(__file__).parent / "pages"


def _page():
    return gzip.decompress((PAGES / "btresult_2025-09-02.html.gz").read_bytes()).decode("utf-8")


def test_parse_trials():
    batches = parse_trials(_page())
    assert len(batches) == 8
    b = batches[0]
    assert (b["venue"], b["batch_no"], b["track"], b["surface"], b["distance_m"]) == \
        ("Sha Tin", 1, "SHA TIN ALL WEATHER TRACK", "AWT", 1200)
    assert (b["going"], b["time_s"], b["sectionals"]) == ("GOOD", 70.24, "24.9 22.8 22.5")
    assert [r["place"] for r in b["runners"]] == list(range(1, 9))
    first, second = b["runners"][:2]
    assert (first["horse_id"], first["name"], first["lbw"], first["time_s"]) == ("HK_2023_J256", "MY WISH", 0.0, 70.24)
    assert (second["jockey"], second["trainer"], second["draw"], second["lbw"]) == ("B Avdulla", "J Richards", 1, 1.5)
    assert second["running_pos"] == "5 3 2"
    last = batches[1]["runners"][-1]
    assert last["name"] == "NAVAL FORCE" and last["lbw"] == 25.75 and last["result"].startswith("Required")
    assert batches[6]["distance_m"] == 1050
    assert parse_trials("<html><body>No information.</body></html>") == []


def test_store_and_trial_features():
    conn = connect(":memory:")
    store_trials(conn, date(2025, 9, 2), parse_trials(_page()))
    assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 8
    assert conn.execute("SELECT COUNT(*) FROM trial_runners").fetchone()[0] == 66
    runs = trial_runs(conn, fit_before="2026/27")
    # Within a batch, figures fall with finishing order; about a point a length at 1200m.
    b1 = runs[runs.trial_id == "2025-09-02_ST_B01"].set_index("horse_id").fig
    assert b1.is_monotonic_decreasing
    assert 1.0 < b1["HK_2023_J256"] - b1["HK_2023_J237"] < 2.5  # 0.24s, beaten 1.5L

    # Three starters in a race a week later: the trial winner, the runner-up, and a horse with no trial.
    df = pd.DataFrame({"race_id": "R1", "horse_id": ["HK_2023_J256", "HK_2023_J237", "X"],
                       "race_date": pd.Timestamp("2025-09-09"), "days": [120.0, 14.0, np.nan],
                       "runs": [5, 3, 0], "first_up": [1.0, 0.0, 0.0]})
    out = add_trial_features(df, runs)
    assert list(out.tr_recent) == [1.0, 1.0, 0.0]
    assert out.tr_won.tolist() == [1.0, 0.0, 0.0]
    assert out.tr_fig_rel[0] > 0 > out.tr_fig_rel[1] and out.tr_fig_rel[2] == 0
    assert out.tr_fig_fresh.tolist() == [out.tr_fig_rel[0], 0.0, 0.0]
    assert np.isclose(out.tr_days[0], np.log(7))
    assert np.isclose(out.tr_count[0], np.log(2)) and out.tr_count[2] == 0
    # A trial on race day or after doesn't count.
    late = add_trial_features(df.assign(race_date=pd.Timestamp("2025-09-02")), runs)
    assert late.tr_recent.sum() == 0
