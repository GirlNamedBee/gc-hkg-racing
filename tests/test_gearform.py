import numpy as np
import pandas as pd

from hkracing.features import _surface_form
from hkracing.gearform import add_gear_features, gear_row, parse_gear


def test_parse_gear():
    assert parse_gear("B1/TT/CP-") == [("B", "1"), ("TT", ""), ("CP", "-")]
    assert parse_gear(None) == [] and parse_gear(float("nan")) == [] and parse_gear("") == []
    assert parse_gear("H2/XB") == [("H", "2"), ("XB", "")]


def test_gear_row():
    r = gear_row("B1/TT")
    assert r["gear_first"] == r["gear_vis_first"] == r["gear_vis"] == 1 and r["gear_n"] == 2
    assert r["gear_off"] == r["gear_tt_first"] == 0
    r = gear_row("CP-/TT1")
    assert r["gear_vis_off"] == r["gear_off"] == r["gear_tt_first"] == r["gear_first"] == 1
    assert r["gear_vis"] == 0 and r["gear_n"] == 1  # cheek pieces came off, only the tongue tie is worn
    assert gear_row("V2")["gear_vis_back"] == 1 and gear_row("V2")["gear_vis_first"] == 0


def test_add_gear_features_unknown_is_zero():
    df = pd.DataFrame({"race_id": ["R1", "R1"], "horse_id": ["A", "B"]})
    gear = pd.DataFrame({"race_id": ["R1"], "horse_id": ["A"], "gear": ["B1"], "gear_known": [1]})
    out = add_gear_features(df, gear)
    assert out.gear_known.tolist() == [True, False]
    assert out.gear_vis_first.tolist() == [1.0, 0.0]


def test_surface_form():
    df = pd.DataFrame({
        "horse_id": ["A"] * 5,
        "surface": ["TURF", "TURF", "AWT", "TURF", "AWT"],
        "fig": [70.0, 74.0, 80.0, 72.0, np.nan],
    })
    df["runs"] = np.arange(5)
    out = _surface_form(df)
    assert out.first_surface.tolist() == [0, 0, 1, 0, 0]
    assert out.log_surf_runs.tolist() == list(np.log1p([0, 1, 0, 2, 1]))
    assert out.fig_surf_best5.iloc[3] == 74 and out.fig_surf_best5.iloc[4] == 80
    # Last run: AWT mean 80 vs turf mean 72 over 1 and 3 runs, harmonic 0.75 -> 8 * 0.75 / 3.75
    assert abs(out.surf_pref.iloc[4] - 1.6) < 1e-9
    assert out.surf_pref.iloc[3] < 0  # back on turf, which it handled worse than its one AWT run
