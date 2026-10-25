import numpy as np
import pandas as pd

from hkracing.courseform import add_course_features, venue_offset


def _runs():
    df = pd.DataFrame({
        "race_id": [f"R{i}" for i in range(6)],
        "horse_id": ["A"] * 6,
        "season": ["2017/18"] * 6,
        "race_date": pd.date_range("2017-09-01", periods=6, freq="14D"),
        "venue": ["ST", "HV", "HV", "ST", "ST", "HV"],
        "surface": ["TURF", "TURF", "TURF", "AWT", "TURF", "TURF"],
        "fig": [70.0, 73.0, 75.0, 90.0, 71.0, np.nan],
        "won": [0, 1, 0, 0, 0, 0.0],
        "p_market": [0.1, 0.2, 0.25, 0.1, 0.1, 0.2],
    })
    df["runs"] = np.arange(6)
    return df


def test_venue_offset():
    # Turf pairs: ST->HV +3, HV->HV, HV->ST (skipping the AWT run) -4 -> (3 + 4) / 2
    assert venue_offset(_runs(), "2018/19") == 3.5


def test_course_features():
    out = add_course_features(_runs(), delta=1.0)
    assert out.log_venue_runs.tolist() == list(np.log1p([0, 0, 1, 0, 1, 2]))
    assert out.first_venue.tolist() == [0, 1, 0, 0, 0, 0]  # AWT run doesn't count
    assert out.fig_venue_best5.iloc[5] == 74 and out.fig_venue_best5.iloc[4] == 70
    # Last run at HV: adjusted HV figs 72, 74 (mean 73) vs ST turf 70, 71 (70.5); harmonic 1 -> 2.5 * 1 / 4
    assert abs(out.venue_pref.iloc[5] - 0.625) < 1e-9
    assert out.venue_pref.iloc[3] == 0 and out.venue_ae.iloc[3] == 0  # all-weather race
    # HV: 1 win from 0.45 expected; ST turf: 0 from 0.2 -> log(2/1.45) - log(1/1.2)
    assert abs(out.venue_ae.iloc[5] - (np.log(2 / 1.45) - np.log(1 / 1.2))) < 1e-9
    assert np.allclose(out.hv_form_at_st, [0, 0, 0, 2 / 3, 2 / 3, 0])
    assert np.allclose(out.st_form_at_hv, [0, 1, 0.5, 0, 0, 2 / 3])
    assert out.venue_switch.tolist() == [0, 1, 0, 1, 0, 1]
