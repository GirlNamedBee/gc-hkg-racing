import gzip
from pathlib import Path

from hkracing.db import connect
from hkracing.horses import fill_runner_ratings, parse_horse, store_horse
from hkracing.results import parse_results
from hkracing.scrape import store_race

PAGES = Path(__file__).parent / "pages"


def page(name):
    return gzip.decompress((PAGES / name).read_bytes()).decode("utf-8")


def test_parse_horse():
    info, rows = parse_horse(page("horse_HK_2022_H108_all.html.gz"))
    assert info == {"origin": "NZ", "import_type": "PPG", "sire": "Satono Aladdin", "dam": "Lemonade",
                    "dam_sire": "Bertolini", "colour": "Bay", "sex": "Gelding"}
    assert len(rows) == 42  # 5-10-5-42 starts
    win = next(r for r in rows if r["race_date"] == "2025-07-16")
    assert (win["venue"], win["race_no"], win["race_index"], win["rating"], win["gear"]) == ("HV", 1, 839, 28, "B/TT")
    assert win["finish_time_s"] == 69.96


def test_fill_runner_ratings():
    conn = connect(":memory:")
    conn.execute("INSERT INTO meetings VALUES ('2025-07-16_HV', '2025-07-16', 'HV', '2024/25', 9)")
    store_race(conn, "2025-07-16_HV", parse_results(page("results_2025-07-16_HV_R1.html.gz")))
    store_horse(conn, "HK_2022_H108", *parse_horse(page("horse_HK_2022_H108_all.html.gz")))
    assert fill_runner_ratings(conn) == 1
    row = conn.execute("SELECT rating, gear FROM runners WHERE horse_id = 'HK_2022_H108'").fetchone()
    assert row == (28, "B/TT")
    assert conn.execute("SELECT sire FROM horses WHERE horse_id = 'HK_2022_H108'").fetchone() == ("Satono Aladdin",)
