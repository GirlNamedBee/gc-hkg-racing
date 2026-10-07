# gc-hkg-racing

Win-probability and fair-price model for Hong Kong racing, built from HKJC race data.
See the plan in the project files: `plan/hk-racing-model-plan.md`.

## Layout

- `src/hkracing/fetch.py`: polite page fetcher. Every page is cached to `data/cache/` so it is fetched once; requests are rate-limited.
- `src/hkracing/db.py` + `schema.sql`: SQLite database (`data/hkracing.db`) for meetings, races, runners, sectionals, horses.
- `src/hkracing/results.py`: parsers for the fixture calendar, LocalResults and sectional-time pages.
- `src/hkracing/scrape.py`: finds each season's meetings from the fixture calendar, then stores every race's results and sectionals.
- `src/hkracing/validate.py`: per-season completeness report (HKJC numbers races 1..N each season, so gaps show up).
- `data/` is git-ignored (raw HTML cache and database).

## Setup

    python -m venv .venv && . .venv/bin/activate
    pip install -e ".[dev]"
    pytest

## Scraping

    python -m hkracing.scrape --date 2025-07-16          # one meeting
    python -m hkracing.scrape --season 2024/25           # one season (about 1,700 pages)
    python -m hkracing.scrape --seasons 2016/17 2025/26  # ten seasons
    python -m hkracing.validate

Re-running is cheap: pages already in `data/cache/` are parsed from disk, so the database can be rebuilt
offline by deleting it and running the same command. `horse_id` is HKJC's full id (e.g. `HK_2022_H108`),
because brand codes like `H108` get reused over the years.

## Data etiquette

Personal research only. Default delay is 3 s between the start of live requests; never redistribute scraped data.
Check HKJC's site terms before running at scale.
