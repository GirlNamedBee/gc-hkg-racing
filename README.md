# gc-hkg-racing

Win-probability and fair-price model for Hong Kong racing, built from HKJC race data.
See the plan in the project files: `plan/hk-racing-model-plan.md`.

## Layout

- `src/hkracing/fetch.py`: polite page fetcher. Every page is cached to `data/cache/` so it is fetched once; requests are rate-limited.
- `src/hkracing/db.py` + `schema.sql`: SQLite database (`data/hkracing.db`) for meetings, races, runners, sectionals, horses.
- `src/hkracing/results.py`: parsers for the fixture calendar, LocalResults and sectional-time pages.
- `src/hkracing/scrape.py`: finds each season's meetings from the fixture calendar, then stores every race's results and sectionals.
- `src/hkracing/validate.py`: per-season completeness report (HKJC numbers races 1..N each season, so gaps show up).
- `src/hkracing/standards.py`: HKJC standard times (current season only; HKJC doesn't publish old ones).
- `src/hkracing/speed.py`: speed figures. Fits course pars, class offsets and a daily track variant to winning
  times (starting from the HKJC standards), then rates every runner's time against them. One point is about one
  length at 1200m; a typical Class 4 winner is 80.
- `src/hkracing/sectionals.py`: per-run sectional measures: last-400m speed against the course par and against
  the rest of the field, and figures adjusted for race shape (how the pace suited the horse's running position).
- `src/hkracing/trials.py`: barrier trial scraper. HKJC lists no old trial days, so it probes every date
  (about 3,700 pages for ten seasons) and stores each batch and its runners (`trials`, `trial_runners`).
- `src/hkracing/trialform.py`: a figure for every trial run (time against a trial-course par and day variant),
  flags from the stewards' comment, and per-starter trial features (latest and best trial since the last run,
  trial count, lengths beaten, trial form of first-up horses and debutants).
- `src/hkracing/trackwork.py`: trackwork (morning work) scraper. One page per horse lists every gallop, trot,
  swim and treadmill session since it arrived (`trackwork`). HKJC removes the page when a horse retires, so
  history exists only for horses in training when the scraper runs; re-run it each season to keep it growing.
- `src/hkracing/trackform.py`: per-starter trackwork features (gallops in the last 14/28 days, days since the
  last gallop, swims, treadmill and aqua-walker sessions, share of work at Conghua, whether today's jockey rode
  a recent gallop, gallop speed against the track's norm). They are not in the win model: tested among covered
  starters they add nothing to the market (see the report).
- `src/hkracing/horses.py`: horse profile scraper (pedigree and the full form record, which carries each run's
  pre-race rating and declared gear). Retired horses' pages still have the form record, so every horse is fetched.
- `src/hkracing/gearform.py`: gear features from the declared gear string (first-time, back-on and removed
  flags for vision gear, tongue tie and any item). Kept in the model; on their own they add nothing to the market.
- `src/hkracing/courseform.py`: course preference, Happy Valley vs Sha Tin turf: runs and best figure at
  today's course, a shrunk figure preference and a wins-vs-market preference, plus course-switch and form-level
  features (how much of the recent form came from HV when today is at Sha Tin, and the reverse). HV form is a
  lower level, and HV figures run about a length above Sha Tin ones for the same horse, so that offset (fitted
  on seasons before the first test season) comes off HV figures first. In the model; on their own they add
  nothing to the market.
- `src/hkracing/odds.py`: race-day odds recorder. HKJC keeps no odds history (results pages show only the final
  win odds), so this snapshots the live WIN and PLA odds of local meetings from the betting site's API into
  `data/odds.db`: every 30 minutes the day before, every 10 minutes on the day, every 3 minutes in the last hour
  before a race and every minute in the last 15. Odds-movement features need about a season of this.
- `src/hkracing/features.py`: pre-race features per starter (form figures, sectionals, early speed, draw,
  jockey/trainer, fitness, turf vs all-weather aptitude, gear, HV vs ST course preference...).
- `src/hkracing/model.py`: conditional logit win model, combined with the market in a second logit, evaluated
  walk-forward one season at a time. The report also tests whether the features add anything to the market
  odds on their own ("market increment").
- `src/hkracing/predict.py`: race-day predictions. Reads the declared runners and current win odds from HKJC's
  betting API, builds the same features for them, and writes stage A, combined and market probabilities with fair
  prices; `results` scores them once the races are run.
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
    python -m hkracing.horses                              # every horse's profile (about 5,800 pages)
    python -m hkracing.horses --missing                    # only horses with no form record yet
    python -m hkracing.trials --since 2016-08-01           # barrier trials (every date to yesterday)
    python -m hkracing.trackwork --active-since 2023-09-01 # trackwork of horses still in training

Re-running is cheap: pages already in `data/cache/` are parsed from disk, so the database can be rebuilt
offline by deleting it and running the same command. `horse_id` is HKJC's full id (e.g. `HK_2022_H108`),
because brand codes like `H108` get reused over the years.

## Odds recording

    python -m hkracing.odds snapshot                 # one snapshot of every local meeting with odds on sale
    python -m hkracing.odds watch --minutes 55 --publish /mnt/project-files/gc-hkg-racing/data/odds.db

`watch` keeps snapshotting until the meeting's last race or the time limit, merging into the `--publish` copy
after each snapshot. It needs `info.cld.hkjc.com` in the environment's allowed domains. The API only accepts the
exact query text the betting site sends, kept in `src/hkracing/graphql/`; if HKJC changes it, requests fail
with `WHITELIST_ERROR` and the queries need copying again from bet.hkjc.com's `main.*.js`.

## Model

    pip install -e ".[model]"
    python -m hkracing.standards     # once a season
    python -m hkracing.model         # rebuilds speed figures, then writes data/reports/walk_forward.md

## Race day

    python -m hkracing.scrape --season 2026/27              # bring this season's results up to date first
    python -m hkracing.predict card --out data/reports/2026-10-25_ST
    python -m hkracing.predict results --out data/reports/2026-10-25_ST

## Data etiquette

Personal research only. Default delay is 3 s between the start of live requests; never redistribute scraped data.
Check HKJC's site terms before running at scale.
