-- One row per race meeting (date + venue).
CREATE TABLE IF NOT EXISTS meetings (
    meeting_id   TEXT PRIMARY KEY,          -- e.g. 2025-09-07_ST
    race_date    DATE NOT NULL,
    venue        TEXT NOT NULL,             -- ST (Sha Tin) / HV (Happy Valley)
    season       TEXT NOT NULL,             -- e.g. 2025/26
    n_races      INTEGER                    -- races with results; 0 = abandoned / no results
);

CREATE TABLE IF NOT EXISTS races (
    race_id      TEXT PRIMARY KEY,          -- meeting_id + race number
    meeting_id   TEXT NOT NULL REFERENCES meetings(meeting_id),
    race_no      INTEGER NOT NULL,
    race_index   INTEGER,                   -- HKJC season race number, e.g. 839
    race_name    TEXT,
    class        TEXT,                      -- Class 1-5, Group, Griffin
    rating_band  TEXT,
    distance_m   INTEGER,
    surface      TEXT,                      -- TURF / AWT
    course       TEXT,                      -- A, A+3, B, C, C+3
    going        TEXT,
    prize_hkd    INTEGER,
    winning_time_s REAL
);

CREATE TABLE IF NOT EXISTS horses (
    horse_id     TEXT PRIMARY KEY,          -- HKJC horse id, e.g. HK_2022_H108 (brand codes get reused)
    brand        TEXT,                      -- brand code, e.g. H108
    name         TEXT,
    origin       TEXT,
    sex          TEXT,
    foaled       DATE,
    colour       TEXT,
    import_type  TEXT,                      -- PP, PPG, ISG, ...
    sire         TEXT,
    dam          TEXT,
    dam_sire     TEXT
);

-- Every HK run in a horse's profile form record (back past our 10 seasons), with pre-race rating and gear.
CREATE TABLE IF NOT EXISTS horse_form (
    horse_id       TEXT NOT NULL REFERENCES horses(horse_id),
    race_date      DATE NOT NULL,
    venue          TEXT NOT NULL,
    race_no        INTEGER NOT NULL,
    race_index     INTEGER,
    season         TEXT,
    finish_status  TEXT,
    finish_pos     INTEGER,
    track          TEXT,                    -- e.g. Turf / "B", AWT
    distance_m     INTEGER,
    going          TEXT,                    -- abbreviated: G, GF, GD, WS, ...
    class          TEXT,
    draw           INTEGER,
    rating         INTEGER,                 -- rating going into the race
    trainer        TEXT,
    jockey         TEXT,
    lbw            REAL,
    win_odds       REAL,
    weight_lb      INTEGER,
    running_pos    TEXT,
    finish_time_s  REAL,
    body_weight_lb INTEGER,
    gear           TEXT,
    PRIMARY KEY (horse_id, race_date, race_no)
);

-- One row per runner per race: pre-race declarations plus result.
CREATE TABLE IF NOT EXISTS runners (
    race_id        TEXT NOT NULL REFERENCES races(race_id),
    horse_id       TEXT NOT NULL REFERENCES horses(horse_id),
    horse_no       INTEGER,
    draw           INTEGER,
    weight_lb      INTEGER,                 -- weight carried
    body_weight_lb INTEGER,                 -- declared horse weight
    rating         INTEGER,                 -- pre-race rating (from horse_form)
    jockey         TEXT,
    trainer        TEXT,
    gear           TEXT,
    finish_pos     INTEGER,                 -- NULL for scratched / DNF
    finish_status  TEXT,                    -- raw placing text (WV, PU, DISQ, ...)
    lbw            REAL,                    -- lengths behind winner
    running_pos    TEXT,                    -- positions at each call, space-separated
    finish_time_s  REAL,
    win_odds       REAL,                    -- final win odds
    PRIMARY KEY (race_id, horse_id)
);

CREATE TABLE IF NOT EXISTS sectionals (
    race_id      TEXT NOT NULL,
    horse_id     TEXT NOT NULL,
    section_no   INTEGER NOT NULL,          -- 1 = first section
    position     INTEGER,                   -- position at end of section
    margin_l     REAL,                      -- lengths behind leader (leader: lengths ahead of 2nd)
    time_s       REAL,                      -- this horse's time for the section
    splits       TEXT,                      -- 200m sub-splits within the section, space-separated
    PRIMARY KEY (race_id, horse_id, section_no)
);

CREATE TABLE IF NOT EXISTS standard_times (
    venue        TEXT NOT NULL,
    surface      TEXT NOT NULL,
    distance_m   INTEGER NOT NULL,
    class        TEXT NOT NULL,
    time_s       REAL NOT NULL,
    season       TEXT NOT NULL,
    PRIMARY KEY (venue, surface, distance_m, class, season)
);
