"""Benter-style win model: a conditional logit on runner features, then a second logit with the market.

    python -m hkracing.model                       # walk-forward evaluation, report to data/reports/
    python -m hkracing.model --first-test 2019/20

Stage A (fundamental):  p_i = exp(β·x_i) / Σ_j exp(β·x_j) over the runners in a race.
Stage B (combined):     p_i ∝ exp(α·log pA_i + γ·log pMarket_i), where pMarket is the final win odds with the
                        takeout stripped, (1/odds) / Σ(1/odds).

Market increment (a diagnostic, not used for prices): a one-stage conditional logit on log pMarket plus every
feature, walk-forward the same way. If the features carry information the final odds don't, it scores above
the market alone; the gain is reported per race with its standard error.

Walk-forward: for each test season, stage A is fitted on every earlier season, and stage B on the stage A
predictions those earlier seasons got when they were themselves the test season (so stage B only ever sees
genuinely out-of-sample stage A probabilities). The first test season has no stage B.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .db import connect
from .features import FEATURES, SECTIONAL_FEATURES, SURFACE_FEATURES, build_features
from .courseform import COURSE_FEATURES, venue_offset
from .gearform import GEAR_FEATURES
from .trackform import TRACKWORK_FEATURES
from .trialform import TRIAL_FEATURES
from .speed import build as build_speed

log = logging.getLogger("hkracing.model")


@dataclass
class ConditionalLogit:
    """Multinomial logit with one choice set per race. `won` may be fractional for dead heats."""
    l2: float = 1.0
    standardize: bool = True
    coef_: np.ndarray | None = None
    se_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def _prep(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit(self, X, race, won):
        X = np.asarray(X, dtype=float)
        if self.standardize:
            self.mean_, self.scale_ = X.mean(0), X.std(0)
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.mean_, self.scale_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
        Z = self._prep(X)
        codes, starts = _groups(race)
        won = np.asarray(won, dtype=float)
        n_races = len(starts)

        def nll(beta):
            s = Z @ beta
            lse = _group_logsumexp(s, codes, starts)
            p = np.exp(s - lse[codes])
            ll = (won * (s - lse[codes])).sum()
            # gradient: Σ_races (x_winner − E_p[x])
            wsum = np.add.reduceat(won, starts)
            grad = Z.T @ won - Z.T @ (p * wsum[codes])
            return (-ll + 0.5 * self.l2 * beta @ beta) / n_races, (-grad + self.l2 * beta) / n_races

        res = minimize(nll, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
        self.coef_ = res.x
        # Standard errors from the observed information matrix.
        p = self._probs(Z, codes, starts)
        zbar = np.add.reduceat(Z * p[:, None], starts)
        H = (Z * p[:, None]).T @ Z - zbar.T @ zbar + self.l2 * np.eye(Z.shape[1])
        self.se_ = np.sqrt(np.diag(np.linalg.pinv(H)))
        return self

    def _probs(self, Z, codes, starts):
        s = Z @ self.coef_
        return np.exp(s - _group_logsumexp(s, codes, starts)[codes])

    def predict(self, X, race):
        codes, starts = _groups(race)
        return self._probs(self._prep(X), codes, starts)


def _groups(race) -> tuple[np.ndarray, np.ndarray]:
    """Rows must be contiguous by race. Returns per-row race code and the start row of each race."""
    race = np.asarray(race)
    change = np.r_[True, race[1:] != race[:-1]]
    starts = np.flatnonzero(change)
    codes = np.cumsum(change) - 1
    if len(starts) != len(pd.unique(race)):
        raise ValueError("rows are not grouped by race")
    return codes, starts


def _group_logsumexp(s, codes, starts):
    m = np.maximum.reduceat(s, starts)
    return m + np.log(np.add.reduceat(np.exp(s - m[codes]), starts))


def race_log_loss(p, won, race) -> float:
    """Mean over races of −Σ won·log p (the winner's log probability)."""
    codes, starts = _groups(race)
    return float(-np.add.reduceat(np.asarray(won) * np.log(np.clip(p, 1e-12, 1)), starts).mean())


def score(df: pd.DataFrame, col: str) -> dict:
    ll = race_log_loss(df[col].to_numpy(), df.won.to_numpy(), df.race_id.to_numpy())
    ll0 = race_log_loss(1 / df.field.to_numpy(), df.won.to_numpy(), df.race_id.to_numpy())
    return {"log_loss": ll, "r2": 1 - ll / ll0}


def walk_forward(df: pd.DataFrame, first_train: str, first_test: str, l2: float = 1.0,
                 features: list[str] = FEATURES):
    seasons = sorted(df.season.unique())
    tests = [s for s in seasons if s >= first_test]
    oos = []
    rows = []
    for season in tests:
        train = df[(df.season >= first_train) & (df.season < season)]
        test = df[df.season == season].copy()
        a = ConditionalLogit(l2=l2).fit(train[features], train.race_id, train.won)
        test["p_model"] = a.predict(test[features], test.race_id)
        test["p_combined"] = np.nan
        alpha = gamma = np.nan
        if oos:
            prior = pd.concat(oos)
            Xb = np.log(prior[["p_model", "p_market"]].to_numpy())
            b = ConditionalLogit(l2=0.0, standardize=False).fit(Xb, prior.race_id, prior.won)
            alpha, gamma = b.coef_
            test["p_combined"] = b.predict(np.log(test[["p_model", "p_market"]].to_numpy()), test.race_id)
        oos.append(test)
        row = {"season": season, "races": test.race_id.nunique(), "train_races": train.race_id.nunique(),
               "alpha": alpha, "gamma": gamma}
        for col in ("p_market", "p_model", "p_combined"):
            if test[col].notna().all():
                s = score(test, col)
                row[f"{col}_ll"], row[f"{col}_r2"] = s["log_loss"], s["r2"]
        rows.append(row)
        log.info("%s  market R² %.4f  model R² %.4f  combined R² %s", season, row["p_market_r2"],
                 row["p_model_r2"], f"{row['p_combined_r2']:.4f}" if "p_combined_r2" in row else "-")
    return pd.DataFrame(rows), pd.concat(oos), a


def market_increment(df: pd.DataFrame, first_train: str, first_test: str, l2: float = 1.0,
                     features: list[str] = FEATURES) -> dict:
    """Walk-forward log-likelihood gain per race of log pMarket + features over log pMarket alone."""
    seasons = [s for s in sorted(df.season.unique()) if s >= first_test]
    ll = {}
    for name, cols in (("market", []), ("market+features", features)):
        out = []
        for season in seasons:
            train = df[(df.season >= first_train) & (df.season < season)]
            test = df[df.season == season]
            X = lambda d: np.c_[np.log(d.p_market.to_numpy()), d[cols].to_numpy()]
            m = ConditionalLogit(l2=l2).fit(X(train), train.race_id, train.won)
            out.append(test.assign(p=m.predict(X(test), test.race_id)))
        o = pd.concat(out)
        _, starts = _groups(o.race_id.to_numpy())
        ll[name] = np.add.reduceat(o.won.to_numpy() * np.log(np.clip(o.p.to_numpy(), 1e-12, 1)), starts)
    gain = ll["market+features"] - ll["market"]
    return {"seasons": f"{seasons[0]}–{seasons[-1]}", "races": len(gain), "gain_per_race": gain.mean(),
            "se": gain.std() / np.sqrt(len(gain)), "z": gain.mean() / (gain.std() / np.sqrt(len(gain)))}


def covered_starters(df: pd.DataFrame, min_field: int = 4) -> pd.DataFrame:
    """Each race cut down to its starters with trackwork, where the winner is one of them and at least
    `min_field` are. HKJC only keeps trackwork for horses still in training, so few races have it for every
    starter; under the logit, who wins among a subset of the field follows the same model with that smaller
    choice set, so tests on these subsets stay fair to the market. Caveat: every covered horse is still
    racing today, so these are survivors' races."""
    sub = df[df.tw_covered]
    g = sub.groupby("race_id")
    return sub[(g.won.transform("sum") > 0.99) & (g.horse_id.transform("size") >= min_field)]


def trackwork_increment(df: pd.DataFrame, l2: float = 1.0,
                        first_test: str = "2025-01") -> tuple[list[dict], pd.DataFrame]:
    """Does trackwork add to the market among covered starters? Two tests, since covered races only become
    common from 2024/25: a walk-forward one month at a time from `first_test` (each month fitted on every
    covered race before it), and an in-sample likelihood-ratio test of all trackwork features at once."""
    from scipy.stats import chi2

    sub = covered_starters(df).copy()
    if sub.empty:
        return [], pd.DataFrame()
    sub["month"] = sub.race_date.dt.to_period("M")
    X = lambda d, cols: np.c_[np.log(d.p_market.to_numpy()), d[cols].to_numpy()]

    def race_ll(model, d, cols):
        _, starts = _groups(d.race_id.to_numpy())
        p = model.predict(X(d, cols), d.race_id)
        return np.add.reduceat(d.won.to_numpy() * np.log(np.clip(p, 1e-12, 1)), starts)

    rows = []
    months = [m for m in sorted(sub.month.unique()) if m >= pd.Period(first_test)]
    for name, cols in (("trackwork", TRACKWORK_FEATURES),
                       ("all but trackwork", [f for f in FEATURES if f not in TRACKWORK_FEATURES])):
        gain = []
        for m in months:
            train, test = sub[sub.month < m], sub[sub.month == m]
            fit = lambda c: ConditionalLogit(l2=l2).fit(X(train, c), train.race_id, train.won)
            gain.append(race_ll(fit(cols), test, cols) - race_ll(fit([]), test, []))
        gain = np.concatenate(gain)
        se = gain.std() / np.sqrt(len(gain))
        rows.append({"features": f"{name} (covered starters, monthly)", "seasons": f"{months[0]}–{months[-1]}",
                     "races": len(gain), "gain_per_race": gain.mean(), "se": se, "z": gain.mean() / se})
    m0 = ConditionalLogit(l2=0.0).fit(X(sub, []), sub.race_id, sub.won)
    m1 = ConditionalLogit(l2=0.0).fit(X(sub, TRACKWORK_FEATURES), sub.race_id, sub.won)
    lr = 2 * (race_ll(m1, sub, TRACKWORK_FEATURES).sum() - race_ll(m0, sub, []).sum())
    rows.append({"features": f"trackwork in-sample LR test: chi² {lr:.1f} on {len(TRACKWORK_FEATURES)} df, "
                             f"p = {chi2.sf(lr, len(TRACKWORK_FEATURES)):.2f}",
                 "seasons": f"{sub.season.min()}–{sub.season.max()}", "races": sub.race_id.nunique()})
    coefs = pd.DataFrame({"feature": ["log p_market"] + TRACKWORK_FEATURES, "coef": m1.coef_, "se": m1.se_,
                          "z": m1.coef_ / m1.se_})
    return rows, coefs


def gear_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per season: share of starters whose gear is known (they have a profile form row) and how often the
    gear-change flags fire among those."""
    known = df[df.gear_known]
    out = df.groupby("season").agg(starters=("horse_id", "size"), gear_known=("gear_known", "mean"))
    rates = known.groupby("season")[["gear_first", "gear_vis_first", "gear_vis_off", "gear_off"]].mean()
    return out.join(rates).reset_index()


def betting(df: pd.DataFrame, col: str, edges=(1.0, 1.1, 1.2, 1.3, 1.5)) -> pd.DataFrame:
    """Flat $1 win bets wherever p × final odds exceeds the edge. HK odds include the stake."""
    out = []
    for e in edges:
        bets = df[df[col] * df.win_odds > e]
        ret = (bets.win_odds * bets.won).sum()
        out.append({"edge": e, "bets": len(bets), "winners": round(bets.won.sum(), 1),
                    "roi": ret / len(bets) - 1 if len(bets) else np.nan})
    return pd.DataFrame(out)


def calibration(df: pd.DataFrame, col: str, bins=(0, .02, .05, .1, .15, .2, .3, .45, 1)) -> pd.DataFrame:
    b = pd.cut(df[col], bins)
    return df.groupby(b, observed=True).agg(runners=("won", "size"), predicted=(col, "mean"),
                                            actual=("won", "mean")).reset_index()


def report(results, oos, final, path: Path, increment: list[dict] | None = None,
           tw_coefs: pd.DataFrame | None = None, coverage: pd.DataFrame | None = None,
           hv_offset: float | None = None) -> str:
    pooled = oos[oos.p_combined.notna()]
    lines = ["# Walk-forward evaluation", "",
             "Pseudo-R² is McFadden's, against a model that gives every runner 1/field. Higher is better.",
             "Stage B (combined) starts from the second test season.", "",
             "## By test season", "",
             results.to_markdown(index=False, floatfmt=".4f"), "",
             f"## Pooled over {pooled.season.nunique()} seasons with a combined model "
             f"({pooled.race_id.nunique()} races)", ""]
    pooled_rows = [{"model": c, **score(pooled, c)} for c in ("p_market", "p_model", "p_combined")]
    lines += [pd.DataFrame(pooled_rows).to_markdown(index=False, floatfmt=".4f"), "",
              "## Market increment", "",
              "Walk-forward one-stage logit on log market probability plus every feature, against log market "
              "probability alone. Gain is in log-likelihood per race; above zero means the features know "
              "something the final odds don't. Each row is a separate test with that group of features.", "",
              pd.DataFrame(increment).to_markdown(index=False, floatfmt=".5f") if increment else "(not run)", "",
              "Trackwork exists only for horses still in training when it was scraped, so trackwork is not in the "
              "stage A model and is tested only among covered starters (each race cut down to them). "
              "In-sample weights with the market, standardised features:", "",
              tw_coefs.to_markdown(index=False, floatfmt=".3f") if tw_coefs is not None and len(tw_coefs)
              else "(no trackwork)", "",
              "Gear comes from horse profile pages. Share of starters with known gear by season, and how often "
              "each change flag fires among them:", "",
              coverage.to_markdown(index=False, floatfmt=".3f") if coverage is not None else "(not run)", "",
              f"Happy Valley figures run {hv_offset:.2f} points above Sha Tin turf figures for the same horse "
              "(from course switches before the first test season); the course aptitude features take this off "
              "HV runs." if hv_offset is not None else "", "",
              "## Stage A weights (last fit, standardised features)", "",
              pd.DataFrame({"feature": FEATURES, "coef": final.coef_, "se": final.se_,
                            "z": final.coef_ / final.se_}).to_markdown(index=False, floatfmt=".3f"), "",
              "## Calibration of the combined model (pooled)", "",
              calibration(pooled, "p_combined").to_markdown(index=False, floatfmt=".4f"), "",
              "## Flat-stake betting at final odds (pooled)", "",
              "Bets every runner where p × final odds > edge. Final odds aren't the odds you get when you bet, "
              "so treat this as a rough check only.", "",
              "Combined model:", "", betting(pooled, "p_combined").to_markdown(index=False, floatfmt=".3f"), "",
              "Stage A model alone:", "", betting(pooled, "p_model").to_markdown(index=False, floatfmt=".3f"), ""]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/hkracing.db")
    ap.add_argument("--first-train", default="2017/18", help="2016/17 is history warm-up only")
    ap.add_argument("--first-test", default="2019/20")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--report", default="data/reports/walk_forward.md")
    ap.add_argument("--skip-speed", action="store_true", help="reuse the speed_figures table as it is")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conn = connect(args.db)
    if not args.skip_speed:
        build_speed(conn)
    df = build_features(conn, args.first_test)
    df = df.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    results, oos, final = walk_forward(df, args.first_train, args.first_test, args.l2)
    base = [f for f in FEATURES if f not in TRIAL_FEATURES]
    no_sg = [f for f in FEATURES if f not in SURFACE_FEATURES + GEAR_FEATURES + COURSE_FEATURES]
    no_course = [f for f in FEATURES if f not in COURSE_FEATURES]
    groups = {"all": FEATURES, "all but trials": base, "all but course": no_course,
              "all but surface, gear and course": no_sg, "sectionals": SECTIONAL_FEATURES,
              "trials": TRIAL_FEATURES, "surface": SURFACE_FEATURES, "gear": GEAR_FEATURES,
              "course": COURSE_FEATURES}
    groups.update({f"course: {c}": [c] for c in COURSE_FEATURES})
    increment = [{"features": name, **market_increment(df, args.first_train, results.season.iloc[1], args.l2, cols)}
                 for name, cols in groups.items()]
    tw_rows, tw_coefs = trackwork_increment(df, args.l2)
    increment += tw_rows
    print(report(results, oos, final, Path(args.report), increment, tw_coefs, gear_coverage(df),
                 venue_offset(df, args.first_test)))
    oos[["race_id", "horse_id", "season", "won", "win_odds", "p_market", "p_model", "p_combined"]] \
        .to_csv(Path(args.report).with_name("walk_forward_predictions.csv.gz"), index=False)


if __name__ == "__main__":
    main()
