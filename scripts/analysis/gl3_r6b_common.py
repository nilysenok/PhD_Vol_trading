# -*- coding: utf-8 -*-
"""Round 6b, h=5: pure evaluation helpers (no I/O, no data readers).

Annual QLIKE, stress rows, per-ticker comparison with the final, one-sided binomial p
(U5), admission by Zh-188 item 1 (U2, U3, U5) and the leak verdict (U18).
"""
from math import comb
import numpy as np
import pandas as pd

YEARS = (2017, 2018, 2019, 2020, 2021)
EPS = 1e-12
N_TICKERS_WIN = 10


def qlike_terms(y, p):
    r = np.asarray(y, float) / np.asarray(p, float)
    return r - np.log(r) - 1


def score(y, p, year, m2020, mrvi):
    """Same keys as Data.score5: five annual means, their equal-weight mean, two stress rows and ratios."""
    y = np.asarray(y, float); p = np.asarray(p, float); year = np.asarray(year)
    assert np.isfinite(p).all() and (p > 0).all(), 'forecast must be finite and positive'
    l = qlike_terms(y, p)
    out = {str(k): float(l[year == k].mean()) for k in YEARS}
    out['mean_QLIKE'] = float(np.mean([out[str(k)] for k in YEARS]))
    m2020 = np.asarray(m2020, bool); mrvi = np.asarray(mrvi, bool)
    out['stress2020'] = float(l[m2020].mean()); out['stressRVI'] = float(l[mrvi].mean())
    r = y / p
    out['ratio_stress2020'] = float(r[m2020].mean()); out['ratio_stressRVI'] = float(r[mrvi].mean())
    return out


def ticker_means(y, p, year, ticker):
    """U5: QLIKE of a ticker = equal-weight mean of its five annual means."""
    f = pd.DataFrame({'l': qlike_terms(y, p), 'year': np.asarray(year), 'ticker': np.asarray(ticker)})
    return f.groupby(['ticker', 'year']).l.mean().groupby('ticker').mean()


def tickers_better(cand, final):
    """Number of tickers where cand is strictly better (lower) than final, and the number compared."""
    c, f = cand.align(final, join='inner')
    return int((c < f - EPS).sum()), int(len(c))


def binom_sf(k, n=17, p=.5):
    """One-sided P(X >= k | n, p)."""
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(int(k), n + 1)))


def admission(sc, base, k_tickers, eligible=True):
    """Zh-188 item 1: >=3 of 5 years better (strict, 1e-12), both stress rows not worse, mean better, >=10 tickers better."""
    n = sum(sc[str(y)] < base[str(y)] - EPS for y in YEARS)
    stress = all(sc[k] <= base[k] + EPS for k in ('stress2020', 'stressRVI'))
    mean = sc['mean_QLIKE'] < base['mean_QLIKE'] - EPS
    tick = k_tickers >= N_TICKERS_WIN
    return dict(better_years=int(n), stress_pass=bool(stress), better_mean=bool(mean), tickers_pass=bool(tick),
                admitted=bool(eligible and n >= 3 and stress and mean and tick))


def leak_verdict(archive_mean, pit_mean, final_mean):
    """U18: the gain disappears if the PIT mean is not better than the final, or the PIT gain is below half of the archive gain.
    Without an archive gain there is nothing to lose: no leak is claimed."""
    ga, gp = final_mean - archive_mean, final_mean - pit_mean
    if ga <= EPS:
        return False, 'no_archive_gain'
    if gp <= EPS:
        return True, 'pit_not_better_than_final'
    if gp < ga / 2:
        return True, 'pit_gain_below_half_of_archive'
    return False, 'gain_kept'
