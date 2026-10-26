#!/usr/bin/env python3
"""Приложение В: тест «будущее не влияет» (правило tests/leak_detector.py: входы после даты t -> шум, всё на датах <= t обязано совпасть
побайтно; даты t — leak_detector.DATES) для данных 20 тикеров (TGKA, MSNG, TRMK вне 17), собранных scripts/analysis/prilozhenie_v.py data:
  (а) RRV-признаки и цели features_pit_v3_rrv_20 — d172_rrv_features.new_columns на ряду rrv_1850 с шумом после t (проверка test_d172_rrv_leak.check);
  (б) ночной гэп и прочие колонки intraday1850_20 у трёх новых тикеров — d13_lib.intraday_one на барах с шумом после t;
  (в) слагаемые HAR-X⁺ цели h=1 на составе 20 (в т. ч. RV рынка = средняя RV_m по составу) — d22_harxp.leak_test (все числовые входы > t -> бутстрап);
  (г) признаки v3 новых тикеров — tests/test_d121_leak28.py (тот же m0_pit_lib.assemble, что features_pit_v3_28).
Самопроверки: подложенная утечка (значение следующего дня) обязана ловиться в (а), (б), (в).
Запуск: python3 tests/test_prilozhenie_v_leak.py [--skip-d121] [--log файл]; код выхода 0 — всё в порядке, 1 — утечка, 2 — самопроверка не сработала."""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'scripts' / 'analysis'))
from leak_detector import DATES  # noqa: E402
import test_d172_rrv_leak as TR  # noqa: E402
import d172_rrv_features as F172  # noqa: E402
import prilozhenie_v as PV  # noqa: E402


def eqn(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.array_equal(a, b, equal_nan=True))


def rrv_check(log):
    keys = pd.read_parquet(PV.F_RRV20, columns=['date', 'ticker'])
    keys['date'] = pd.to_datetime(keys['date'])
    rrv = F172.load_rrv(PV.RRV_RAW)
    ok = True
    for t in DATES:
        n, bad_f, bad_t, post = TR.check(keys, rrv, t)
        good = not bad_f and not bad_t and post > 0
        ok &= good
        log(f'(а) RRV 20 тикеров t={t}: строк <= t {n}, расхождений признаков {bad_f}, целей {bad_t}, колонок с изменением после t {post} -> {"PASS" if good else "FAIL"}')
    _, bad_f, _, _ = TR.check(keys, rrv, DATES[0], leak=True)
    caught = bool(bad_f)
    log(f'(а) самопроверка rrv_pk_d = ln RRV(t+1): {"поймана" if caught else "НЕ поймана"}')
    return ok, caught


def noisy_bars(c, t, seed):
    c = c.copy()
    m = pd.to_datetime(c['begin']).dt.normalize() > pd.Timestamp(t)
    rng = np.random.default_rng(seed)
    f = rng.lognormal(0, 0.05, m.sum())
    for col in ('open', 'high', 'low', 'close'):
        c.loc[m, col] = c.loc[m, col].values * f
    return c


def gap_check(log):
    import d13_lib as D13
    ok, caught = True, True
    for tk in PV.NEW3:
        bars = pd.read_parquet(PV.BARS / f'{tk}.parquet')
        base = D13.intraday_one(bars, tk)
        for i, t in enumerate(DATES):
            pert = D13.intraday_one(noisy_bars(bars, t, 7 + i), tk)
            m = base.merge(pert, on=['date', 'ticker'], suffixes=('', '_p'), how='outer')
            pre = pd.to_datetime(m['date']) <= pd.Timestamp(t)
            bad = [c for c in D13.INTRA_COLS if not eqn(m.loc[pre, c], m.loc[pre, c + '_p'])]
            changed = sum(not eqn(m.loc[~pre, c], m.loc[~pre, c + '_p']) for c in D13.INTRA_COLS)
            good = not bad and changed > 0
            ok &= good
            log(f'(б) intraday {tk} t={t}: дат <= t {int(pre.sum())}, расхождений {bad}, колонок с изменением после t {changed} -> {"PASS" if good else "FAIL"}')
        t = DATES[0]
        pert = D13.intraday_one(noisy_bars(bars, t, 7), tk)
        lk = lambda d: d['gap_on'].shift(-1)  # утечка: гэп следующего дня
        m = base.assign(lk=lk(base)).merge(pert.assign(lk=lk(pert)), on=['date', 'ticker'], suffixes=('', '_p'))
        pre = pd.to_datetime(m['date']) <= pd.Timestamp(t)
        c = not eqn(m.loc[pre, 'lk'], m.loc[pre, 'lk_p'])
        caught &= c
        log(f'(б) самопроверка {tk} gap_on(t+1): {"поймана" if c else "НЕ поймана"}')
    return ok, caught


def terms_check(log):
    import harx as HX
    import d22_harxp as P
    tmp = Path(tempfile.mkdtemp(prefix='prilV_leak_'))
    P.FEAT, P.INTRA, P.OUT = PV.F_RRV20, PV.INTRA20, tmp  # leak_test пишет временный файл гэпа в P.OUT
    A, harj = HX.load(PV.F_RRV20)
    A = P.add_targets(A)
    names = P.terms_for('1')
    P.TERMS['_leak_next'] = lambda A_: A_.groupby('ticker')['rv_d'].shift(-1).astype('float64')  # самопроверка: RV_d следующей строки
    r = P.leak_test(A, names + ['_leak_next'], cuts=tuple(DATES))
    r.to_csv(tmp / 'leak_terms.csv', index=False)
    real = r[r.term != '_leak_next']
    ok = int(real.mismatch.sum()) == 0
    for t, g in real.groupby('t'):
        log(f'(в) слагаемые HAR-X⁺ h=1, состав 20, t={t}: слагаемых {len(g)}, строк <= t {int(g.n.iloc[0])}, расхождений {int(g.mismatch.sum())} '
            f'(ln_rvm_mkt: {int(g[g.term == "ln_rvm_mkt"].mismatch.sum())}) -> {"PASS" if g.mismatch.sum() == 0 else "FAIL"}')
    caught = int(r[r.term == '_leak_next'].mismatch.sum()) > 0
    log(f'(в) самопроверка RV_d(t+1): {"поймана" if caught else "НЕ поймана"} ({int(r[r.term == "_leak_next"].mismatch.sum())} строк)')
    return ok, caught


def d121_check(log):
    env = dict(os.environ, TMPDIR=tempfile.mkdtemp(prefix='prilV_d121_'))
    p = subprocess.run([sys.executable, '-W', 'ignore', str(HERE / 'test_d121_leak28.py')], capture_output=True, text=True, env=env)
    for s in (p.stdout + p.stderr).strip().splitlines()[-12:]:
        log('(г) ' + s)
    log(f'(г) tests/test_d121_leak28.py: код выхода {p.returncode} -> {"PASS" if p.returncode == 0 else "FAIL"}')
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-d121', action='store_true')
    ap.add_argument('--log', default=None)
    a = ap.parse_args()
    fh = open(a.log, 'w') if a.log else None

    def log(s):
        print(s, flush=True)
        if fh:
            fh.write(s + '\n'); fh.flush()
    log(f'даты t: {DATES} (tests/leak_detector.py)')
    o1, c1 = rrv_check(log)
    o2, c2 = gap_check(log)
    o3, c3 = terms_check(log)
    o4 = True if a.skip_d121 else d121_check(log)
    ok, caught = o1 and o2 and o3 and o4, c1 and c2 and c3
    log(f'ИТОГО: утечек {"нет" if ok else "ЕСТЬ"}; самопроверки {"OK" if caught else "СБОЙ"}')
    return 0 if ok and caught else (2 if not caught else 1)


if __name__ == '__main__':
    sys.exit(main())
