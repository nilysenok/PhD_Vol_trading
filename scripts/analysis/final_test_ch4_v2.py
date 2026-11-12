#!/usr/bin/env python3
"""Второе открытие теста главы 4 (раунд стратегий v2) — с замком. Владелец 26.10, 22:25, п. 4.

Постановка — docs/zadaniya/2026-10-26_razrabotka_strategiy_v2.md; черновик дизайна —
docs/otchety/FINAL_DESIGN_strategies_v2_chernovik.md (поля, правило выбора С8, сетки); спецификация —
models/designs/ch4_v2/ch4_v2_sborka.yaml. Первое открытие теста главы 4 было 26.10 10:32 (до исправления выравнивания
входов σ); это — второе, одно, по «да» владельца.

Режимы
  python3 scripts/analysis/final_test_ch4_v2.py --selftest [--boot-reps 10000]
      Псевдотест на годах 2018–2021 (окна валидации/калибровки заморожены на 2017): σ — data/regen/ch4_dev/ (SHA против
      data/regen/ch4_dev/SHA256.txt), бары, RV/RVI, ставка — с фильтром date ≤ 2021-12-31 при чтении; ни одного года ≥ 2022,
      test_locked/ не читается, замок не трогается. Дизайны — заглушки spec.selftest_zaglushki (опорные точки, не выбор).
      Выход — data/fastbt/dev_v2/selftest_ch4_v2/.
  python3 scripts/analysis/final_test_ch4_v2.py --scope test
      ОДИН запуск после «да». Порядок (каждый шаг — до следующего; отказ на шагах 1–3 открытие не тратит):
        1) спецификация: status = zamorozheno, «да» владельца, по одному дизайну на подход (B, C, D, MSP), все SHA заполнены —
           иначе отказ до чтения данных (код 2);
        2) замок data/regen/test_locked/ch4_v2/OPENED_CH4_V2.json — если есть, отказ (второе открытие v2 было);
        3) SHA-256: FINAL_DESIGN_strategies_v2.md; σ открытия 21:36 h=1, 5 (гибрид, HAR-J (б)) — против спецификации И
           r2_A_sborka.yaml: sobrano; features_pit_v3, бары, ставка ЦБ; код (spec.kod) и построители дизайнов (kod_sha256);
        4) замок пишется (O_CREAT|O_EXCL) ДО чтения рядов теста;
        5) σ теста — тем же кодом, что разработка S0 (h2_d0_ch4.sigma_ch3, naive_from_rv), срез 03.02.2026; A — fastbt.s9._job
           (scope test, окна заморожены на 2018–2021); надстройки — к позиции A, как в разработке;
        6) метрики — fastbt.s9.metrics, Δ к A — fastbt.s9_stats.lw_sharpe_diff (Ледуа–Вольф, блок 5, 10 000), как в разработке;
           окна 2022–2026 и 2025–01.2026; без правила закрытия биржи и с ним — рядом (столбец pravilo);
           H3 и табл. 4.9 — fastbt.bench после #242 (код не раньше 642f084, иначе отказ); перед счётом — проверка сдвига
           каждого σ-ряда теста, включая наивную √RV_w для C* (правило S0; не прошёл — отказ);
        7) таблицы → docs/tables/strat_test_v2/, ряды → data/fastbt/test_v2/; замок дописывается (status «завершено», SHA выходов).

Считаются только дизайны spec.dizayny (по одному на подход) + A и таблица H2 внутри выбранной точки spec.h2.dizayn
(гибрид / HAR-J (б) / наивная — одной таблицей, без выбора). Комиссия (владелец 22:45): основная 0,04 % за сторону «всё
включено» — фактический тариф Т-Банка «Инвестор»; чувствительность — 0,035 / 0,05 / 0,06 % (без p). «Ставка комиссии изменена
с 0,05 % (FINAL_DESIGN_strategies v1) на 0,04 % до второго открытия, по фактическому тарифу, не по результатам разработки.»
Семьи С9 «выбор по валидации» — ключ tochki (+ otsev), Engine.select на замороженном окне. Все строки — «до аудита», пока FA не закрыл аудит (столбец pometka; правило пересчёта — FINAL v2, раздел 6).

Семьи (встроенная математика = fastbt.s9 / fastbt.s9_stats; дизайн может указать kod: '<файл>.py:<функция>' построителя раунда —
тогда вызывается он, SHA сверяется):
  C1  {W, p, razgon}   ранг σ в прошлом окне W (текущий бар не входит) — затвор по категории (тренд: R > 1−p; контртренд: R < p;
                       диапазон: 1−p ≤ R ≤ p), NaN — открыт; razgon — мин. число значений окна (W504: растущее окно 252 → 504, Ъ)
  C2  {W, p1, p2}      частичная экспозиция 1 → 0,5 → 0 по двум порогам
  Cmed {W, p, med}     то же на x = σ / медиана σ за med прошлых дней (реакция на всплеск, а не на уровень)
  C*1 {W, p} / C*2 {W, p1, p2} / C2W {W, p1, p2}  новые точки C раунда (Ъ 22:30, PR #240) — fastbt.s9.c_dop_gates
                       по разделу dop fastbt/designs/s9/setka_C.yaml; C* — ранг σ_pred / √RV_w (наивная S0; в тесте строится
                       в открытии и проходит проверку сдвига)
  D1 {W, p}            закрыт при R > p;  D2 {W, p, g} — гистерезис p−g / p;  D4 {W, p, span} — ранг EMA σ
  Dcont {W, gamma, cap} вес min(cap, (медиана W / σ)^γ)
  B  {cap, band, W}    вес min(cap, медиана W / σ), полоса бездействия band
  P1 {baza, sigma}     веса 1/σ_i между тикерами (по дате нормированы к 1) поверх позиции baza (A или дизайн подхода)
  P2 {sostav}          равные доли капитала портфелей sostav
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

MAIN = Path(os.environ.get('FASTBT_MAIN', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading'))
REGEN = MAIN / 'data' / 'regen'
DEV_CUT = pd.Timestamp('2021-12-31')
PODHODY = ('B', 'C', 'D', 'MSP')
DEV_FILES = {   # метка: (файл data/regen/ch4_dev, столбец, тег горизонта s9)
    'hyb_h1': ('sigma_hyb_h1_dev.parquet', 'sigma_pred_h1', 'h1'),
    'hyb_h5': ('sigma_hyb_h5_dev.parquet', 'sigma_pred_h5', 'h5'),
    'harj_b_h1': ('sigma_harj_b_h1_dev.parquet', 'sigma_pred_h1', 'h1'),
    'harj_b_h5': ('sigma_harj_b_h5_dev.parquet', 'sigma_pred_h5', 'h5'),
    'naive': ('sigma_naive_dev.parquet', 'sigma_pred_h1', 'h1'),
}
POMETKA = 'до аудита'
VHODY = ('features', 'bary', 'stavka', 'mcftr', 'rgbitr')     # файлы данных теста (кроме σ) — SHA в спецификации
KOD_NE_RANSHE = '642f084'      # исправления аудита FA: #238 Н1, #239 Н2, #241 Н3, #242 H3-выравнивание


class Stop(SystemExit):
    """Отказ: код 2, сообщение в stderr."""

    def __init__(self, msg):
        super().__init__(2)
        self.msg = msg
        print(f'СТОП: {msg}', file=sys.stderr, flush=True)


@dataclasses.dataclass
class Puti:
    """Все пути одним объектом — тесты подменяют их временной папкой (scratchpad / tmp_path)."""
    root: Path = ROOT
    main: Path = MAIN
    spec: Path = ROOT / 'models' / 'designs' / 'ch4_v2' / 'ch4_v2_sborka.yaml'
    r2spec: Path = ROOT / 'models' / 'designs' / 'r2' / 'r2_A_sborka.yaml'
    locked: Path = REGEN / 'test_locked'
    lock: Path = REGEN / 'test_locked' / 'ch4_v2' / 'OPENED_CH4_V2.json'
    dev: Path = REGEN / 'ch4_dev'
    out_tables_test: Path = ROOT / 'docs' / 'tables' / 'strat_test_v2'
    out_data_test: Path = MAIN / 'data' / 'fastbt' / 'test_v2'
    out_selftest: Path = MAIN / 'data' / 'fastbt' / 'dev_v2' / 'selftest_ch4_v2'


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now() -> str:
    return time.strftime('%F %T')


def git_head(root) -> str:
    try:
        return subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:          # noqa: BLE001
        return ''


# ======================= 1–4: спецификация, замок, SHA =======================

def load_spec(P: Puti) -> dict:
    return yaml.safe_load(Path(P.spec).read_text()) or {}


def check_frozen(spec: dict) -> None:
    """Шаг 1: без заморозки — отказ до чтения данных."""
    bad = []
    if spec.get('status') != 'zamorozheno':
        bad.append(f'status = {spec.get("status")!r}, нужно zamorozheno')
    if not str(spec.get('da_vladeltsa') or '').strip():
        bad.append('нет «да» владельца (da_vladeltsa)')
    diz = spec.get('dizayny') or {}
    for k in PODHODY:
        d = diz.get(k) or {}
        if not d.get('semya'):
            bad.append(f'нет дизайна подхода {k} (один дизайн на подход)')
    extra = sorted(set(diz) - set(PODHODY))
    if extra:
        bad.append(f'лишние подходы в dizayny: {extra} (один дизайн на подход)')
    if not (spec.get('final_v2') or {}).get('sha256'):
        bad.append('нет SHA FINAL_DESIGN_strategies_v2.md')
    for grp in ('sigma_test',):
        for lab, v in ((spec.get('vhody') or {}).get(grp) or {}).items():
            if not v.get('sha256'):
                bad.append(f'нет SHA σ {lab}')
    for k in VHODY:
        if not ((spec.get('vhody') or {}).get(k) or {}).get('sha256'):
            bad.append(f'нет SHA входа {k}')
    for p, s in (spec.get('kod') or {}).items():
        if not s:
            bad.append(f'нет SHA кода {p}')
    if not spec.get('fastbt_sha_vse'):
        bad.append('нет fastbt_sha_vse (SHA всего кода fastbt)')
    for k, d in diz.items():
        if (d or {}).get('kod') and not d.get('kod_sha256'):
            bad.append(f'{k}: kod без kod_sha256')
    if bad:
        raise Stop('спецификация не заморожена — тест закрыт: ' + '; '.join(bad))


def check_lock_free(P: Puti) -> None:
    """Шаг 2: одно открытие."""
    if Path(P.lock).exists():
        raise Stop(f'{P.lock} уже есть — второе открытие v2 было: {Path(P.lock).read_text()[:300]}')


def verify_sha(spec: dict, P: Puti) -> dict:
    """Шаг 3: SHA FINAL v2, σ теста (против спецификации и r2_A_sborka: sobrano), входов, кода. Возвращает {путь: SHA}."""
    got, bad = {}, []

    def one(path, want, what):
        path = Path(path)
        if not path.exists():
            bad.append(f'{what}: нет файла {path}')
            return
        g = sha(path)
        got[str(path)] = g
        if g != want:
            bad.append(f'{what}: SHA {g[:16]}… ≠ {str(want)[:16]}…')

    fv = spec['final_v2']
    one(P.root / fv['put'], fv['sha256'], 'FINAL v2')
    sob = (yaml.safe_load(Path(P.r2spec).read_text()) or {}).get('sobrano') or {}
    for lab, v in spec['vhody']['sigma_test'].items():
        if sob.get(v['fayl']) != v['sha256']:
            bad.append(f'σ {lab}: SHA спецификации ≠ r2_A_sborka.yaml: sobrano[{v["fayl"]}]')
        one(P.locked / v['fayl'], v['sha256'], f'σ {lab}')
    for k in VHODY:
        v = spec['vhody'][k]
        one(P.main / v['fayl'], v['sha256'], f'вход {k}')
    for p, s in (spec.get('kod') or {}).items():
        one(P.root / p, s, f'код {p}')
    agg = fastbt_agg(P.root)
    got['fastbt/**'] = agg
    if agg != spec.get('fastbt_sha_vse'):
        bad.append(f'код fastbt (все *.py, *.yaml): SHA {agg[:16]}… ≠ fastbt_sha_vse {str(spec.get("fastbt_sha_vse"))[:16]}…')
    for k, d in (spec.get('dizayny') or {}).items():
        if (d or {}).get('kod'):
            one(P.root / d['kod'].split(':')[0], d['kod_sha256'], f'построитель {k}')
    if bad:
        raise Stop('SHA не сходятся — открытие не начато: ' + '; '.join(bad))
    return got


def acquire_lock(P: Puti, info: dict) -> None:
    """Шаг 4: замок до чтения рядов теста; O_EXCL — два одновременных запуска не пройдут оба."""
    Path(P.lock).parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(P.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise Stop(f'{P.lock} уже есть — второе открытие v2 было') from None
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(dict(info, status='начато', vremya=now()), ensure_ascii=False, indent=1))


def finish_lock(P: Puti, extra: dict, status: str = 'завершено') -> None:
    lk = Path(P.lock)
    d = json.loads(lk.read_text())
    d.update(extra, status=status, vremya_konets=now())
    os.chmod(lk, 0o644)
    lk.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    os.chmod(lk, 0o444)


def require_base(root, commit) -> None:
    """Код — не раньше commit (исправления аудита FA #238, #239, #241, #242 — 642f084)."""
    r = subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', str(commit), 'HEAD'],
                       capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        raise Stop(f'код старше {commit} (нет исправлений аудита FA) — отказ')


def precheck_test(P: Puti) -> tuple:
    """Шаги 1–3 (без чтения данных теста); возвращает (spec, SHA)."""
    spec = load_spec(P)
    check_frozen(spec)
    check_lock_free(P)
    for c in {KOD_NE_RANSHE, spec.get('kod_ne_ranshe') or KOD_NE_RANSHE}:   # спецификация может только поднять порог
        require_base(P.root, c)
    return spec, verify_sha(spec, P)


# ======================= чтение с жёстким срезом =======================

def read_capped(path, cols, cut) -> pd.DataFrame:
    """Фильтр pyarrow date ≤ cut при чтении, затем маска и assert — до любых других операций."""
    cut = pd.Timestamp(cut)
    d = pd.read_parquet(path, columns=cols, filters=[('date', '<=', cut)])
    d['date'] = pd.to_datetime(d['date'])
    d = d[d['date'] <= cut].reset_index(drop=True)
    assert len(d) == 0 or d['date'].max() <= cut, f'{path}: дата > {cut.date()}'
    return d


def dev_sha_table(P: Puti) -> dict:
    out = {}
    for line in Path(P.dev / 'SHA256.txt').read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            out[parts[1]] = parts[0]
    return out


def verify_dev(P: Puti) -> dict:
    tab = dev_sha_table(P)
    got = {}
    for lab, (fn, _, _) in DEV_FILES.items():
        want = tab.get(f'data/regen/ch4_dev/{fn}')
        g = sha(P.dev / fn)
        if g != want:
            raise Stop(f'σ разработки {fn}: SHA {g[:16]}… ≠ SHA256.txt {str(want)[:16]}…')
        got[fn] = g
    return got


def cap_fastbt_inputs(cut) -> None:
    """Псевдотест: всё, что fastbt читает о барах, σ, событиях и ставке, — только date ≤ cut (фильтр при чтении)."""
    from fastbt import data as fd
    from fastbt import msp
    cut = pd.Timestamp(cut)

    @lru_cache(maxsize=1)
    def daily_df():
        return read_capped(fd.OHLCV, None, cut)

    @lru_cache(maxsize=1)
    def vpred_df():
        return read_capped(fd.VPRED, None, cut)

    corp0 = fd.corp_actions

    def corp_actions(t):
        c = corp0(t)
        return c[c['date'] <= cut]

    def rate_cbr():
        k = read_capped(msp.KEY_RATE, None, cut)
        return k.set_index(pd.to_datetime(k['date']))['value'].sort_index() / 100.0

    fd.daily_df, fd.vpred_df, fd.corp_actions, msp.rate_cbr = daily_df, vpred_df, corp_actions, rate_cbr
    fd.load_ticker.cache_clear()


# ======================= σ-источники =======================

def sources_dev(P: Puti, out: Path) -> tuple:
    """σ разработки S0 (≤ 2021) и RVI из features_pit_v3 (≤ 2021)."""
    src = {lab: (P.dev / fn, col, h) for lab, (fn, col, h) in DEV_FILES.items()}
    for lab, (p, col, _) in src.items():
        d = read_capped(p, ['date', 'ticker', col], DEV_CUT)
        assert d['date'].max() <= DEV_CUT
    rvi = rvi_file(P, out, DEV_CUT)
    return src, rvi


def rvi_file(P: Puti, out: Path, cut) -> Path:
    f = read_capped(P.main / 'data' / 'regen' / 'features_pit_v3.parquet', ['date', 'ticker', 'close_RVI'], cut)
    rvi = f.dropna(subset=['close_RVI']).groupby('date')['close_RVI'].first().reset_index()
    out.mkdir(parents=True, exist_ok=True)
    p = out / 'rvi_1850.parquet'
    rvi.to_parquet(p, index=False)
    return p


def sources_test(spec: dict, P: Puti, out: Path) -> tuple:
    """σ теста: √ряда открытия 21:36 (h2_d0_ch4.sigma_ch3 — как S0), наивная d0 (naive_from_rv), RVI; срез spec.test.srez."""
    import h2_d0_ch4 as H
    cut = pd.Timestamp(spec['test']['srez'])
    out.mkdir(parents=True, exist_ok=True)
    src = {}
    for lab, v in spec['vhody']['sigma_test'].items():
        d = read_capped(P.locked / v['fayl'], ['date', 'ticker', v['stolbets']], cut)
        s = H.sigma_ch3(d, v['stolbets'], v['vyhod'])
        assert not s[v['vyhod']].isna().any(), f'{lab}: NaN в σ'
        p = out / f'sigma_{lab}_test.parquet'
        s.to_parquet(p, index=False)
        src[lab] = (p, v['vyhod'], 'h1' if v['vyhod'].endswith('h1') else 'h5')
    f = read_capped(P.main / spec['vhody']['features']['fayl'], ['date', 'ticker', 'rv'], cut)
    nv = H.naive_from_rv(f.sort_values(['ticker', 'date']).reset_index(drop=True), cut=cut)
    p = out / 'sigma_naive_test.parquet'
    nv.to_parquet(p, index=False)
    src['naive'] = (p, 'sigma_pred_h1', 'h1')
    return src, rvi_file(P, out, cut)


# ======================= A (fastbt.s9._job) и движок =======================

def build_A(scope_args: tuple, workers: int) -> dict:
    """A основного дизайна (main.yaml: ансамбль + фильтр оборота) кодом fastbt.s9._job, без C (srcs пустой)."""
    from fastbt import s9
    from fastbt.data import TICKERS
    jobs = [(t, {}) + scope_args for t in TICKERS]
    t0 = time.time()
    if workers <= 1:
        res = dict(map(s9._job, jobs))
    else:
        import multiprocessing as mp
        os.environ['NUMBA_NUM_THREADS'] = '1'
        with mp.get_context('spawn').Pool(min(workers, 3)) as pool:
            res = dict(pool.imap_unordered(s9._job, jobs))
    print(f'A: {len(res)} тикеров, {time.time() - t0:.0f} с', flush=True)
    return res


def configure_s9(years, val, cut, sources: dict, rvi: Path) -> None:
    """Локальный override fastbt.s9 (общие модули не правятся): годы, окна, срез, σ-источники."""
    from fastbt import s9
    s9.YEARS = tuple(years)
    s9.VAL_START, s9.VAL_END = val
    s9.CUT = pd.Timestamp(cut)
    s9.SOURCES.clear()
    s9.SOURCES.update(sources)
    s9.RVI = rvi
    s9.load_c_dop()                          # раздел dop setka_C.yaml (min_periods, сигналы C*); √RV_w — наивная этого режима
    s9.C_DOP['rvw'] = dict(file=str(sources['naive'][0]), column=sources['naive'][1])


# ======================= проверка сдвига σ (до счёта) =======================

def shift_ts(df, col, k):
    """σ_d := σ_{d+k} по строкам тикера (k = −1 — отставание, +1 — утечка) — контроли, как S0."""
    d = df.sort_values(['ticker', 'date']).copy()
    d[col] = d.groupby('ticker')[col].shift(-k)
    return d.dropna(subset=[col]).reset_index(drop=True)


def shift_check(sources: dict, feat: Path, lo, hi, eps=0.01) -> pd.DataFrame:
    """Проверка сдвига каждого σ-ряда на [lo, hi] до счёта — правило S0 (ch4_dev_sigma.shift_row): профиль Спирмена
    r2_features.shift_table (ln σ²_d с ln RV_{d+s}, s = −2…+1, медиана по тикерам), решение r2_loyo.sdvig_ok, K = {0}:
    модельный ряд — по уровню (ε = 0,01), наивная (оконное среднее) — по приращению. Контроли −1 / +1 обязаны блокироваться.
    r2_features.LAST_DEV — локальный override на hi (сам модуль не правится)."""
    import r2_features as RF
    import r2_loyo as O
    RF.LAST_DEV = pd.Timestamp(hi)
    rv = read_capped(feat, ['date', 'ticker', 'rv'], hi)
    rows = []
    for lab, (p, col, _) in sources.items():
        d = read_capped(p, ['date', 'ticker', col], hi)
        tip = 'okno' if lab == 'naive' else 'model'
        for var, x in (('ryad', d), ('kontrol_-1', shift_ts(d, col, -1)), ('kontrol_+1', shift_ts(d, col, +1))):
            x = x[x['date'] >= pd.Timestamp(lo)]
            F = x[['date', 'ticker']].assign(ln_sigma2=np.log(x[col].astype(float).values ** 2))
            pr = RF.shift_table(F, {'B': rv}, f'{lab}:{var}').iloc[0]
            if tip == 'model':
                am, corr = pr['argmax'], {s: pr[f'rho_{s}'] for s in (-2, -1, 0, 1)}
                ok, why = O.sdvig_ok(None if am is None or am != am else int(am), (0,), corr, eps)
            else:
                am = pr['d_argmax']
                ok, why = O.sdvig_ok(None if am is None or am != am else int(am), (0,))
            rows.append(dict(ryad=lab, variant=var, tip=tip, period=f'{pd.Timestamp(lo).date()}…{pd.Timestamp(hi).date()}',
                             argmax_uroven=pr['argmax'], argmax_prirascheniya=pr['d_argmax'],
                             **{f'rho_k{s:+d}': pr[f'rho_{s}'] for s in (-2, -1, 0, 1)},
                             **{f'rho_prir_k{s:+d}': pr[f'drho_{s}'] for s in (-2, -1, 0, 1)},
                             ok=bool(ok), reshenie=why or ('по уровню' if tip == 'model' else 'по приращению')))
    t = pd.DataFrame(rows)
    t['proshel'] = np.where(t['variant'] == 'ryad', t['ok'], ~t['ok'])   # контроль «прошёл» = заблокирован
    return t


def closure_engine(e, lo, hi):
    """Правило закрытия биржи: доходность бара d тикера, чей интервал (d, следующий бар] задевает остановку торгов [lo, hi],
    не начисляется (lr = 0: позиция через закрытие не зарабатывает и не теряет). Возвращает (копию движка, {тикер: дата d})."""
    lo, hi = np.datetime64(pd.Timestamp(lo), 'D'), np.datetime64(pd.Timestamp(hi), 'D')
    one = np.timedelta64(1, 'D')
    ez = copy.copy(e)
    lr = e.lr_row.copy()
    hit = {}
    for ti, t in enumerate(e.tickers):
        dts = pd.to_datetime(e.cache[t]['dates'])
        d = dts.values.astype('datetime64[D]')
        if len(d) < 2:
            continue
        nx = d[1:]
        # календарные дни строго между баром d и следующим баром: [d+1, next−1]; правило — если они задевают [lo, hi]
        m = np.r_[np.maximum(d[:-1] + one, lo) <= np.minimum(nx - one, hi), False]
        bars = np.flatnonzero(m)
        if len(bars):
            hit[t] = [str(dts[b].date()) for b in bars]
            sel = (e.r['tick'] == ti) & np.isin(e.r['bar'], bars)
            lr[sel] = 0.0
    ez.lr_row = lr
    return ez, hit


# ======================= семьи дизайнов =======================

def _per_ticker(e, X, fn):
    T, N = X.shape
    out = np.full((T, N), np.nan)
    for ti in range(T):
        n = e.n_t[ti]
        out[ti, :n] = fn(np.ascontiguousarray(X[ti, :n]))
    return out


def _rank(e, X, W, razgon=None):
    from fastbt.s9_stats import minp, past_rank
    # по умолчанию — как С9 «как были» (s9_stats.minp: max(50, W/5)); W504 с разгоном 252 → 504 (Ъ) — razgon: 252 в ключе дизайна
    mp_ = int(razgon) if razgon else minp(int(W))
    return _per_ticker(e, X, lambda s: past_rank(s.copy(), int(W), mp_))


def _cat_gate(R, p):
    nanR = np.isnan(R)
    G = np.ones((3,) + R.shape)
    G[0] = np.where(nanR | (R > 1 - p), 1.0, 0.0)          # тренд
    G[1] = np.where(nanR | (R < p), 1.0, 0.0)              # контртренд
    G[2] = np.where(nanR | ((R >= 1 - p) & (R <= p)), 1.0, 0.0)   # диапазон
    return G


def _cat_partial(R, p1, p2):
    nanR = np.isnan(R)
    G = np.ones((3,) + R.shape)
    G[0] = np.where(R > 1 - p1, 1.0, np.where(R > 1 - p2, 0.5, 0.0))
    G[1] = np.where(R < p1, 1.0, np.where(R < p2, 0.5, 0.0))
    inner, outer = (R >= 1 - p1) & (R <= p1), (R >= 1 - p2) & (R <= p2)
    G[2] = np.where(inner, 1.0, np.where(outer, 0.5, 0.0))
    for k in range(3):
        G[k][nanR] = 1.0
    return G


def c_dop_gate(e, d: dict, S, S_of):
    """Новые точки C раунда (Ъ 22:30, PR #240) — тем же кодом, что разработка: fastbt.s9.c_dop_gates по разделу dop
    fastbt/designs/s9/setka_C.yaml (min_periods, сигнал), одна точка: W и p (фильтр) или p1, p2 (частичная).
    √RV_w — наивная S0 (S_of('naive'); в тесте — построена в открытии тем же кодом и прошла проверку сдвига)."""
    from fastbt import s9
    dop = s9.load_c_dop() if s9.C_DOP is None else s9.C_DOP
    f = dict(dop['semi'][d['semya']])
    W = int(d['W'])
    f['okna'] = [W]
    f.pop('filtr', None)
    f.pop('chastichnaya', None)
    if 'p' in d:
        f['filtr'] = [round(float(d['p']) * 100)]
    else:
        f['chastichnaya'] = [[round(float(d['p1']) * 100), round(float(d['p2']) * 100)]]
    spec = dict(min_periods=dop['min_periods'], semi={d['semya']: f})
    RV = None if f['signal'] != 'sigma_over_rvw' else S_of('naive')
    out = s9.c_dop_gates(S, RV, e.n_t, spec)
    if not out or len(out[0][1]) != 1:
        raise Stop(f'{d}: C dop — нет точки (C* на наивной σ не строится)')
    return out[0][1][0][1]


def fam_gate(e, d: dict, S_of):
    """Затвор/вес [тикер, бар] или [кат, тикер, бар] семьи дизайна d. S_of(метка) → σ [тикер, бар]."""
    from fastbt.masks import hyst_gate
    from fastbt.s9_stats import band_hold, ema, past_median
    fam = d['semya']
    S = S_of(d['sigma'])
    if fam == 'C1':
        return _cat_gate(_rank(e, S, d['W'], d.get('razgon')), float(d['p']))
    if fam == 'C2':
        return _cat_partial(_rank(e, S, d['W'], d.get('razgon')), float(d['p1']), float(d['p2']))
    if fam == 'Cmed':
        mw = int(d.get('med', 22))    # медиана прошлых mw дней (сдвиг на 1); s9_stats.past_median требует W ≥ 50 — здесь min = mw
        med = _per_ticker(e, S, lambda s: pd.Series(s).shift(1).rolling(mw, min_periods=mw).median().values)
        return _cat_gate(_rank(e, S / med, d['W'], d.get('razgon')), float(d['p']))
    if fam in ('C2W', 'C*1', 'C*2'):
        return c_dop_gate(e, d, S, S_of)
    if fam == 'D1':
        R = _rank(e, S, d['W'], d.get('razgon'))
        return np.where(R > float(d['p']), 0.0, 1.0)
    if fam == 'D2':
        R = _rank(e, S, d['W'], d.get('razgon'))
        p, g = float(d['p']), float(d['g'])
        return _per_ticker(e, R, lambda r: hyst_gate(r, p - g, p).astype(float))   # NaN ранга — открыт (hyst_gate)
    if fam == 'D4':
        Se = _per_ticker(e, S, lambda s: ema(s, int(d['span'])))
        R = _rank(e, Se, d['W'], d.get('razgon'))
        return np.where(R > float(d['p']), 0.0, 1.0)
    if fam == 'D5':
        chg = np.full(S.shape, np.nan)
        chg[:, 5:] = S[:, 5:] / S[:, :-5] - 1.0
        return np.where(chg > float(d['x']), 0.0, 1.0)
    if fam == 'D6':                                       # σ_pred / RVI 18:50 (лаг 0 или 1), ранг в окне W
        RV = e.rvi(int(d.get('lag', 0)))
        R = _rank(e, S * np.sqrt(252) * 100 / RV, d.get('W', 252), d.get('razgon'))
        return np.where(R > float(d['p']), 0.0, 1.0)
    if fam == 'Dcont':
        med = _per_ticker(e, S, lambda s: past_median(s, int(d['W'])))
        w = np.minimum(float(d['cap']), (med / S) ** float(d['gamma']))
        return np.where(np.isnan(w), 1.0, w)
    if fam == 'B':
        W = int(d.get('W', 252))
        cap, band = float(d['cap']), float(d['band'])

        def one(s):
            med = past_median(s, W)
            return band_hold(np.ascontiguousarray(np.minimum(cap, med / s)), band)
        return _per_ticker(e, S, one)
    raise Stop(f'неизвестная семья {fam}')


def external_gate(e, d: dict, S_of, root: Path):
    path, fn = d['kod'].split(':')
    spec = importlib.util.spec_from_file_location(f'ch4v2_kod_{Path(path).stem}', root / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)(e, S_of, **{k: v for k, v in d.items() if k not in ('kod', 'kod_sha256', 'klyuch')})


def p1_weights(e, S):
    """Веса 1/σ_i между тикерами: на каждой дате Σ по строкам (тикер × стратегия) = 1; строки без σ — средний вес даты."""
    inv = 1.0 / e.g_row(S)
    inv[~np.isfinite(inv)] = np.nan
    di = e.r['date']
    nd = len(e.dates)
    ok = ~np.isnan(inv)
    s = np.bincount(di[ok], inv[ok], nd)
    c = np.bincount(di[ok], None, nd)
    mean = np.where(c > 0, s / np.maximum(c, 1), 1.0)
    inv = np.where(ok, inv, mean[di])
    return inv / np.bincount(di, inv, nd)[di]


class Raschet:
    """Позиции дизайнов (один раз) и портфели на любом движке (без/с правилом закрытия) и комиссии."""

    def __init__(self, e, root: Path):
        self.e, self.root = e, root
        self.pos = {'A': (e.A_row, None)}

    def S_of(self, lab):
        return self.e.sigma(lab)

    def add(self, key: str, d: dict):
        e = self.e
        if d['semya'] == 'P2':
            self.pos[key] = ('mix', tuple(d['sostav']))
            return
        if d['semya'] == 'P1':
            base = self.pos[d.get('baza', 'A')][0]
            self.pos[key] = (base, p1_weights(e, self.S_of(d.get('sigma', 'hyb_h1'))))
            return
        if d.get('tochki'):
            self.pos[key] = (e.A_row * self.semya_po_validatsii(d), None)
            return
        G = external_gate(e, d, self.S_of, self.root) if d.get('kod') else fam_gate(e, d, self.S_of)
        self.pos[key] = (e.A_row * e.g_row(G), None)

    def semya_po_validatsii(self, d: dict) -> np.ndarray:
        """Семья С9 «выбор по валидации»: точки tochki одной семьи, на пару-год — первый лучший Шарп A·g на ЗАМОРОЖЕННОМ окне
        (Engine.select, окна s9.VAL_START…VAL_END); otsev — отсев по доле закрытых баров окна (как D в С9)."""
        e = self.e
        base = {k: v for k, v in d.items() if k not in ('tochki', 'otsev', 'klyuch')}
        Gs = [fam_gate(e, dict(base, **t), self.S_of) for t in d['tochki']]
        stack = np.stack(Gs)
        kind = 'mask' if np.isin(stack[~np.isnan(stack)], (0.0, 1.0)).all() else 'scale'
        g, cnt = e.select(stack, kind, tuple(d['otsev']) if d.get('otsev') else None, cat=stack.ndim == 4)
        self.vybor = getattr(self, 'vybor', {})
        self.vybor[str(d.get('klyuch') or d['semya'])] = {i: int(v) for i, v in cnt.items()}
        return g

    def port(self, eng, key, comm):
        pos, w = self.pos[key]
        if isinstance(pos, str):
            return eng.mix([self.port(eng, k, comm) for k in w])
        return eng.port(pos, comm=comm, wgt=w)


def metrics_okno(p: pd.DataFrame, lo, hi) -> tuple:
    from fastbt import s9
    q = p.loc[pd.Timestamp(lo):pd.Timestamp(hi)]
    yrs0 = s9.YEARS
    s9.YEARS = tuple(y for y in yrs0 if pd.Timestamp(lo).year <= y <= pd.Timestamp(hi).year)
    try:
        m = s9.metrics(q)
    finally:
        s9.YEARS = yrs0
    m['n_dney'] = len(q)
    return m, q


def lw(x1, x0, reps, block=5):
    from fastbt.s9_stats import lw_sharpe_diff
    return lw_sharpe_diff(x1, x0, reps, block)


def opis(d: dict) -> str:
    return ' '.join(f'{k}={v}' for k, v in d.items() if k not in ('kod_sha256',))


def run_tables(e, diz: dict, h2: dict, okna: dict, zakr, comm, comm_ch, reps, root, pometka, dop=None, osn=None) -> tuple:
    """Таблица подходов (A + по одному дизайну на подход [+ dop — только псевдотест]) и таблица H2."""
    R = Raschet(e, root)
    order = ['B', 'C', 'D'] + [k for k in diz if k not in ('B', 'C', 'D', 'MSP')] + (['MSP'] if 'MSP' in diz else [])
    for k in order:
        R.add(k, diz[k])
    dop = dop or {}
    for k, d in dop.items():
        R.add(f'dop:{k}', d)
    ez, hit = closure_engine(e, *zakr)
    engines = {'bez': e, 's_pravilom': ez}
    R.engines = engines
    rows = []
    keys = ['A'] + order + [f'dop:{k}' for k in dop]
    main_c, main_podpis = comm if isinstance(comm, tuple) else (comm, 'основная')
    comm = main_c
    podpis = {main_c: main_podpis}
    for c in (osn or []):                   # Ъ 27.10 12:05: обе основные ставки — в основных таблицах
        podpis[float(c['stavka'])] = f"основная: {c['podpis']}"
    glavnye = list(podpis)
    for c in comm_ch:                       # [{stavka, podpis}] — приложение (0,05 и 0,06)
        s, pp = (float(c['stavka']), c['podpis']) if isinstance(c, dict) else (float(c), 'чувствительность')
        if all(abs(s - x) > 1e-12 for x in podpis):
            podpis[s] = f'приложение (без p): {pp}'
    comms = list(podpis)
    for pr, eng in engines.items():
        for cm in comms:
            ports = {k: R.port(eng, k, cm) for k in keys}
            for ok, (lo, hi) in okna.items():
                mA, qA = metrics_okno(ports['A'], lo, hi)
                for k in keys:
                    m, q = metrics_okno(ports[k], lo, hi)
                    d = diz.get(k) or dop.get(k.replace('dop:', '')) or {}
                    r = dict(podhod=k, dizayn=opis(d) if d else 'main.yaml:A (ансамбль + фильтр оборота)',
                             okno=ok, pravilo=pr, komissiya_pct=cm * 100,
                             komissiya_rol=podpis[cm], tablitsa='osnovnaya' if cm in glavnye else 'prilozhenie',
                             rol='справочно (псевдотест, не в тест)' if k.startswith('dop:') else
                             ('эталон' if k == 'A' else 'один дизайн на подход'))
                    r.update(m)
                    r['sharpe_A'] = mA['sharpe']
                    if k != 'A' and cm in glavnye:
                        dd, se, pv = lw(q['excess'].values, qA['excess'].values, reps)
                        r.update(d_vs_A=dd, se_lw_vs_A=se, p_lw_vs_A=pv)
                    r['pometka'] = pometka
                    rows.append(r)
    tab = pd.DataFrame(rows)
    h2s = [run_h2(R, engines, diz, h2, okna, c, reps, pometka) for c in glavnye]
    h2t = pd.concat([x for x in h2s if len(x)], ignore_index=True) if any(len(x) for x in h2s) else pd.DataFrame()
    R.glavnye = glavnye
    return tab, h2t, hit, R


def run_h2(R, engines, diz, h2, okna, comm, reps, pometka) -> pd.DataFrame:
    """H2 внутри выбранной точки подхода h2.dizayn: та же точка на σ гибрида / HAR-J (б) / наивной — без выбора."""
    k = (h2 or {}).get('dizayn', 'C')
    d = diz.get(k)
    if not d or d.get('semya') in ('P1', 'P2') or 'sigma' not in d:
        return pd.DataFrame()
    hh = d['sigma'].rsplit('_', 1)[-1] if d['sigma'] != 'naive' else 'h1'
    lab = {'hyb': f'hyb_{hh}', 'harj_b': f'harj_b_{hh}', 'naive': 'naive'}
    srcs = [s for s in h2.get('istochniki', ['hyb', 'harj_b', 'naive'])]
    for s in srcs:
        R.add(f'h2:{s}', dict(d, sigma=lab[s]))
    rows = []
    for pr, eng in engines.items():
        ports = {s: R.port(eng, f'h2:{s}', comm) for s in srcs}
        for ok, (lo, hi) in okna.items():
            q = {s: metrics_okno(ports[s], lo, hi) for s in srcs}
            r = dict(podhod=k, dizayn=opis(d), okno=ok, pravilo=pr, komissiya_pct=comm * 100,
                     **{f'sharpe_{s}': q[s][0]['sharpe'] for s in srcs},
                     **{f'sigma_{s}': lab[s] for s in srcs})
            for a, b in (('hyb', 'harj_b'), ('hyb', 'naive'), ('harj_b', 'naive')):
                if a in q and b in q:
                    dd, se, pv = lw(q[a][1]['excess'].values, q[b][1]['excess'].values, reps)
                    r[f'd_{a}−{b}'], r[f'se_lw_{a}−{b}'], r[f'p_lw_{a}−{b}'] = dd, se, pv
            if str(d['semya']).startswith('C*'):
                r['zamechanie'] = 'C*: на наивной σ_pred / √RV ≡ 1 — строка наивной вырождена'
            r['rol'] = 'H2: одной таблицей, без выбора'
            r['pometka'] = pometka
            rows.append(r)
    return pd.DataFrame(rows)


def run_h3(R, engines, diz, okna, comm, reps, pometka) -> pd.DataFrame:
    """H3 и табл. 4.9 — кодом fastbt.bench после #242 (экспозиция d × доходность бенчмарка d→d+1, ставка за [d, d_next)):
    МСП подхода против MCFTR с той же экспозицией, MCFTR 16 % + ставка, MCFTR/RGBITR 20/80, 60/40, 80/20; p — ЛВ, блок 5.
    Основная строка — A (как табл. 4.9 v1, chapter4_a); выбранные дизайны — рядом."""
    from fastbt import bench, msp
    rate = msp.rate_cbr()
    out = []
    for pr, eng in engines.items():
        for k in ['A'] + [x for x in diz if x in PODHODY]:
            p = R.port(eng, k, comm)
            for ok, (lo, hi) in okna.items():
                t = bench.h3_table(p, rate, reps, lo, hi, lw_block=5)
                out.append(t.assign(podhod=k, okno=ok, pravilo=pr, komissiya_pct=comm * 100,
                                    rol='табл. 4.9: МСП-A (основная)' if k == 'A' else 'H3 для выбранного дизайна (рядом)',
                                    pometka=pometka))
    return pd.concat(out, ignore_index=True)


def synthetic_bench(dates) -> None:
    """Псевдотест: индексы MCFTR/RGBITR в data/ начинаются 03.01.2022 — не читаются; вместо них синтетика на датах
    разработки (проверка механики H3, числа смысла не имеют)."""
    from fastbt import bench
    rng = np.random.RandomState(7)
    idx = pd.DatetimeIndex(dates)
    ser = {str(bench.MCFTR): np.exp(np.cumsum(rng.normal(3e-4, 0.012, len(idx)))) * 1000,
           str(bench.RGBITR): np.exp(np.cumsum(rng.normal(2e-4, 0.003, len(idx)))) * 100}
    bench._close = lambda path: pd.Series(ser[str(path)], index=idx, name='close')


def komissiya_main(tst: dict) -> tuple:
    return float(tst['komissiya']), str(tst.get('komissiya_podpis') or 'основная')


KOMISSIYA_OTBORA_A = 0.0005      # внутренний отбор A: fastbt.s9.COMM (одно место; use_v2 скрипт не зовёт) — не трогать


def a_comm_ok() -> bool:
    """Отбор A идёт при 0,05 % (Ъ 27.10 12:05): fastbt.s9.COMM — единственный источник комиссии отбора."""
    from fastbt import s9
    return abs(float(s9.COMM) - KOMISSIYA_OTBORA_A) < 1e-12


def split_prilozhenie(tab: pd.DataFrame) -> tuple:
    """Основная таблица — обе основные ставки (0,035 и 0,04); приложение — 0,05 и 0,06 (отдельный файл)."""
    return (tab[tab['tablitsa'] == 'osnovnaya'].reset_index(drop=True),
            tab[tab['tablitsa'] == 'prilozhenie'].reset_index(drop=True))


def run_h3_glavnye(R, diz, okna, reps, pometka) -> pd.DataFrame:
    return pd.concat([run_h3(R, R.engines, diz, okna, c, reps, pometka) for c in R.glavnye], ignore_index=True)


def osn_ok(tab_osn: pd.DataFrame, tab_pril: pd.DataFrame, tst: dict) -> bool:
    """Обе основные ставки есть и различаются; p только на основных; 0,05 и 0,06 — только в приложении."""
    def pct(t):
        return sorted(t['komissiya_pct'].round(4).unique().tolist())
    a, b = (tab_osn[tab_osn.komissiya_pct.round(4) == c] for c in (0.035, 0.04))
    m = a.merge(b, on=['podhod', 'okno', 'pravilo'], suffixes=('_a', '_b'))
    return bool(abs(float(tst['komissiya']) - 0.0004) < 1e-12 and pct(tab_osn) == [0.035, 0.04]
                and pct(tab_pril) == [0.05, 0.06] and len(m) == len(a) == len(b) > 0
                and (m['sharpe_a'] != m['sharpe_b']).any() and tab_pril['p_lw_vs_A'].isna().all()
                and tab_osn.loc[tab_osn.podhod != 'A', 'p_lw_vs_A'].notna().all())


def fastbt_sha(root: Path) -> dict:
    """SHA-256 всех файлов кода fastbt (*.py, *.yaml) — в замок; fastbt.cache.code_hash — в спецификацию (fastbt_code_hash)."""
    return {str(p.relative_to(root)): sha(p) for p in sorted((Path(root) / 'fastbt').rglob('*'))
            if p.suffix in ('.py', '.yaml') and '__pycache__' not in p.parts}


def fastbt_agg(root: Path) -> str:
    """Одна SHA-256 по всем файлам кода fastbt: sha256 строк «путь SHA» (печать: --sha-fastbt)."""
    return hashlib.sha256('\n'.join(f'{k} {v}' for k, v in fastbt_sha(root).items()).encode()).hexdigest()


def to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=';', decimal=',', index=False, float_format='%.4f', encoding='utf-8-sig')
    return path


# ======================= детектор «будущее не влияет» (псевдотест) =======================

def leak_check(R: Raschet, diz: dict, dop: dict) -> list:
    """σ после среза t0 (середина ряда каждого тикера) заменяется шумом — затвор/вес на барах ≤ t0 обязан не измениться."""
    e = R.e
    rng = np.random.RandomState(20261026)
    out = []
    base = {lab: e.sigma(lab).copy() for lab in e.sig}
    items = []
    for k, d in list(diz.items()) + [(f'dop:{k}', v) for k, v in dop.items()]:
        if d.get('tochki'):            # семья: каждая точка отдельно (выбор по валидации — на прошлом окне, причинен)
            b0 = {a: b for a, b in d.items() if a not in ('tochki', 'otsev')}
            items += [(f'{k}[{i}]', dict(b0, **t)) for i, t in enumerate(d['tochki'])]
        else:
            items.append((k, d))
    for k, d in items:
        if d['semya'] in ('P1', 'P2') or d.get('kod'):
            continue
        G0 = fam_gate(e, d, R.S_of)
        cut = e.n_t // 2
        pert = {}
        for lab, S in base.items():
            X = S.copy()
            for ti in range(X.shape[0]):
                X[ti, cut[ti] + 1:e.n_t[ti]] *= np.exp(rng.normal(0, 0.5, e.n_t[ti] - cut[ti] - 1))
            pert[lab] = X
        G1 = fam_gate(e, d, lambda lab: pert[lab])
        same, moved = True, False
        for ti in range(len(e.tickers)):
            a, b = G0[..., ti, :cut[ti] + 1], G1[..., ti, :cut[ti] + 1]
            same &= bool(np.allclose(a, b, equal_nan=True))
            moved |= not np.allclose(G0[..., ti, cut[ti] + 1:e.n_t[ti]], G1[..., ti, cut[ti] + 1:e.n_t[ti]],
                                     equal_nan=True)
        out.append(dict(dizayn=k, semya=d['semya'], proshloe_ne_izmenilos=same, budushchee_izmenilos=moved,
                        ok=bool(same and moved)))
    return out


# ======================= режимы =======================

def _ok(fn) -> bool:
    try:
        fn()
        return True
    except SystemExit:
        return False


def selftest(P: Puti, reps: int) -> int:
    from fastbt import s9
    t0 = time.time()
    spec = load_spec(P)
    zag = spec['selftest_zaglushki']
    diz = {k: zag[k] for k in PODHODY}
    dop = zag.get('dop') or {}
    lock_before = Path(P.lock).exists()
    shas = verify_dev(P)
    out = Path(P.out_selftest)
    os.environ['FASTBT_VPRED'] = str(P.dev / DEV_FILES['hyb_h1'][0])
    from fastbt import data as fd
    fd.VPRED = Path(os.environ['FASTBT_VPRED'])
    cap_fastbt_inputs(DEV_CUT)
    src, rvi = sources_dev(P, out)
    sdv = shift_check(src, P.main / 'data' / 'regen' / 'features_pit_v3.parquet', '2018-01-01', DEV_CUT)
    to_csv(sdv, out / 'selftest_sdvig.csv')
    configure_s9((2018, 2019, 2020, 2021), (2017, 2017), DEV_CUT, src, rvi)   # окна заморожены на 2017 (как тест — на 2018–2021)
    cache = build_A((), 1)                       # в процессе: срез ≤ 2021 действует и на A
    from fastbt.s9 import Engine
    e = Engine(cache)
    assert e.dates.max() <= DEV_CUT, 'псевдотест: дата > 2021'
    for t in e.tickers:
        assert pd.to_datetime(cache[t]['dates']).max() <= DEV_CUT, f'{t}: бар > 2021'
    okna = {'2018–2021': ('2018-01-01', '2021-12-31'), '2020–2021': ('2020-01-01', '2021-12-31')}
    zakr = ('2020-03-28', '2020-03-29')          # псевдозакрытие (выходные): проверка механики правила
    tst = spec['test']
    tab, h2t, hit, R = run_tables(e, diz, spec.get('h2') or {}, okna, zakr, komissiya_main(tst), tst['komissiya_chuvstv'],
                                  reps, P.root, 'псевдотест 2018–2021, заглушки; ' + POMETKA, dop,
                                  tst['komissiya_osnovnye'])
    tab_osn, tab_pril = split_prilozhenie(tab)
    synthetic_bench(e.dates)
    h3t = run_h3_glavnye(R, diz, okna, reps, 'псевдотест, СИНТЕТИЧЕСКИЕ индексы; ' + POMETKA)
    to_csv(h3t, out / 'selftest_h3_sintetika.csv')
    leaks = leak_check(R, diz, dop)
    to_csv(tab_osn, out / 'selftest_podhody.csv')
    to_csv(tab_pril, out / 'selftest_podhody_prilozhenie.csv')
    to_csv(h2t, out / 'selftest_h2.csv')
    checks = dict(
        net_let_2022=bool(e.dates.max() <= DEV_CUT),
        zamok_ne_tronut=Path(P.lock).exists() == lock_before,
        po_odnomu_na_podhod=sorted(diz) == sorted(PODHODY),
        stroki=len(tab) == len(['A'] + list(diz) + list(dop)) * 2 * 2 * (len(tst['komissiya_osnovnye']) + len(tst['komissiya_chuvstv'])),
        komissii_dve_osnovnye=osn_ok(tab_osn, tab_pril, tst),
        otbor_A_005=a_comm_ok(),
        h2=len(h2t) == 2 * 2 * 2 and all(f'p_lw_{a}' in h2t for a in ('hyb−harj_b', 'hyb−naive', 'harj_b−naive')),
        p_v_0_1=bool(tab['p_lw_vs_A'].dropna().between(0, 1).all()),
        pravilo_srabotalo=len(hit) > 0 and bool((tab[tab.pravilo == 's_pravilom']['sharpe'].values
                                                  != tab[tab.pravilo == 'bez']['sharpe'].values).any()),
        detektor=all(r['ok'] for r in leaks),
        sdvig=bool(sdv['proshel'].all()) and len(sdv) == 3 * len(src),
        kod_posle_642f084=_ok(lambda: require_base(P.root, KOD_NE_RANSHE)),
        h3=len(h3t) == len(['A'] + list(diz)) * 2 * 2 * 6 * 2 and bool(h3t['p_boot'].dropna().between(0, 1).all()),
        A_sovpadaet_s_s9=None)
    # A той же математикой, что разработка S9 (Engine.port по позиции A) — Шарп по годам ≠ NaN
    mA = s9.metrics(e.port(e.A_row))
    checks['A_sovpadaet_s_s9'] = bool(all(np.isfinite(mA[str(y)]) for y in s9.YEARS))
    ok = all(v for v in checks.values())
    log = dict(rezhim='selftest (псевдотест 2018–2021)', vremya=now(), sekund=round(time.time() - t0, 1),
               boot_reps=reps, sha_dev=shas, sha_skripta=sha(__file__), sha_spec=sha(P.spec), git=git_head(P.root),
               pravilo_bary=hit, detektor=leaks, proverki=checks, ok=ok,
               A=dict(sharpe=mA['sharpe'], **{str(y): mA[str(y)] for y in s9.YEARS}))
    (out / 'selftest_log.json').write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(dict(proverki=checks, A=log['A'], sekund=log['sekund']), ensure_ascii=False, default=str))
    print('SELFTEST', 'OK' if ok else 'НЕ ПРОШЁЛ', '→', out)
    return 0 if ok else 1


def run_test(P: Puti, workers: int, reps_override: int | None = None) -> int:
    spec, shas = precheck_test(P)              # шаги 1–3: отказ не тратит открытие
    tst = spec['test']
    reps = int(reps_override or tst['butstrap']['reps'])
    if reps != 10000 or int(tst['butstrap']['blok']) != 5:
        raise Stop('тест: бутстрап ЛВ блок 5, 10 000 (FINAL v2) — иначе отказ')
    from fastbt.cache import code_hash
    acquire_lock(P, dict(skript=sha(__file__), spec=sha(P.spec), git=git_head(P.root), argv=sys.argv, sha=shas,
                         fastbt=fastbt_sha(P.root), fastbt_code_hash=code_hash(),
                         dizayny=spec['dizayny'], h2=spec.get('h2')))
    from fastbt import s9
    s9.set_scope('test', int(tst['val'][0]))   # замок fastbt (FINAL_DESIGN_strategies.md) + годы 2022–2026
    out_d, out_t = Path(P.out_data_test), Path(P.out_tables_test)
    src, rvi = sources_test(spec, P, out_d)
    # проверка сдвига каждого σ-ряда теста (в т. ч. наивной √RV_w для C*) — до любого счёта; не прошёл — отказ
    sdv = shift_check(src, P.main / spec['vhody']['features']['fayl'], tst['okna']['2022–2026'][0], tst['srez'])
    out_t.mkdir(parents=True, exist_ok=True)
    p_sdv = to_csv(sdv, out_t / 'ch4_v2_sdvig_test.csv')
    if not sdv['proshel'].all():
        bad = sdv.loc[~sdv['proshel'], ['ryad', 'variant', 'reshenie']].to_dict('records')
        finish_lock(P, dict(sdvig={str(p_sdv): sha(p_sdv)}, otkaz=bad), status='отказ: проверка сдвига не пройдена')
        raise Stop(f'сдвиг σ теста: {bad} — в расчёты не допускается, счёт не начат')
    os.environ['FASTBT_VPRED'] = str(src['hyb_h1'][0])
    from fastbt import data as fd
    fd.VPRED = Path(os.environ['FASTBT_VPRED'])
    if not a_comm_ok():
        raise Stop('отбор A не при 0,05 % (fastbt.s9.COMM) — отказ')
    cache = build_A((('test', int(tst['val'][0])),), workers)
    configure_s9(s9.TEST_YEARS, tuple(tst['val']), tst['srez'], src, rvi)
    e = s9.Engine(cache)
    okna = {k: tuple(v) for k, v in tst['okna'].items()}
    pom = POMETKA if not spec.get('audit_zakryt') else 'после аудита'
    tab, h2t, hit, R = run_tables(e, spec['dizayny'], spec.get('h2') or {}, okna, tuple(tst['zakrytie']),
                                  komissiya_main(tst), tst['komissiya_chuvstv'], reps, P.root, pom, None, tst['komissiya_osnovnye'])
    h3t = run_h3_glavnye(R, spec['dizayny'], okna, reps, pom)
    tab_osn, tab_pril = split_prilozhenie(tab)
    outs = [to_csv(tab_osn, out_t / 'ch4_v2_podhody_test.csv'),
            to_csv(tab_pril, out_t / 'ch4_v2_podhody_test_prilozhenie.csv'), to_csv(h2t, out_t / 'ch4_v2_h2_test.csv'),
            to_csv(h3t, out_t / 'ch4_v2_tabl_4_9_h3_test.csv'), p_sdv]
    for k in R.pos:
        if not k.startswith('h2:'):
            for c in R.glavnye:
                R.port(e, k, c).to_parquet(out_d / f'port_{k}_k{round(c * 1e5)}_test.parquet')
    finish_lock(P, dict(vyhody={str(p): sha(p) for p in outs}, pravilo_bary=hit))
    print('ТЕСТ v2 завершён →', out_t)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--selftest', action='store_true', help='псевдотест 2018–2021 на заглушках, без замка')
    g.add_argument('--scope', choices=['test'], help='одно открытие теста (только после «да»)')
    g.add_argument('--sha-fastbt', action='store_true', help='печать SHA кода fastbt для заморозки (данных не читает)')
    ap.add_argument('--boot-reps', type=int, default=None)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args(argv)
    P = Puti()
    if a.sha_fastbt:
        print(json.dumps(dict(fastbt_sha_vse=fastbt_agg(P.root), git=git_head(P.root), faily=fastbt_sha(P.root)),
                         ensure_ascii=False, indent=1))
        return 0
    if a.selftest:
        return selftest(P, a.boot_reps or 10000)
    return run_test(P, a.workers, a.boot_reps)


if __name__ == '__main__':
    sys.exit(main())
