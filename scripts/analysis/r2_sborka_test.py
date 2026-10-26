#!/usr/bin/env python3
"""R2, агент A: тестовая сборка победителей второго раунда по правилу 13:00 (протокол v3 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md,
разделы 6, 8). Заготовка агента О (PR #144), доведена агентом A под v3. До ворот E боевой запуск невозможен; проверка — только --selftest.

Правило доступа (раздел 8):
- data/regen/test_locked/ читает только агент A и только этим скриптом; только после ворот E (файлы R2/O/gate_E_h{1,5,22}.yaml есть все,
  их SHA-256 записаны в спецификации) и только файлы компонент победителей, перечисленные в спецификации;
- спецификация models/designs/r2/r2_A_sborka.yaml пишется после ворот E, статус «zamorozheno»; её SHA-256 пишется в журнал до первого
  чтения test_locked и в SHA256.txt выхода; повторный запуск обязан дать те же байты (иначе СТОП);
- комбинирование механическое, по правилу, выбранному на 2017–2021 (дизайн победителя, design_sha из gate_E). Параметры:
    * константа — значение, выбранное на 2017–2021 вложенным выбором (гиперпараметр дизайна);
    * по шагам — правило t−1, как на валидации: к тесту шага k применяется оценка по val шагов <= k (годы < 2016 + k);
      у шагов k >= 7 это потребовало бы лет >= 2022, поэтому берётся последняя оценка по годам <= 2021 — шага 6 (заморозка).
      В спецификации допустимы только ключи 1..6 (assert);
    * построчные множители (s² от состояния, поправки калибровки кластера 3, ec и т. п.) — готовые файлы авторов в test_locked/R2/<агент>/, колонки
      date, ticker, step, mnozh; параметры в них оценены по годам <= 2021 (авторы, раздел 6);
- целевая колонка не загружается: белый список колонок, assert; матрица признаков не открывается; оценок качества нет;
- запись — только test_locked/R2/final/final_r2_h{h}_test.parquet (date, ticker, year, step, rv_pred, design) и SHA256.txt;
- журнал — docs/otchety/dev_log_r2_A.md: прочитанные и записанные файлы с SHA-256.
Проверка в PR — grep протокола (раздел 8): совпадения допустимы только в строках с меткой R2-guard.

python3 scripts/analysis/r2_sborka_test.py --selftest        # синтетика во временном каталоге; test_locked не читается
python3 scripts/analysis/r2_sborka_test.py                   # фаза F, после ворот E; один раз (повтор — сверка байтов)

Ворота E — R2/O/gate_E_h{h}.yaml (r2_loyo.gate_E / write_gate): phase: E; h; pobeditel: {pobeditel: <единица> | 'F_h(d0)', V0: bool,
poryadok}. Победил V0 — финал = эконометрическая компонента: pravilo single, одна компонента. design_sha в воротах, если появится, сверяется.

Спецификация (models/designs/r2/r2_A_sborka.yaml):
  status: zamorozheno
  gate_E_sha: {1: <sha256 gate_E_h1.yaml>, 5: ..., 22: ...}
  h:
    5:
      design: <pobeditel.pobeditel из gate_E_h5>
      design_sha: <SHA-256 канонического yaml победителя (схема v1, раздел 5)>
      design_yaml: {file: models/designs/r2/<...>.yaml, sha: <sha256 файла>}          # сверка SHA
      pravilo: single | mean | median | weights | ridge_noga
      komponenty:                                  # сырые тестовые прогнозы шагов 1-10 (без s²), по одному файлу
        - {name: luchshiy, file: test_locked/R2/B/<ряд>_h5_test.parquet, factors: {1: F1, ..., 6: F6} | F,
           mnozh_file: test_locked/R2/L/<ряд>_h5_mnozh_test.parquet}          # factors и mnozh_file — необязательно
      weights: {1: {luchshiy: w, noga: w}, ..., 6: {...}} | {luchshiy: w, noga: w}   # pravilo weights (w >= 0, сумма 1)
      beta: {1: b1, ..., 6: b6} | b                # pravilo ridge_noga: ln p = ln noga + β·(ln luchshiy − ln noga)
      factor_itog: {1: ..., 6: ...} | F            # необязательно: множитель к итогу
      mnozh_itog_file: test_locked/R2/L/<ряд>_h5_mnozh_test.parquet               # необязательно: построчный множитель к итогу
      istochnik_parametrov: {file: data/regen/R2/..., sha: <sha256>}             # val-файл, из которого взяты параметры (сверка SHA)
  F_h_d0: [1, 22]                                  # h, где победителя нет: сборки нет, строка гибрида — F_h(d0) (final_test_r2)
"""
import argparse, hashlib, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
REGEN = MAIN / 'data' / 'regen'
LOCKED = REGEN / 'test_locked'
OUTDIR = LOCKED / 'R2' / 'final'
GATES = REGEN / 'R2' / 'O'
SPEC = REPO / 'models' / 'designs' / 'r2' / 'r2_A_sborka.yaml'
JOURNAL = REPO / 'docs' / 'otchety' / 'dev_log_r2_A.md'
HS = (1, 5, 22)
KEEP = ('date', 'ticker', 'year', 'step', 'rv_pred')  # белый список колонок прогноза
KEEP_M = ('date', 'ticker', 'step', 'mnozh')  # белый список колонок построчного множителя
TARGET = ('rv_actual', 'rv_target', 'y')  # R2-guard: целевые колонки, которые не загружаются никогда
KEY = ['date', 'ticker', 'step']
LAST = 6  # последний шаг разработки (val 2021)
RULES = ('single', 'mean', 'median', 'weights', 'ridge_noga')
F_D0 = (None, 'F_h(d0)', 'F_h_d0')  # «победителя нет» в gate_E (r2_loyo.pobeditel пишет 'F_h(d0)')
READ, WROTE = [], []


class Stop(Exception):
    pass


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(MAIN.resolve()))
    except ValueError:
        return str(p)


def par(d, k):
    """Параметр к тесту шага k. Константа — как есть. По шагам: оценка по val шагов <= min(k, 6) (t−1 как на валидации;
    k >= 7 — заморозка на шаге 6). Ключей > 6 быть не может: оценок по годам >= 2022 в спецификации нет."""
    if not isinstance(d, dict):
        return d
    d = {int(a): b for a, b in d.items()}
    assert max(d) <= LAST, f'параметр с ключом шага {max(d)} > {LAST}: оценка по годам >= 2022 запрещена'
    assert set(range(1, LAST + 1)) <= set(d), f'параметр: нет шагов {sorted(set(range(1, LAST + 1)) - set(d))}'
    return d[min(int(k), LAST)]


def par_steps(d, steps):
    u = {k: float(par(d, k)) for k in np.unique(steps)}
    return np.array([u[k] for k in steps], dtype=float)


# ---------------------------------------------------------------- ворота E и спецификация (до любого чтения test_locked)
def vorota(spec_path):
    """Ворота E пройдены на всех h, SHA каждого gate_E совпал со спецификацией, победители и design_sha совпали.
    -> (spec, {h: c}) — только h с победителем R2."""
    spec = yaml.safe_load(Path(spec_path).read_text())
    if spec.get('status') != 'zamorozheno':
        raise Stop(f'{spec_path}: статус {spec.get("status")!r}, нужен zamorozheno — сборка только после ворот E')
    gsha = {int(h): s for h, s in (spec.get('gate_E_sha') or {}).items()}
    hs = {int(h): c for h, c in (spec.get('h') or {}).items()}
    fd0 = {int(h) for h in (spec.get('F_h_d0') or [])}
    if set(hs) & fd0 or set(hs) | fd0 != set(HS):
        raise Stop(f'спецификация: h со сборкой {sorted(hs)} и с F_h(d0) {sorted(fd0)} должны разбивать {HS}')
    for h in HS:
        g = GATES / f'gate_E_h{h}.yaml'
        if not g.exists():
            raise Stop(f'нет {g} — ворота E на h={h} не пройдены')
        if sha(g) != gsha.get(h):
            raise Stop(f'{g.name}: SHA-256 не совпал со спецификацией (ворота изменены после записи спецификации)')
        G = yaml.safe_load(g.read_text())
        if G.get('phase') != 'E' or int(G['h']) != h:
            raise Stop(f'{g.name}: не ворота E на h={h}')
        pb = G['pobeditel'] if isinstance(G.get('pobeditel'), dict) else dict(pobeditel=G.get('pobeditel'), V0=False)
        win = pb['pobeditel']
        if win in F_D0:
            if h not in fd0:
                raise Stop(f'h={h}: по воротам победителя нет (F_h(d0)), а спецификация собирает {hs.get(h, {}).get("design")}')
            continue
        if h not in hs:
            raise Stop(f'h={h}: победитель {win}, а в спецификации сборки нет')
        c = hs[h]
        if c.get('design') != win or (G.get('design_sha') and c.get('design_sha') != G['design_sha']):
            raise Stop(f'h={h}: дизайн спецификации {c.get("design")} != ворота {win} (или design_sha)')
        if c.get('pravilo') not in RULES:
            raise Stop(f'h={h}: правило {c.get("pravilo")!r} не из {RULES}')
        if pb.get('V0') and (c['pravilo'] != 'single' or len(c['komponenty']) != 1):
            raise Stop(f'h={h}: победил V0 — финал = эконометрическая компонента (single, одна компонента)')
        for key in ('design_yaml', 'istochnik_parametrov'):  # yaml дизайна и val-файл параметров: вне test_locked, SHA совпал
            src = c.get(key)
            if src:
                p = Path(src['file']) if Path(src['file']).is_absolute() else MAIN / src['file']
                assert 'test_locked' not in p.parts, f'h={h}: {key} — только вне test_locked'
                if sha(p) != src['sha']:
                    raise Stop(f'h={h}: {p.name}: SHA-256 ({key}) не совпал')
    return spec, hs


# ---------------------------------------------------------------- чтение test_locked (только белый список)
def allowed(r):
    p = (REGEN / r).resolve() if not Path(r).is_absolute() else Path(r).resolve()
    lk = LOCKED.resolve()
    assert str(p).startswith(str(lk) + '/'), f'{r}: не в test_locked'
    assert not str(p).startswith(str(OUTDIR.resolve()) + '/'), f'{r}: выход сборки не читается'
    assert not any(x in p.parts for x in ('vyr', 'final3')), f'{r}: замороженные ряды эталонов и первого раунда сборка не читает'
    assert p.exists(), f'{r}: нет файла'
    return p


def read_locked(r, keep):
    p = allowed(r)
    names = pq.read_schema(p).names
    use = [c for c in keep if c in names]
    assert use == list(keep), f'{p.name}: нет колонок {set(keep) - set(use)}'
    assert not set(use) & set(TARGET)  # R2-guard: целевая колонка не в белом списке
    d = pd.read_parquet(p, columns=use)
    assert not set(d.columns) & set(TARGET)  # R2-guard: целевая колонка не загружена
    READ.append((rel(p), sha(p), len(d)))
    assert d.duplicated(KEY).sum() == 0 and set(d.step.unique()) <= set(range(1, 11)), f'{p.name}: дубли ключа или шаг вне 1–10'
    return d


def read_pred(r):
    d = read_locked(r, KEEP)
    assert d.rv_pred.gt(0).all() and np.isfinite(d.rv_pred).all(), f'{r}: прогноз <= 0 или не конечен'
    return d


def read_mnozh(r, keys):
    d = read_locked(r, KEEP_M)
    m = keys.merge(d, on=KEY, how='left', validate='1:1')
    assert m.mnozh.notna().all() and (m.mnozh > 0).all(), f'{r}: множитель есть не у всех строк или <= 0'
    return m.mnozh.values


# ---------------------------------------------------------------- сборка
def combine(c, h):
    """Компонента × свой множитель (по шагам и/или построчный) → объединение по правилу → множители итога."""
    K = c['komponenty']
    names = [k['name'] for k in K]
    assert len(set(names)) == len(names), f'h={h}: имена компонент повторяются'
    parts = []
    for comp in K:
        d = read_pred(comp['file'])
        f = par_steps(comp['factors'], d.step.values) if comp.get('factors') is not None else 1.0
        if comp.get('mnozh_file'):
            f = f * read_mnozh(comp['mnozh_file'], d[KEY])
        parts.append(d.assign(rv_pred=d.rv_pred * f)[KEY + ['year', 'rv_pred']].rename(columns={'rv_pred': comp['name']}))
    m = parts[0]
    for p in parts[1:]:
        m = m.merge(p.drop(columns='year'), on=KEY, how='inner', validate='1:1')
    n0 = len(parts[0])
    assert all(len(p) == n0 for p in parts) and len(m) == n0, f'h={h}: строки компонент не совпадают'
    X = m[names].values
    rule = c['pravilo']
    if rule == 'single':
        assert len(names) == 1, f'h={h}: single — одна компонента'
        pred = X[:, 0]
    elif rule == 'mean':
        pred = X.mean(axis=1)
    elif rule == 'median':
        pred = np.median(X, axis=1)
    elif rule == 'weights':
        w = c['weights']
        if all(nm in w for nm in names):  # константа, выбранная на 2017–2021
            W = np.tile([float(w[nm]) for nm in names], (len(m), 1))
        else:  # по шагам: t−1, заморозка шага 6
            S = {k: [float(par(w, k)[nm]) for nm in names] for k in np.unique(m.step.values)}
            W = np.array([S[k] for k in m.step.values])
        assert np.all(W >= 0) and np.allclose(W.sum(axis=1), 1), f'h={h}: веса'
        pred = (W * X).sum(axis=1)
    elif rule == 'ridge_noga':
        assert 'noga' in names and len(names) == 2, f'h={h}: ridge_noga — компоненты noga и ещё одна'
        other = [x for x in names if x != 'noga'][0]
        ln_n, ln_b = np.log(m['noga'].values), np.log(m[other].values)
        pred = np.exp(ln_n + par_steps(c['beta'], m.step.values) * (ln_b - ln_n))
    else:
        raise Stop(f'неизвестное правило {rule}')
    if c.get('factor_itog') is not None:
        pred = pred * par_steps(c['factor_itog'], m.step.values)
    if c.get('mnozh_itog_file'):
        pred = pred * read_mnozh(c['mnozh_itog_file'], m[KEY])
    out = m[['date', 'ticker', 'year', 'step']].assign(rv_pred=pred, design=c['design'])
    assert np.isfinite(out.rv_pred).all() and (out.rv_pred > 0).all(), f'h={h}: итог не конечен или <= 0'
    return out.sort_values(['step', 'date', 'ticker']).reset_index(drop=True)


def journal(lines):
    new = not JOURNAL.exists()
    with open(JOURNAL, 'a') as f:
        if new:
            f.write('# Журнал R2, агент A (сборка теста, второе открытие, FINAL_DESIGN_round2)\n')
        f.write('\n'.join(lines) + '\n')


def table(rows):
    return ['| файл | SHA-256 | строк |', '|---|---|---|'] + [f'| {f} | {s} | {n} |' for f, s, n in rows]


def run(spec_path):
    spec_path = Path(spec_path)
    spec, hs = vorota(spec_path)  # до любого чтения test_locked
    s_spec, s_code = sha(spec_path), sha(__file__)
    journal([f'\n## Сборка теста R2 — старт {now()}', '',
             f'Спецификация {rel(spec_path)} SHA-256 {s_spec}; код r2_sborka_test.py SHA-256 {s_code}.',
             f'Ворота E: ' + '; '.join(f'h={h} {s}' for h, s in sorted(spec['gate_E_sha'].items())) + '.',
             f'Сборка на h = {sorted(hs)}; F_h(d0) на h = {sorted(spec.get("F_h_d0") or [])}. test_locked до этой строки не читался.'])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for h, c in sorted(hs.items()):
        o = combine(c, h)
        f = OUTDIR / f'final_r2_h{h}_test.parquet'
        tmp = OUTDIR / f'.final_r2_h{h}_test.parquet.new'
        o.to_parquet(tmp, index=False)
        if f.exists():  # повтор: байты обязаны совпасть
            if sha(tmp) != sha(f):
                tmp.unlink()
                raise Stop(f'{f.name}: повтор сборки дал другие байты — СТОП (спецификация или входы изменены)')
            tmp.unlink()
        else:
            tmp.rename(f)
        WROTE.append((rel(f), sha(f), len(o)))
        out[h] = o
        print(f'h={h}: {c["design"]} -> {f.name} ({len(o)} строк, шаги {o.step.min()}–{o.step.max()}); оценок качества нет', flush=True)
    (OUTDIR / 'SHA256.txt').write_text(''.join(f'{s}  {Path(p).name}\n' for p, s, _ in WROTE)
                                       + f'{s_spec}  {spec_path.name} (спецификация)\n{s_code}  r2_sborka_test.py (код)\n')
    journal(['', 'Прочитано (только колонки белого списка; целевой колонки нет):', ''] + table(READ)
            + ['', 'Записано:', ''] + table(WROTE) + ['', f'Конец {now()}.'])
    return out


# ---------------------------------------------------------------- самопроверка (синтетика во временном каталоге)
def selftest():
    global REGEN, LOCKED, OUTDIR, GATES, JOURNAL
    saved = (REGEN, LOCKED, OUTDIR, GATES, JOURNAL)
    tmp = Path(tempfile.mkdtemp(prefix='r2_sborka_selftest_'))
    REGEN, JOURNAL = tmp / 'regen', tmp / 'dev_log_r2_A.md'
    LOCKED, GATES = REGEN / 'test_locked', REGEN / 'R2' / 'O'
    OUTDIR = LOCKED / 'R2' / 'final'
    try:
        rng = np.random.default_rng(13)
        for d in (LOCKED / 'R2' / 'B', LOCKED / 'R2' / 'L', LOCKED / 'vyr', GATES, REGEN / 'R2' / 'val'):
            d.mkdir(parents=True)
        dates = pd.bdate_range('2017-01-03', '2026-02-27')
        base = pd.DataFrame([(d, t) for d in dates[::7] for t in ('SBER', 'LKOH', 'AFLT')], columns=['date', 'ticker'])
        base['year'] = base.date.dt.year
        base['step'] = base.year - 2016
        nb = base.assign(rv_pred=np.exp(rng.normal(-8, .3, len(base))))
        lb = base.assign(rv_pred=np.exp(rng.normal(-8, .3, len(base))))
        nb.assign(rv_actual=1.0).to_parquet(LOCKED / 'R2' / 'L' / 'noga_h5_test.parquet', index=False)  # R2-guard: целевая в файле есть — не читается
        lb.to_parquet(LOCKED / 'R2' / 'B' / 'luchshiy_h5_test.parquet', index=False)
        mz = base[KEY].assign(mnozh=np.exp(rng.normal(0, .05, len(base))))
        mz.to_parquet(LOCKED / 'R2' / 'L' / 'mnozh_h5_test.parquet', index=False)
        nb.to_parquet(LOCKED / 'R2' / 'L' / 'noga_h22_test.parquet', index=False)
        (LOCKED / 'vyr' / 'etalon_d0_h5_test.parquet').write_bytes(b'x')
        pv = REGEN / 'R2' / 'val' / 'factors_h5.csv'
        pv.write_text('step,F\n1,1.1\n')
        F = {k: 1 + k / 100 for k in range(1, 7)}
        B = {k: 0.5 + k / 100 for k in range(1, 7)}
        gates = {1: dict(phase='E', h=1, pobeditel=dict(pobeditel='F_h(d0)', V0=False, poryadok=['u1'])),
                 5: dict(phase='E', h=5, pobeditel=dict(pobeditel='D_h5_x', V0=False, poryadok=['D_h5_x']), design_sha='a' * 64),
                 22: dict(phase='E', h=22, pobeditel=dict(pobeditel='V0_HARX_h22', V0=True, poryadok=['u2', 'V0_HARX_h22']))}
        for h, g in gates.items():
            (GATES / f'gate_E_h{h}.yaml').write_text(yaml.safe_dump(g, allow_unicode=True))
        spec = dict(status='zamorozheno', gate_E_sha={h: sha(GATES / f'gate_E_h{h}.yaml') for h in HS}, F_h_d0=[1], h={
            5: dict(design='D_h5_x', design_sha='a' * 64, pravilo='ridge_noga', beta=B, factor_itog=F,
                    komponenty=[dict(name='noga', file='test_locked/R2/L/noga_h5_test.parquet', factors=1.02),
                                dict(name='luchshiy', file='test_locked/R2/B/luchshiy_h5_test.parquet')],
                    mnozh_itog_file='test_locked/R2/L/mnozh_h5_test.parquet', istochnik_parametrov=dict(file=str(pv), sha=sha(pv))),
            22: dict(design='V0_HARX_h22', design_sha='b' * 64, pravilo='single',
                     komponenty=[dict(name='noga', file='test_locked/R2/L/noga_h22_test.parquet', factors=F)])})
        sp = tmp / 'spec.yaml'

        def put(s, p=sp):
            p.write_text(yaml.safe_dump(s, allow_unicode=True, sort_keys=False))
            return p

        def stop(s, what, exc=Stop):
            READ.clear(); WROTE.clear()
            try:
                run(put(s, tmp / 'bad.yaml'))
            except exc as e:
                assert what in str(e), (what, e)
                return
            raise AssertionError(f'не пойман: {what}')

        # 1. сборка: значения по правилу, t−1 с заморозкой шага 6, построчный множитель, V0 = single
        out = run(put(spec))
        assert sorted(out) == [5, 22] and not (OUTDIR / 'final_r2_h1_test.parquet').exists()
        o = out[5].merge(nb.rename(columns={'rv_pred': 'n'})[KEY + ['n']], on=KEY).merge(
            lb.rename(columns={'rv_pred': 'b'})[KEY + ['b']], on=KEY).merge(mz, on=KEY)
        k6 = o.step.clip(upper=6)
        ln_n = np.log(o.n * 1.02)
        want = np.exp(ln_n + k6.map(B) * (np.log(o.b) - ln_n)) * k6.map(F) * o.mnozh
        assert np.allclose(o.rv_pred, want, rtol=1e-12), 'ridge_noga'
        assert (o[o.step >= 7].step.clip(upper=6) == 6).all() and o.step.max() == 10
        o22 = out[22].merge(nb.rename(columns={'rv_pred': 'n'})[KEY + ['n']], on=KEY)
        assert np.allclose(o22.rv_pred, o22.n * o22.step.clip(upper=6).map(F), rtol=1e-12), 'single + F(t−1, заморозка 6)'
        assert set(out[5].columns) == {'date', 'ticker', 'year', 'step', 'rv_pred', 'design'}
        assert all('rv_actual' not in pq.read_schema(OUTDIR / f'final_r2_h{h}_test.parquet').names for h in (5, 22))  # R2-guard
        assert len(READ) == 4 and all('vyr' not in r for r, _, _ in READ)
        j = JOURNAL.read_text()
        assert sha(sp) in j and 'noga_h5_test.parquet' in j and j.index('test_locked до этой строки не читался') < j.index('Прочитано')
        # 2. повтор — те же байты; изменённый вход — СТОП
        READ.clear(); WROTE.clear()
        run(sp)
        s2 = {**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'komponenty': [dict(name='noga', file='test_locked/R2/L/noga_h22_test.parquet',
                                                                                       factors={**F, 6: 1.5})]}}}
        stop(s2, 'другие байты')
        # 3. ворота: нет gate_E, изменён gate_E, дизайн не тот, статус, разбиение h
        stop({**spec, 'status': 'ждёт ворот E'}, 'zamorozheno')
        stop({**spec, 'F_h_d0': []}, 'разбивать')
        stop({**spec, 'h': {**spec['h'], 5: {**spec['h'][5], 'design_sha': 'c' * 64}}}, 'ворота')
        stop({**spec, 'h': {**spec['h'], 5: {**spec['h'][5], 'pravilo': 'stack'}}}, 'правило')
        stop({**spec, 'h': {**spec['h'], 5: {**spec['h'][5], 'istochnik_parametrov': dict(file=str(pv), sha='0' * 64)}}}, 'istochnik')
        stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'pravilo': 'mean'}}}, 'V0')
        g5 = GATES / 'gate_E_h5.yaml'
        g5.write_text(g5.read_text() + '# правка\n')
        stop(spec, 'SHA-256 не совпал')
        g5.unlink()
        stop(spec, 'нет')
        (GATES / 'gate_E_h5.yaml').write_text(yaml.safe_dump(gates[5], allow_unicode=True))
        # 4. параметр с ключом шага > 6 (оценка по годам >= 2022) — отказ; файл вне test_locked, в vyr, в выходе — отказ
        stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'komponenty': [dict(name='noga', file='test_locked/R2/L/noga_h22_test.parquet',
                                                                                     factors={**F, 7: 1.3})]}}}, '>= 2022', AssertionError)
        for bad, what in (('R2/val/factors_h5.csv', 'не в test_locked'), ('test_locked/vyr/etalon_d0_h5_test.parquet', 'замороженные'),
                          ('test_locked/R2/final/final_r2_h5_test.parquet', 'выход сборки')):
            stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'komponenty': [dict(name='noga', file=bad)]}}}, what, AssertionError)
        # 5. белый список: целевая колонка в файле есть, но не загружается; файла без rv_pred — отказ
        nb[KEY + ['year']].assign(rv_actual=1.0).to_parquet(LOCKED / 'R2' / 'L' / 'bez_pred_h22_test.parquet', index=False)  # R2-guard
        stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'komponenty': [dict(name='noga', file='test_locked/R2/L/bez_pred_h22_test.parquet')]}}},
             'нет колонок', AssertionError)
    finally:
        REGEN, LOCKED, OUTDIR, GATES, JOURNAL = saved
        READ.clear(); WROTE.clear()
    print('selftest r2_sborka_test: OK (ворота E и SHA спецификации до чтения; правила single / ridge_noga; t−1 с заморозкой шага 6, ключи > 6 '
          '— отказ; построчный множитель; белый список без целевой колонки; пути только test_locked/R2, не vyr и не выход; повтор — те же '
          'байты, иначе СТОП):', tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default=str(SPEC))
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        run(a.spec)
    except Stop as e:
        print('СТОП:', e, flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
