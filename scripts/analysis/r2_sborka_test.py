#!/usr/bin/env python3
"""R2, агент A: тестовая сборка второго раунда по правилу 13:00 (протокол v3 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, разделы 6, 8)
и s² последовательно при открытии (решение владельца; эмбарго s² — PR #194, #195). Заготовка О (PR #144), доведена агентом A.
До ворот E боевой запуск невозможен; проверка — только --selftest.

Правило доступа (раздел 8):
- data/regen/test_locked/ читает только агент A и только этим скриптом; только после ворот E (gate_E_h{1,5,22}.yaml есть все, их SHA-256
  записаны в спецификации); только файлы компонент победителей и (для s²) замороженные ряды ВЫР-1 vyr/etalon_d0_h{h}_test, vyr/final_d0_h{h}_test;
- спецификация models/designs/r2/r2_A_sborka.yaml — статус zamorozheno; её SHA пишется в журнал до первого чтения test_locked и в SHA256.txt;
  повторный запуск обязан дать те же байты (иначе СТОП);
- комбинирование механическое, по правилу, выбранному на 2017–2021. Константы — как выбраны; параметры по шагам (веса, β) — t−1 с
  заморозкой шага 6 (ключи только 1..6, assert);
- s² калибровки (Б0.1, F = exp(κ·s²), s2_est ∈ sm / smrob / sm_roll1y / sm_roll2y / sm_pooled, κ — константа) у эталона (б) d0, F_h(d0) и победителя —
  одним правилом, симметрично (s2_pravilo в спецификации):
    * posledovatelno (решение владельца, по умолчанию): к тесту шага k (тестовый год ty = 2016 + k) — s² по OOS ln-остаткам val шагов окна
      (sm: k−2..k), то есть по годам ≤ ty − 1 (t−1); эмбарго — без строк, чьё окно цели [t+1, t+h] заходит в ty (calib.s2_embargo = on).
      Для k ≥ 7 нужны остатки val шагов 7–10 (2022–2025): цель читается ТОЛЬКО из part = val этих шагов (строки с меткой R2-guard-s2),
      цель тестовой части не загружается никогда; оценок качества нет — s² это множитель;
    * zamorozka (протокол v3, раздел 8): шаги 1–6 — так же, шаги 7–10 — s² шага 6; цель из test_locked не читается вовсе;
  контроль: тот же код без эмбарго обязан повторить F_sm файлов ВЫР-1 (правило R1, gibrid_bias.s2_est), иначе СТОП;
- выход — test_locked/R2/final/:
    final_r2_h{h}_test.parquet (date, ticker, year, step, rv_pred, design) — победитель;
    etalon_d0_s2emb_h{h}_test.parquet (date, ticker, year, step, har_raw, har_corr, F_sm, F_sm_bez_emb) — эталоны (а), (б) d0 с новым s²;
    final_d0_s2emb_h{h}_test.parquet (date, ticker, year, step, rv_pred, leg_raw, F_sm, F_sm_bez_emb, design) — F_h(d0) с новым s²;
    SHA256.txt; цели в выходе нет;
- журнал — docs/otchety/dev_log_r2_A.md: прочитанные (с пометкой части val для s²) и записанные файлы с SHA-256.
Проверка в PR — grep протокола (раздел 8): совпадения допустимы только в строках с меткой R2-guard (R2-guard-s2 — её частный случай).

python3 scripts/analysis/r2_sborka_test.py --selftest        # синтетика во временном каталоге; test_locked не читается
python3 scripts/analysis/r2_sborka_test.py                   # фаза F, после ворот E; один раз (повтор — сверка байтов)

Ворота E — R2/O/gate_E_h{h}.yaml (r2_loyo.gate_E): phase: E; h; pobeditel: {pobeditel: <единица> | 'F_h(d0)', V0, poryadok}.
Победил V0 — финал = эконометрическая компонента: pravilo single, одна компонента. design_sha в воротах, если появится, сверяется.

Спецификация (models/designs/r2/r2_A_sborka.yaml):
  status: zamorozheno
  s2_pravilo: posledovatelno | zamorozka
  s2_etalon: {est: sm, kappa: 0.5, embargo: on}    # эталон (б) d0 и F_h(d0): Б0.1 sm, как в ВЫР-1, плюс эмбарго
  gate_E_sha: {1: <sha256 gate_E_h1.yaml>, 5: ..., 22: ...}
  F_h_d0: [1]                                      # h без победителя: строка гибрида — F_h(d0) с новым s²
  h:
    5:
      design, design_sha, design_yaml: {file, sha}
      pravilo: single | mean | median | weights | ridge_noga
      komponenty:                                  # сырые прогнозы без s²; part = test (шаги 1–10) и val (шаги 7–10; нужен при posledovatelno)
        - {name: noga, file: test_locked/R2/L/<ряд>_h5_test.parquet}
        - {name: luchshiy, file: test_locked/R2/B/<ряд>_h5_test.parquet}
      weights / beta: константа | {1..6}           # как выбрано на 2017–2021; шаги 7–10 — шаг 6
      s2: {est: sm, kappa: 0.5, embargo: on, val_file: {file: data/regen/R2/cache/<...>_val.parquet, sha: ...}}
           # val_file — сырой итог дизайна (до калибровки) val шагов 1–6, вне test_locked: date, ticker, year, step, цель, rv_pred
      factors / mnozh_file / factor_itog / mnozh_itog_file — множители без s² (не калибровка s²); при s2 калибровку s² не дублировать
    22:                                            # E1 (+E8) — финалисты R2 на h=5 и 22 (решение владельца 21:05)
      design: E1_E8_h22
      pravilo: e1
      e1:
        n_hyb: 1 | 5                               # h=5: 1; h=22: 1 — (hyb₁ + 21·HAR₂₋₂₂)/22, 5 — (hyb₅ + 17·HAR₆₋₂₂)/22
        har:  {file: test_locked/R2/<агент>/<HAR подгоризонта>_test.parquet, s2: {est, kappa, embargo, val_file: {file, sha}}}
        noga: {file: test_locked/R2/<агент>/<нога h>_test.parquet, s2: {...}}     # нога финалиста; s² — t−1 с эмбарго
        e8: {lam: 0.5, k: 0 | 3 | 5 | 10, w: 0.5, g_file: test_locked/R2/G/<ворота>_h22_test.parquet}   # константы разработки; g_file при k > 0
      hyb — финал сборки своего h (победитель h=1 / h=5 или F_h(d0) с s² сборки). halt (остановка торгов) применяется всегда.
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
TARGET = ('rv_actual', 'rv_target', 'y')  # R2-guard: целевые колонки, которые в тестовой части не загружаются никогда
KEY = ['date', 'ticker', 'step']
ID = ['date', 'ticker', 'year', 'step']
LAST = 6  # последний шаг разработки (val 2021)
RULES = ('single', 'mean', 'median', 'weights', 'ridge_noga', 'e1')
sys.path.insert(0, str(HERE))
import r2_gate_abstain as GA  # noqa: E402  E8 агента G: ворота (shock_flags, halt_g) и операции e8_e1 — импорт, не своя формула
S2_RULES = ('posledovatelno', 'zamorozka')
S2_WIN = {'sm': 3, 'smrob': 3, 'sm_roll1y': 1, 'sm_roll2y': 2, 'sm_pooled': None}  # окно в val-годах (как r2_calib.WIN_Y)
F_D0 = (None, 'F_h(d0)', 'F_h_d0')  # «победителя нет» в gate_E (r2_loyo.pobeditel пишет 'F_h(d0)')
VYR_OK = tuple(f'{n}_d0_h{h}_test.parquet' for n in ('etalon', 'final') for h in HS)  # для s² — только эти файлы vyr/
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


def s2_cfg(c, what):
    c = dict(c or {})
    est, kappa, emb = c.get('est', 'sm'), c.get('kappa', 0.5), c.get('embargo', 'on')
    if est not in S2_WIN or not isinstance(kappa, (int, float)) or emb not in ('on', 'off', True, False):
        raise Stop(f'{what}: s² {c} — сборка поддерживает s2_est {sorted(S2_WIN)}, κ — константа, embargo on/off')
    return dict(est=est, kappa=float(kappa), emb=emb in ('on', True))


# ---------------------------------------------------------------- ворота E и спецификация (до любого чтения test_locked)
def vorota(spec_path):
    """Ворота E пройдены на всех h, SHA каждого gate_E совпал со спецификацией, победители совпали. -> (spec, {h: c})."""
    spec = yaml.safe_load(Path(spec_path).read_text())
    if spec.get('status') != 'zamorozheno':
        raise Stop(f'{spec_path}: статус {spec.get("status")!r}, нужен zamorozheno — сборка только после ворот E')
    if spec.get('s2_pravilo') not in S2_RULES:
        raise Stop(f'{spec_path}: s2_pravilo {spec.get("s2_pravilo")!r} не из {S2_RULES}')
    s2_cfg(spec.get('s2_etalon'), 's2_etalon')
    gsha = {int(h): s for h, s in (spec.get('gate_E_sha') or {}).items()}
    hs = {int(h): c for h, c in (spec.get('h') or {}).items()}
    fd0 = {int(h) for h in (spec.get('F_h_d0') or [])}
    if set(hs) & fd0 or set(hs) | fd0 != set(HS):
        raise Stop(f'спецификация: h со сборкой {sorted(hs)} и с F_h(d0) {sorted(fd0)} должны разбивать {HS}')
    dsha = {int(h): s for h, s in (spec.get('gate_D_sha') or {}).items()}
    for h in HS:
        g = GATES / f'gate_E_h{h}.yaml'
        if not g.exists():  # ворота E не решались: допустимо, только если ворота D без лучшего дизайна (драйвер: на h остаётся F_h(d0))
            gd = GATES / f'gate_D_h{h}.yaml'
            if not gd.exists() or h not in dsha:
                raise Stop(f'нет {g} — ворота E на h={h} не пройдены (и нет gate_D_h{h} в спецификации)')
            if sha(gd) != dsha[h]:
                raise Stop(f'{gd.name}: SHA-256 не совпал со спецификацией')
            GD = yaml.safe_load(gd.read_text())
            if GD.get('phase') != 'D' or int(GD['h']) != h or GD.get('luchshiy') is not None:
                raise Stop(f'h={h}: нет gate_E, а gate_D с лучшим дизайном {GD.get("luchshiy")!r} — нужны ворота E')
            if h not in fd0:
                raise Stop(f'h={h}: по воротам D дизайна нет (F_h(d0)), а спецификация собирает {hs.get(h, {}).get("design")}')
            continue
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
        if pb.get('V0') and (c['pravilo'] != 'single' or len(c.get('komponenty') or []) != 1):
            raise Stop(f'h={h}: победил V0 — финал = эконометрическая компонента (single, одна компонента)')
        s2s = [c.get('s2')]
        if c['pravilo'] == 'e1':
            e1_check(c, h, set(hs))
            s2s += [c['e1']['har']['s2']]
        for s in s2s:
            if s is not None:
                s2_cfg(s, f'h={h}')
                if not (s.get('val_file') or {}).get('file'):
                    raise Stop(f'h={h}: s2 без val_file (сырой ряд val шагов 1–6)')
        for src in [c.get('design_yaml'), c.get('istochnik_parametrov')] + [(s or {}).get('val_file') for s in s2s]:  # вне test_locked, SHA совпал
            if src:
                p = Path(src['file']) if Path(src['file']).is_absolute() else MAIN / src['file']
                assert 'test_locked' not in p.parts, f'h={h}: {p.name} — только вне test_locked'
                if sha(p) != src['sha']:
                    raise Stop(f'h={h}: {p.name}: SHA-256 не совпал')
    return spec, hs


# ---------------------------------------------------------------- чтение test_locked (только белый список)
def allowed(r, s2=False):
    p = (REGEN / r).resolve() if not Path(r).is_absolute() else Path(r).resolve()
    lk = LOCKED.resolve()
    assert str(p).startswith(str(lk) + '/'), f'{r}: не в test_locked'
    assert not str(p).startswith(str(OUTDIR.resolve()) + '/'), f'{r}: выход сборки не читается'
    if 'vyr' in p.parts:
        assert s2 and p.parent == (lk / 'vyr') and p.name in VYR_OK, f'{r}: замороженные ряды — только эталон и F_h(d0) ВЫР-1 и только для s²'
    assert 'final3' not in p.parts, f'{r}: замороженные ряды первого раунда сборка не читает'
    assert p.exists(), f'{r}: нет файла'
    return p


def read_locked(r, keep, part='test', s2=False):
    """Тестовая часть (part = test или файл без part): только колонки белого списка, без цели."""
    p = allowed(r, s2)
    names = pq.read_schema(p).names
    use = [c for c in keep if c in names]
    assert use == list(keep), f'{p.name}: нет колонок {set(keep) - set(use)}'
    assert not set(use) & set(TARGET)  # R2-guard: целевая колонка не в белом списке
    if 'part' in names:
        d = pd.read_parquet(p, columns=use, filters=[('part', '==', part)])
    else:
        assert part == 'test', f'{p.name}: нет колонки part — val-части нет'
        d = pd.read_parquet(p, columns=use)
    assert not set(d.columns) & set(TARGET)  # R2-guard: целевая колонка не загружена
    READ.append((rel(p) + ('' if part == 'test' else f' [part = {part}, без цели]'), sha(p), len(d)))
    if 'step' not in d:  # ряд по дате (ворота G)
        assert len(d) and not d.date.duplicated().any(), f'{p.name}: пусто или дубли даты'
        return d
    assert len(d) and d.duplicated(KEY).sum() == 0 and set(d.step.unique()) <= set(range(1, 11)), f'{p.name}: пусто, дубли ключа или шаг вне 1–10'
    if part == 'val':
        assert d.step.between(LAST + 1, 10).all(), f'{p.name}: val в test_locked — только шаги 7–10'
    return d


def read_val_s2(r, col):
    """OOS-остатки для s² при открытии: part = val, шаги 7–10 (годы 2022–2025) — цель и сырой прогноз. Оценок качества нет."""
    p = allowed(r, s2=True)
    d = pd.read_parquet(p, columns=ID + [col, 'rv_actual'], filters=[('part', '==', 'val')])  # R2-guard-s2: цель только val шагов 7–10
    assert len(d) and d.step.between(LAST + 1, 10).all() and (d.year == 2015 + d.step).all(), f'{p.name}: val — не шаги 7–10'  # R2-guard-s2
    READ.append((rel(p) + ' [part = val, шаги 7–10: цель для s²]', sha(p), len(d)))  # R2-guard-s2
    return d.rename(columns={col: 'raw', 'rv_actual': 'yv'})  # R2-guard-s2


def read_dev(p, col, div=None):
    """Val шагов 1–6 вне test_locked (годы <= 2021): цель и сырой прогноз; div — {шаг: F} (сырой = col / F)."""
    p = Path(p)
    assert 'test_locked' not in p.resolve().parts, f'{p}: dev-часть — только вне test_locked'
    d = pd.read_parquet(p, columns=ID + ['rv_actual', col], filters=[('step', '<=', LAST)])  # R2-guard-s2: val 2016–2021, не тест
    assert len(d) and d.year.max() <= 2015 + LAST and d.step.max() <= LAST, f'{p}: dev-часть с годом > 2021'
    READ.append((rel(p) + ' [val шагов 1–6]', sha(p), len(d)))
    d = d.rename(columns={col: 'raw', 'rv_actual': 'yv'})  # R2-guard-s2
    if div is not None:
        d['raw'] = d.raw / d.step.map(div).values
    return d


def read_pred(r, part='test'):
    d = read_locked(r, KEEP, part)
    assert d.rv_pred.gt(0).all() and np.isfinite(d.rv_pred).all(), f'{r}: прогноз <= 0 или не конечен'
    return d


def read_mnozh(r, keys):
    d = read_locked(r, KEEP_M)
    m = keys.merge(d, on=KEY, how='left', validate='1:1')
    assert m.mnozh.notna().all() and (m.mnozh > 0).all(), f'{r}: множитель есть не у всех строк или <= 0'
    return m.mnozh.values


# ---------------------------------------------------------------- s² последовательно (t−1, эмбарго)
def s2_steps(val, test, h, est='sm', kappa=0.5, emb=True, zamorozka=False):
    """F к тесту шага k = exp(κ·s²); s² — по ln-остаткам e = ln(y + 1e-10) − ln raw val шагов окна (sm: max(1, k−2)..k) с y > 1e-12,
    то есть по годам < ty = 2016 + k (как gibrid_bias.s2_est / ВЫР-1 sm_F). emb — без строк, чьё окно цели [t+1, t+h] (h-й следующий
    торговый день тикера) заходит в ty. zamorozka — шаги k >= 7 = шаг 6. val: date, ticker, year, step, yv, raw; test — ключи. -> {k: F}."""
    cal = pd.concat([val[['date', 'ticker']], test[['date', 'ticker']]]).drop_duplicates().sort_values(['ticker', 'date'])
    cal['tend'] = cal.groupby('ticker').date.shift(-h)
    v = val.merge(cal, on=['date', 'ticker'], how='left', validate='m:1')
    v = v[v.yv > 1e-12]  # R2-guard-s2: строки с целью, как в R1
    e = np.log(v.yv.values + 1e-10) - np.log(v.raw.values)  # R2-guard-s2: ln-остаток для s² (множитель, не оценка качества)
    st, yr, tend = v.step.values, v.year.values, v.tend.values
    w = S2_WIN[est]
    F = {}
    for k in sorted(int(x) for x in np.unique(test.step)):
        if zamorozka and k > LAST:
            assert LAST in F, 'zamorozka: нет шага 6'
            F[k] = F[LAST]
            continue
        ty = 2016 + k
        m = (st <= k) & (st >= (1 if w is None else max(1, k - w + 1)))
        assert (yr[m] < ty).all(), f'k={k}: остаток года >= {ty}'
        if not {j for j in range(1 if w is None else max(1, k - w + 1), k + 1)} <= set(np.unique(st)):
            raise Stop(f'k={k}: нет val-шагов окна s²')
        if emb:
            m = m & (tend < np.datetime64(f'{ty}-01-01'))  # NaT (нет конца окна) — False: в пул не идёт
        x = e[m]
        if len(x) < 30:
            raise Stop(f'k={k}: для s² меньше 30 строк')
        s2 = float((1.4826 * np.median(np.abs(x - np.median(x)))) ** 2) if est == 'smrob' else float(np.var(x))
        F[k] = float(np.exp(kappa * s2))
    return F


def kontrol_r1(F_off, test, col, what):
    """Без эмбарго — правило R1: F по шагам обязан совпасть с F_sm файла ВЫР-1."""
    f = test.step.map(F_off).values
    if not np.allclose(f, test[col].values, rtol=1e-9, atol=0):
        raise Stop(f'{what}: s² без эмбарго не повторил F_sm ВЫР-1 (max отн. {np.max(np.abs(f / test[col].values - 1)):.2e})')


def etalony_s2(h, cfg, zam):
    """Эталоны (а), (б) d0 и F_h(d0) с s² по правилу сборки (t−1, эмбарго). Сырые ряды — ВЫР-1, без изменений."""
    et_f = f'test_locked/vyr/etalon_d0_h{h}_test.parquet'
    et = read_locked(et_f, tuple(ID + ['har_raw', 'har_corr', 'F_sm']), s2=True)
    ev = read_dev(REGEN / 'vyr' / f'etalon_d0_h{h}_val.parquet', 'har_raw')
    if not zam:
        ev = pd.concat([ev, read_val_s2(et_f, 'har_raw')], ignore_index=True)
    kw = dict(est=cfg['est'], kappa=cfg['kappa'], zamorozka=zam)
    Fon, Foff = s2_steps(ev, et, h, emb=cfg['emb'], **kw), s2_steps(ev, et, h, emb=False, **kw)
    if not zam and cfg['est'] == 'sm' and cfg['kappa'] == 0.5:
        kontrol_r1(Foff, et, 'F_sm', f'эталон (б) d0 h={h}')
    E = et[ID + ['har_raw']].assign(F_sm=et.step.map(Fon).values, F_sm_bez_emb=et.step.map(Foff).values)
    E['har_corr'] = E.har_raw * E.F_sm
    fd_f = f'test_locked/vyr/final_d0_h{h}_test.parquet'
    fd = read_locked(fd_f, tuple(ID + ['rv_pred', 'leg_raw', 'F_sm', 'design']), s2=True)
    assert np.allclose(fd.rv_pred, fd.leg_raw * fd.F_sm, rtol=1e-9), f'F_{h}(d0): rv_pred != leg_raw × F_sm'
    if h == 22:  # F_22(d0) = эталон (б) d0
        m = fd.merge(E[KEY + ['har_raw', 'F_sm', 'F_sm_bez_emb']], on=KEY, validate='1:1', suffixes=('', '_e'))
        if len(m) != len(fd) or not np.allclose(m.leg_raw, m.har_raw, rtol=1e-12):
            raise Stop('F_22(d0): leg_raw != har_raw эталона')
        Fo, Fb = Fon, Foff
    else:
        fv = pd.read_csv(REGEN / 'vyr' / 'vyr1' / f'factors_f{h}.csv')
        fv = fv[fv.seed == 42].set_index('step').Fv.to_dict()
        dv = read_dev(REGEN / 'R2' / 'cache' / 'VYR' / f'F_d0_h{h}_s42_val.parquet', 'rv_pred', div=fv)
        if not zam:
            dv = pd.concat([dv, read_val_s2(fd_f, 'leg_raw')], ignore_index=True)
        fdk = fd.rename(columns={'leg_raw': 'raw'})
        Fo, Fb = s2_steps(dv, fdk, h, emb=cfg['emb'], **kw), s2_steps(dv, fdk, h, emb=False, **kw)
        if not zam and cfg['est'] == 'sm' and cfg['kappa'] == 0.5:
            kontrol_r1(Fb, fd, 'F_sm', f'F_{h}(d0)')
    Fh = fd[ID + ['leg_raw', 'design']].assign(F_sm=fd.step.map(Fo).values, F_sm_bez_emb=fd.step.map(Fb).values)
    Fh['rv_pred'] = Fh.leg_raw * Fh.F_sm
    Fh['design'] = Fh.design.astype(str) + ' | s² ' + ('t−1, эмбарго' if cfg['emb'] else 't−1') + (' (шаги 7–10 = шаг 6)' if zam else '')
    srt = lambda d: d.sort_values(['step', 'date', 'ticker']).reset_index(drop=True)
    return srt(E[ID + ['har_raw', 'har_corr', 'F_sm', 'F_sm_bez_emb']]), srt(Fh[ID + ['rv_pred', 'leg_raw', 'F_sm', 'F_sm_bez_emb', 'design']])


# ---------------------------------------------------------------- сборка
def combine(c, h, part='test'):
    """Компонента × свой множитель (по шагам и/или построчный) → объединение по правилу → множители итога. Сырой итог — без s²."""
    K = c['komponenty']
    names = [k['name'] for k in K]
    assert len(set(names)) == len(names), f'h={h}: имена компонент повторяются'
    parts = []
    for comp in K:
        d = read_pred(comp['file'], part)
        f = par_steps(comp['factors'], d.step.values) if comp.get('factors') is not None else 1.0
        if comp.get('mnozh_file'):
            assert part == 'test', f'h={h}: построчный множитель только для теста'
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
        assert part == 'test', f'h={h}: построчный множитель итога только для теста'
        pred = pred * read_mnozh(c['mnozh_itog_file'], m[KEY])
    out = m[ID].assign(rv_pred=pred, design=c['design'])
    assert np.isfinite(out.rv_pred).all() and (out.rv_pred > 0).all(), f'h={h}: итог не конечен или <= 0'
    return out.sort_values(['step', 'date', 'ticker']).reset_index(drop=True)


def pobeditel(c, h, zam):
    """Итог победителя: сырой итог по правилу × F s² (t−1, эмбарго) — если в спецификации есть s2."""
    o = combine(c, h)
    if c.get('s2') is None:
        return o
    cfg = s2_cfg(c['s2'], f'h={h}')
    vf = c['s2']['val_file']['file']
    dev = read_dev(Path(vf) if Path(vf).is_absolute() else MAIN / vf, 'rv_pred')
    if not zam:
        raw7 = combine(c, h, part='val')
        y7 = read_val_s2(c['komponenty'][0]['file'], 'rv_pred')[KEY + ['yv']]
        v7 = raw7.merge(y7, on=KEY, how='inner', validate='1:1')
        assert len(v7) == len(raw7), f'h={h}: val 7–10 — цель не у всех строк сырого итога'
        dev = pd.concat([dev, v7.rename(columns={'rv_pred': 'raw'})[ID + ['raw', 'yv']]], ignore_index=True)
    F = s2_steps(dev, o, h, est=cfg['est'], kappa=cfg['kappa'], emb=cfg['emb'], zamorozka=zam)
    return o.assign(rv_pred=o.rv_pred * o.step.map(F).values)


# ---------------------------------------------------------------- E1 (+E8): разложение по горизонту, настройка под t (владелец 20:10, 20:20, 21:05)
def e1_check(c, h, hs):
    """Спецификация E1 (+E8): h ∈ {5, 22}; hyb — финал h=1 (или h=5 при h=22); λ, k, w — константы разработки; k > 0 — с g_file G."""
    e = c.get('e1') or {}
    nh = e.get('n_hyb')
    if h not in (5, 22) or nh not in ((1,) if h == 5 else (1, 5)):
        raise Stop(f'h={h}: E1 — h=5: hyb₁; h=22: hyb₁ или hyb₅ (n_hyb = {nh})')
    if not (e.get('har') or {}).get('file') or not (e['har'].get('s2') or {}):
        raise Stop(f'h={h}: E1 — нужен har: file (test_locked/R2/<агент>/) и s2 (с val_file)')
    e8 = e.get('e8') or {}
    for k in ('lam', 'k'):
        if not isinstance(e8.get(k), (int, float)) or isinstance(e8.get(k), bool):
            raise Stop(f'h={h}: E8 — {k} должен быть константой, выбранной на разработке (а не {e8.get(k)!r}; пока — ждёт G)')
    if 'w' in e8:
        raise Stop(f'h={h}: E8 — веса ноги w в E8 агента G нет (r2_gate_abstain.e8_e1: только λ и k)')
    if not (0 <= e8['lam'] <= 1 and e8['k'] in GA.E8_K):
        raise Stop(f'h={h}: E8 — λ ∈ [0, 1], k ∈ {GA.E8_K}')
    if (e8['k'] > 0) != bool(e8.get('g_file')):
        raise Stop(f'h={h}: E8 — при k > 0 нужен g_file ворот G (shock_fresh), при k = 0 его быть не должно')


def ryad_s2(r, h, zam, what):
    """Сырой ряд автора (part = test; part = val шагов 7–10 при posledovatelno) × F s² по t−1 с эмбарго окна h. -> ряд (ID, rv_pred)."""
    cfg = s2_cfg(r['s2'], what)
    o = read_pred(r['file'])
    vf = r['s2']['val_file']['file']
    dev = read_dev(Path(vf) if Path(vf).is_absolute() else MAIN / vf, 'rv_pred')
    if not zam:
        dev = pd.concat([dev, read_val_s2(r['file'], 'rv_pred')], ignore_index=True)
    F = s2_steps(dev, o, h, est=cfg['est'], kappa=cfg['kappa'], emb=cfg['emb'], zamorozka=zam)
    return o.assign(rv_pred=o.rv_pred * o.step.map(F).values)[ID + ['rv_pred']]


def tend_cal(df, h):
    """Конец окна цели: h-й следующий торговый день тикера по календарю ряда (как _tend{h} в r2_nogi: shift(−h) по тикеру); NaT — нет."""
    cal = df[['date', 'ticker']].drop_duplicates().sort_values(['ticker', 'date'])
    cal['tend'] = cal.groupby('ticker').date.shift(-h)
    return df[['date', 'ticker']].merge(cal, on=['date', 'ticker'], how='left').tend.values


def e8_ops():
    """Операции E8 агента G. Если G вынес их в функции (e8_ln, e8_uroven) — импорт; иначе — дословно выражения
    r2_gate_abstain.e8_e1 (f и mk; в ней это локальные lambda): ln ŷ = lnL + g·halt·λ·(ln E1 − lnL), ŷ = нога·exp(ln ŷ − lnL).
    Побайтное совпадение с рядами G на val проверяет proverka_G (Δ = 0)."""
    if hasattr(GA, 'e8_ln') and hasattr(GA, 'e8_uroven'):
        return GA.e8_ln, GA.e8_uroven
    f = lambda lnL, c, lam, g, hg: lnL + g * hg * lam * (c - lnL)  # = e8_e1: f(lam, k, c) с G[k] = g
    mk = lambda Lv, ln, lnL: Lv * np.exp(ln - lnL)  # = e8_e1: mk(ln)
    return f, mk


def g_k(dates, k, gser=None):
    """Ворота G по дате: k = 0 — единицы (без воздержания); k > 0 — ряд g по дате (GA.shock_flags на val; на тесте — файл G),
    как в e8_e1: dates.map(gk[k]).fillna(1.0)."""
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    return np.ones(len(d)) if int(k) == 0 else d.map(gser).fillna(1.0).values


def e8(Lv, p1, dates, tend, h, lam, k, gser=None):
    """E8 поверх E1 функциями G: нога Lv (эталон В2 = HAR-J d0 + s² с эмбарго, как в e8_e1), компонента p1 (ансамбль E1)."""
    f, mk = e8_ops()
    lnL = np.log(Lv)
    hg = GA.halt_g(dates, tend, h)  # halt G — всегда
    return mk(Lv, f(lnL, np.log(p1), float(lam), g_k(dates, k, gser), hg), lnL)


def e1(c, h, out, zam):
    """E1: h=5 — (hyb₁ + 4·HAR₂₋₅)/5; h=22 — (hyb₁ + 21·HAR₂₋₂₂)/22 или (hyb₅ + 17·HAR₆₋₂₂)/22 (уровни). hyb — финал сборки своего h
    (победитель или F_h(d0) с s² сборки); HAR подгоризонта — с s² (t−1, эмбарго окна h).
    E8 поверх — функциями G (e8): нога — эталон (б) d0 с эмбарго s² этой же сборки, g — ворота G (k > 0: файл G по дате), halt — всегда.
    λ, k — константы разработки (от G после вложенного счёта)."""
    e, e8p = c['e1'], c['e1']['e8']
    nh = int(e['n_hyb'])
    hyb = out.get(nh, out[('F', nh)])[KEY + ['rv_pred']].rename(columns={'rv_pred': 'hyb'})
    har = ryad_s2(e['har'], h, zam, f'h={h} HAR подгоризонта').rename(columns={'rv_pred': 'har'})
    leg = out[('E', h)][KEY + ['har_corr']].rename(columns={'har_corr': 'leg'})
    m = har.merge(hyb, on=KEY, how='inner', validate='1:1').merge(leg, on=KEY, how='inner', validate='1:1')
    if len(m) != len(har) or len(m) != len(leg):
        raise Stop(f'h={h}: E1 — строки HAR ({len(har)}), hyb{nh} и эталона ({len(leg)}) не совпали ({len(m)})')
    p1 = (nh * m.hyb.values + (h - nh) * m.har.values) / h
    gser = None
    if e8p['k'] > 0:
        G = read_locked(e8p['g_file'], ('date', 'g'))  # ворота G по дате (рыночный признак, как shock_flags)
        if G.date.duplicated().any() or not G.g.between(0, 1).all():
            raise Stop(f'h={h}: g ворот G — дубли дат или вне [0, 1]')
        gser = pd.Series(G.g.values, index=pd.to_datetime(G.date))
        miss = ~pd.to_datetime(m.date).isin(gser.index)
        if miss.any():
            raise Stop(f'h={h}: g ворот G нет у {int(miss.sum())} строк')
    pred = e8(m.leg.values, p1, m.date.values, tend_cal(m, h), h, e8p['lam'], e8p['k'], gser)
    o = m[ID].assign(rv_pred=pred, design=c['design'])
    assert np.isfinite(o.rv_pred).all() and (o.rv_pred > 0).all(), f'h={h}: E1 не конечен или <= 0'
    return o.sort_values(['step', 'date', 'ticker']).reset_index(drop=True)


def proverka_G(h, nm='E1'):
    """Val 2017–2021 (только разработка; test_locked не читается): E8 сборки (функции G) на входах G = ряды G_E8_<nm>__* кэша
    по всей сетке (λ, k) и всем seed, побайтно (Δ = 0). Входы — как в r2_gate_abstain.e8_e1."""
    import r2_core as C
    import r2_driver as DR
    import re
    A, _ = C.load_d0()
    assert A.date.dt.year.max() <= 2015 + LAST
    E = DR.etalon(h)
    hsh, ryad = GA.E8_RYADY_H[h][nm]
    seeds = list(GA.SDRV['E']['seeds_final'])
    S = GA._val_only(C.load_series(hsh, ryad=ryad, h=h, steps=list(range(1, 7))), ryad)
    Lg = E[KEY + ['year', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': 'leg'})  # R2-guard: val 2016–2021
    per = {s: S[S.seed.astype(str) == str(s)][KEY + ['rv_pred']].merge(Lg, on=KEY).sort_values(KEY).reset_index(drop=True) for s in seeds}
    base = per[seeds[0]]
    tend = base[['date', 'ticker']].merge(A[['date', 'ticker', f'_tend{h}']], on=['date', 'ticker'], how='left')[f'_tend{h}'].values
    mrv = GA.market_rv(A)
    X = C.load_series(hsh, h=h, steps=list(range(1, 7)))
    X = X[X.ryad.astype(str).str.startswith(f'G_E8_{nm}__')]
    if X.empty:
        raise Stop(f'h={h}: в кэше {hsh} нет рядов G_E8_{nm}__* — сверять не с чем')
    rows = []
    for ry, gX in X.groupby('ryad'):
        lam = float(re.search(r'lam=([0-9.]+)', ry).group(1)) if 'lam=' in ry else 1.0  # h=5 (k_lam): 'halt', 'k=3' — λ = 1
        mk_ = re.search(r'(?<![a-z])k=(\d+)', ry)
        k = int(mk_.group(1)) if mk_ else 0  # 'halt' и 'lam=…' без k — k = 0
        gser = GA.shock_flags(mrv, k, GA.E8_Q, int(GA.CFG['k16']['shock_fresh']['okno']), 'uroven') if k > 0 else None
        for s in seeds:
            mine = e8(base.leg.values, per[s].rv_pred.values, base.date.values, tend, h, lam, k, gser)
            g = gX[gX.seed.astype(str) == str(s)][KEY + ['rv_pred']].merge(base[KEY], on=KEY).sort_values(KEY)
            d = np.abs(g.rv_pred.values - mine).max() if len(g) == len(base) else np.inf
            rows.append(dict(h=h, ryad=ry, lam=lam, k=k, seed=s, n=len(g), max_abs_delta=float(d), ok=bool(d == 0)))
    R = pd.DataFrame(rows)
    print(f'proverka_G h={h} {nm}: {int(R.ok.sum())}/{len(R)} (ряд G_E8 × seed) побайтно, Δ = 0', flush=True)
    return R


def journal(lines):
    new = not JOURNAL.exists()
    with open(JOURNAL, 'a') as f:
        if new:
            f.write('# Журнал R2, агент A (сборка теста, второе открытие, FINAL_DESIGN_round2)\n')
        f.write('\n'.join(lines) + '\n')


def table(rows):
    return ['| файл | SHA-256 | строк |', '|---|---|---|'] + [f'| {f} | {s} | {n} |' for f, s, n in rows]


def put(df, name):
    """Запись в выход; повтор обязан дать те же байты."""
    assert not set(df.columns) & set(TARGET)  # R2-guard: цели в выходе нет
    f = OUTDIR / name
    tmp = OUTDIR / f'.{name}.new'
    df.to_parquet(tmp, index=False)
    if f.exists():
        same = sha(tmp) == sha(f)
        tmp.unlink()
        if not same:
            raise Stop(f'{f.name}: повтор сборки дал другие байты — СТОП (спецификация или входы изменены)')
    else:
        tmp.rename(f)
    WROTE.append((rel(f), sha(f), len(df)))


def run(spec_path):
    spec_path = Path(spec_path)
    spec, hs = vorota(spec_path)  # до любого чтения test_locked
    zam = spec['s2_pravilo'] == 'zamorozka'
    cfg = s2_cfg(spec.get('s2_etalon'), 's2_etalon')
    s_spec, s_code = sha(spec_path), sha(__file__)
    journal([f'\n## Сборка теста R2 — старт {now()}', '',
             f'Спецификация {rel(spec_path)} SHA-256 {s_spec}; код r2_sborka_test.py SHA-256 {s_code}.',
             'Ворота E: ' + '; '.join(f'h={h} {s}' for h, s in sorted(spec['gate_E_sha'].items())) + '.',
             f'Сборка на h = {sorted(hs)}; F_h(d0) на h = {sorted(spec.get("F_h_d0") or [])}; s²: {spec["s2_pravilo"]}, эталон {cfg}.',
             'test_locked до этой строки не читался.'])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for h in HS:
        E, Fh = etalony_s2(h, cfg, zam)
        put(E, f'etalon_d0_s2emb_h{h}_test.parquet')
        put(Fh, f'final_d0_s2emb_h{h}_test.parquet')
        out[('E', h)], out[('F', h)] = E, Fh
        print(f'h={h}: эталон (б) d0 и F_h(d0) с s² ({spec["s2_pravilo"]}, эмбарго {cfg["emb"]}): F шагов 7–10 '
              f'{[round(float(x), 4) for x in E.groupby("step").F_sm.first().loc[7:].values]}; оценок качества нет', flush=True)
    for h, c in sorted(hs.items()):  # по возрастанию h: hyb₁ и hyb₅ для E1 уже собраны
        o = e1(c, h, out, zam) if c['pravilo'] == 'e1' else pobeditel(c, h, zam)
        put(o, f'final_r2_h{h}_test.parquet')
        out[h] = o
        print(f'h={h}: {c["design"]} -> final_r2_h{h}_test.parquet ({len(o)} строк, шаги {o.step.min()}–{o.step.max()}); оценок качества нет', flush=True)
    (OUTDIR / 'SHA256.txt').write_text(''.join(f'{s}  {Path(p).name}\n' for p, s, _ in WROTE)
                                       + f'{s_spec}  {spec_path.name} (спецификация)\n{s_code}  r2_sborka_test.py (код)\n')
    journal(['', 'Прочитано (тестовая часть — без цели; цель — только val шагов 7–10 для s² и val 1–6 вне test_locked):', ''] + table(READ)
            + ['', 'Записано (цели нет):', ''] + table(WROTE) + ['', f'Конец {now()}.'])
    return out


# ---------------------------------------------------------------- самопроверка (синтетика во временном каталоге)
def _synth_vyr(rng, h, tickers, BI):
    """ВЫР-1 в миниатюре: val шагов 1–10 (годы 2016–2025), тест шагов 1–10 (годы 2017–2026), F_sm по правилу R1 (gibrid_bias.s2_est)."""
    dates = pd.bdate_range('2016-01-04', '2026-02-27')
    g = pd.DataFrame([(d, t) for d in dates for t in tickers], columns=['date', 'ticker'])
    g = g[~((g.ticker == tickers[-1]) & g.date.between('2019-12-20', '2019-12-31'))]  # у тикера пропуск торгов в конце года
    g['year'] = g.date.dt.year
    s = np.exp(rng.normal(-8, .4, len(g)))
    g['y'] = s * np.exp(rng.normal(0, .5, len(g)))
    g.loc[g.groupby('ticker').cumcount(ascending=False) < h, 'y'] = np.nan  # конец данных: цели нет
    g['raw'] = s * np.exp(rng.normal(0.1 * (g.date.dt.month == 12), .3, len(g)))  # декабрь смещён: эмбарго меняет s²
    V = g[g.year <= 2025].assign(step=lambda x: x.year - 2015)
    T = g[g.year >= 2017].assign(step=lambda x: x.year - 2016)
    v = V[V.y > 1e-12]
    E = {'c': {int(st): np.log(q.y.values + 1e-10) - np.log(q.raw.values) for st, q in v.groupby('step')}}
    Ft = {k: float(np.exp(BI.s2_est('sm', E, 'c', k) / 2)) for k in range(1, 11)}
    Fv = {k: 1.0 if k == 1 else float(np.exp(BI.s2_est('sm', E, 'c', k - 1) / 2)) for k in range(1, 11)}
    return V, T, Ft, Fv, g


def _ref_emb(g, h, k, w=3):
    """Независимая сверка эмбарго: из пула шагов max(1, k−2)..k исключаются последние h торговых дней тикера до 1 янв. ty."""
    ty = 2016 + k
    G = g.sort_values(['ticker', 'date']).copy()
    G['pos'] = G.groupby('ticker').cumcount()
    last = G[G.date < f'{ty}-01-01'].groupby('ticker').pos.max()
    G = G[(G.year >= ty - w) & (G.year <= ty - 1) & (G.y > 1e-12)]
    G = G[G.pos <= G.ticker.map(last) - h]
    e = np.log(G.y + 1e-10) - np.log(G.raw)
    return float(np.exp(np.var(e.values) / 2))


def selftest():
    global REGEN, LOCKED, OUTDIR, GATES, JOURNAL
    sys.path.insert(0, str(HERE))
    import gibrid_bias as BI  # правило R1 — независимая сверка «без эмбарго»
    saved = (REGEN, LOCKED, OUTDIR, GATES, JOURNAL)
    tmp = Path(tempfile.mkdtemp(prefix='r2_sborka_selftest_'))
    REGEN, JOURNAL = tmp / 'regen', tmp / 'dev_log_r2_A.md'
    LOCKED, GATES = REGEN / 'test_locked', REGEN / 'R2' / 'O'
    OUTDIR = LOCKED / 'R2' / 'final'
    try:
        rng = np.random.default_rng(13)
        TK = ('SBER', 'LKOH', 'AFLT')
        for d in (LOCKED / 'R2' / 'B', LOCKED / 'R2' / 'L', LOCKED / 'vyr', GATES, REGEN / 'R2' / 'val', REGEN / 'vyr' / 'vyr1',
                  REGEN / 'R2' / 'cache' / 'VYR'):
            d.mkdir(parents=True)
        SY = {}
        for h in HS:  # ВЫР-1: эталон и F_h(d0); цель тестовой части — −1 (если бы её прочли, s² сломался бы)
            V, T, Ft, Fv, g = _synth_vyr(rng, h, TK, BI)
            SY[h] = (V, T, Ft, Fv, g)
            o = lambda d, F, part: d.rename(columns={'y': 'rv_actual', 'raw': 'har_raw'}).assign(  # R2-guard: синтетика
                F_sm=d.step.map(F).values, har_corr=d.raw * d.step.map(F).values, part=part)
            V[V.step <= 6].pipe(o, Fv, 'val').drop(columns='part').to_parquet(REGEN / 'vyr' / f'etalon_d0_h{h}_val.parquet', index=False)
            pd.concat([o(V[V.step >= 7], Fv, 'val'), o(T.assign(y=-1.0), Ft, 'test')]).to_parquet(  # R2-guard: цель теста — заглушка
                LOCKED / 'vyr' / f'etalon_d0_h{h}_test.parquet', index=False)
            if h == 22:
                fV, fT, fFt, fFv = V, T, Ft, Fv
            else:
                fV, fT, fFt, fFv, _ = _synth_vyr(rng, h, TK, BI)
                SY[(h, 'F')] = (fV, fT, fFt, fFv, _)
                pd.DataFrame([dict(h=h, seed=s, step=k, Fv=fFv[k] * (1 if s == 42 else 2), Ft=fFt[k]) for s in (42, 43) for k in range(1, 11)]).to_csv(
                    REGEN / 'vyr' / 'vyr1' / f'factors_f{h}.csv', index=False)
                fV[fV.step <= 6].assign(rv_pred=lambda x: x.raw * x.step.map(fFv)).rename(columns={'y': 'rv_actual'})[  # R2-guard: синтетика
                    ID + ['rv_actual', 'rv_pred']].to_parquet(REGEN / 'R2' / 'cache' / 'VYR' / f'F_d0_h{h}_s42_val.parquet', index=False)  # R2-guard
            of = lambda d, F, part: d.rename(columns={'y': 'rv_actual', 'raw': 'leg_raw'}).assign(  # R2-guard: синтетика
                F_sm=d.step.map(F).values, rv_pred=d.raw * d.step.map(F).values, part=part, design=f'F_{h} d0')
            pd.concat([of(fV[fV.step >= 7], fFv, 'val'), of(fT.assign(y=-1.0), fFt, 'test')]).to_parquet(  # R2-guard: цель теста — заглушка
                LOCKED / 'vyr' / f'final_d0_h{h}_test.parquet', index=False)
        # компоненты победителей: part = test (шаги 1–10) и val (шаги 7–10); цель теста — заглушка
        T5, V5 = SY[5][1][ID], SY[5][0][ID + ['y']]
        nb_t, lb_t = (T5.assign(rv_pred=np.exp(rng.normal(-8, .3, len(T5)))) for _ in range(2))
        nb_v, lb_v = (V5[V5.step >= 7].assign(rv_pred=np.exp(rng.normal(-8, .3, int((V5.step >= 7).sum())))) for _ in range(2))
        pd.concat([nb_t.assign(part='test', rv_actual=-1.0), nb_v.rename(columns={'y': 'rv_actual'}).assign(part='val')]).to_parquet(  # R2-guard
            LOCKED / 'R2' / 'L' / 'noga_h5_test.parquet', index=False)
        pd.concat([lb_t.assign(part='test'), lb_v.drop(columns='y').assign(part='val')]).to_parquet(LOCKED / 'R2' / 'B' / 'luchshiy_h5_test.parquet', index=False)
        dev5 = V5[V5.step <= 6].assign(rv_pred=np.exp(rng.normal(-8, .3, int((V5.step <= 6).sum())))).rename(columns={'y': 'rv_actual'})  # R2-guard
        pdv = REGEN / 'R2' / 'val' / 'itog_h5_val.parquet'
        dev5.to_parquet(pdv, index=False)
        T22 = SY[22][1][ID]
        n22 = T22.assign(rv_pred=np.exp(rng.normal(-8, .3, len(T22))))
        n22.to_parquet(LOCKED / 'R2' / 'L' / 'noga_h22_test.parquet', index=False)  # без part: только тест
        mz = n22[KEY].assign(mnozh=np.exp(rng.normal(0, .05, len(n22))))
        mz.to_parquet(LOCKED / 'R2' / 'L' / 'mnozh_h22_test.parquet', index=False)
        pv = REGEN / 'R2' / 'val' / 'factors_h22.csv'
        pv.write_text('step,F\n1,1.1\n')
        F = {k: 1 + k / 100 for k in range(1, 7)}
        B = {k: 0.5 + k / 100 for k in range(1, 7)}
        gates = {1: dict(phase='E', h=1, pobeditel=dict(pobeditel='F_h(d0)', V0=False, poryadok=['u1'])),
                 5: dict(phase='E', h=5, pobeditel=dict(pobeditel='D_h5_x', V0=False, poryadok=['D_h5_x']), design_sha='a' * 64),
                 22: dict(phase='E', h=22, pobeditel=dict(pobeditel='V0_HARX_h22', V0=True, poryadok=['u2', 'V0_HARX_h22']))}
        for h, g in gates.items():
            (GATES / f'gate_E_h{h}.yaml').write_text(yaml.safe_dump(g, allow_unicode=True))
        spec = dict(status='zamorozheno', s2_pravilo='posledovatelno', s2_etalon=dict(est='sm', kappa=0.5, embargo='on'),
                    gate_E_sha={h: sha(GATES / f'gate_E_h{h}.yaml') for h in HS}, F_h_d0=[1], h={
            5: dict(design='D_h5_x', design_sha='a' * 64, pravilo='ridge_noga', beta=B,
                    komponenty=[dict(name='noga', file='test_locked/R2/L/noga_h5_test.parquet', factors=1.02),
                                dict(name='luchshiy', file='test_locked/R2/B/luchshiy_h5_test.parquet')],
                    s2=dict(est='sm', kappa=0.5, embargo='on', val_file=dict(file=str(pdv), sha=sha(pdv)))),
            22: dict(design='V0_HARX_h22', design_sha='b' * 64, pravilo='single', factor_itog=F, mnozh_itog_file='test_locked/R2/L/mnozh_h22_test.parquet',
                     komponenty=[dict(name='noga', file='test_locked/R2/L/noga_h22_test.parquet')], istochnik_parametrov=dict(file=str(pv), sha=sha(pv)))})
        sp = tmp / 'spec.yaml'

        def putspec(s, p=sp):
            p.write_text(yaml.safe_dump(s, allow_unicode=True, sort_keys=False))
            return p

        def stop(s, what, exc=Stop):
            READ.clear(); WROTE.clear()
            try:
                run(putspec(s, tmp / 'bad.yaml'))
            except exc as e:
                assert what in str(e), (what, e)
                return
            raise AssertionError(f'не пойман: {what}')

        # 1. s² последовательно: без эмбарго = R1 (F_sm ВЫР-1), с эмбарго = независимая сверка; F_22(d0) = эталон (б)
        out = run(putspec(spec))
        for h in HS:
            V, T, Ft, Fv, g = SY[h]
            E = out[('E', h)].groupby('step').first()
            assert np.allclose(E.F_sm_bez_emb.values, [Ft[k] for k in E.index], rtol=1e-12), f'h={h}: без эмбарго != R1'
            ref = [_ref_emb(g, h, k) for k in E.index]
            assert np.allclose(E.F_sm.values, ref, rtol=1e-12), f'h={h}: эмбарго != независимая сверка'
            assert (np.abs(E.F_sm.values / E.F_sm_bez_emb.values - 1) > 1e-6).any(), f'h={h}: эмбарго ничего не изменило'
            assert not {'rv_actual', 'part'} & set(pq.read_schema(OUTDIR / f'etalon_d0_s2emb_h{h}_test.parquet').names)  # R2-guard
            Fh = out[('F', h)].groupby('step').first()
            if h == 22:
                assert np.allclose(Fh.F_sm, E.F_sm, rtol=1e-12)
            else:
                fg = SY[(h, 'F')][4]
                assert np.allclose(Fh.F_sm.values, [_ref_emb(fg, h, k) for k in Fh.index], rtol=1e-12), f'F_{h}(d0): эмбарго'
                assert np.allclose(Fh.F_sm_bez_emb.values, [SY[(h, 'F')][2][k] for k in Fh.index], rtol=1e-12), f'F_{h}(d0): R1'
        # победитель h=5: сырой итог ridge_noga × F по t−1 с эмбарго (val 1–6 — val_file, 7–10 — part = val компонент)
        o = out[5].merge(nb_t.rename(columns={'rv_pred': 'n'})[KEY + ['n']], on=KEY).merge(lb_t.rename(columns={'rv_pred': 'b'})[KEY + ['b']], on=KEY)
        rr = lambda n, b, k: np.exp(np.log(n * 1.02) + k.clip(upper=6).map(B).values * (np.log(b) - np.log(n * 1.02)))
        v7 = nb_v.merge(lb_v[KEY + ['rv_pred']], on=KEY, suffixes=('', '_b'))
        gw = pd.concat([dev5.rename(columns={'rv_actual': 'y', 'rv_pred': 'raw'}),  # R2-guard
                        v7.assign(raw=rr(v7.rv_pred.values, v7.rv_pred_b.values, v7.step))[ID + ['y', 'raw']]])
        gw = gw.merge(pd.concat([gw, T5])[['date', 'ticker']].drop_duplicates(), how='outer').assign(year=lambda x: x.date.dt.year)
        Fw = {k: _ref_emb(gw, 5, k) for k in range(1, 11)}
        want = rr(o.n.values, o.b.values, o.step) * o.step.map(Fw).values
        assert np.allclose(o.rv_pred, want, rtol=1e-12), 'победитель h=5: сырой итог × F (t−1, эмбарго)'
        # V0 h=22 без s2: множители заморожены на шаге 6, построчный множитель
        o22 = out[22].merge(n22.rename(columns={'rv_pred': 'n'})[KEY + ['n']], on=KEY).merge(mz, on=KEY)
        assert np.allclose(o22.rv_pred, o22.n * o22.step.clip(upper=6).map(F) * o22.mnozh, rtol=1e-12)
        assert not (OUTDIR / 'final_r2_h1_test.parquet').exists() and (OUTDIR / 'final_d0_s2emb_h1_test.parquet').exists()
        assert all('rv_actual' not in pq.read_schema(f).names for f in OUTDIR.glob('*.parquet'))  # R2-guard
        j = JOURNAL.read_text()
        assert sha(sp) in j and 'шаги 7–10: цель для s²' in j and j.index('test_locked до этой строки не читался') < j.index('Прочитано')
        assert not any('final3' in r for r, _, _ in READ)
        # 2. повтор — те же байты; изменённый вход — СТОП
        READ.clear(); WROTE.clear()
        run(sp)
        stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'factor_itog': {**F, 6: 1.5}}}}, 'другие байты')
        stop({**spec, 's2_etalon': dict(est='sm', kappa=0.5, embargo='off')}, 'другие байты')
        # 3. заморозка шага 6 (протокол, раздел 8): цель из test_locked не читается вовсе; шаги 7–10 = шаг 6
        z = tmp / 'zam'
        OUTDIR = z / 'final'
        READ.clear(); WROTE.clear()
        oz = run(putspec({**spec, 's2_pravilo': 'zamorozka'}, tmp / 'spec_z.yaml'))
        assert not any('цель для s²' in r for r, _, _ in READ), 'zamorozka: цель val 7–10 не читается'
        Ez = oz[('E', 5)].groupby('step').F_sm.first()
        assert np.allclose(Ez.loc[7:], Ez.loc[6]) and np.allclose(Ez.loc[:6], out[('E', 5)].groupby('step').F_sm.first().loc[:6])
        OUTDIR = LOCKED / 'R2' / 'final'
        # 4. ворота и спецификация
        stop({**spec, 'status': 'ждёт ворот E'}, 'zamorozheno')
        stop({**spec, 's2_pravilo': 'daily'}, 's2_pravilo')
        stop({**spec, 'h': {**spec['h'], 5: {**spec['h'][5], 's2': dict(est='sm_by_ticker', val_file=spec['h'][5]['s2']['val_file'])}}}, 'поддерживает')
        stop({**spec, 'h': {**spec['h'], 5: {**spec['h'][5], 's2': dict(est='sm', kappa='fit', val_file=spec['h'][5]['s2']['val_file'])}}}, 'κ')
        stop({**spec, 'h': {**spec['h'], 5: {**spec['h'][5], 's2': dict(est='sm', val_file=dict(file=str(pdv), sha='0' * 64))}}}, 'SHA-256 не совпал')
        stop({**spec, 'F_h_d0': []}, 'разбивать')
        stop({**spec, 'h': {**spec['h'], 5: {**spec['h'][5], 'design_sha': 'c' * 64}}}, 'ворота')
        stop({**spec, 'h': {**spec['h'], 5: {**spec['h'][5], 'pravilo': 'stack'}}}, 'правило')
        stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'pravilo': 'mean'}}}, 'V0')
        g5 = GATES / 'gate_E_h5.yaml'
        g5.write_text(g5.read_text() + '# правка\n')
        stop(spec, 'SHA-256 не совпал')
        g5.unlink()
        stop(spec, 'нет')
        (GATES / 'gate_E_h5.yaml').write_text(yaml.safe_dump(gates[5], allow_unicode=True))
        # 5. R1-контроль: F_sm ВЫР-1 не тот — СТОП (s² сборки не равен правилу R1 без эмбарго)
        ef = LOCKED / 'vyr' / 'etalon_d0_h1_test.parquet'
        keep = ef.read_bytes()
        x = pd.read_parquet(ef)
        x.loc[(x.part == 'test') & (x.step == 8), 'F_sm'] *= 1.01
        x.to_parquet(ef, index=False)
        stop(spec, 'не повторил F_sm')
        ef.write_bytes(keep)
        # 6. параметр с ключом шага > 6; пути: вне test_locked, чужой файл vyr/, final3, выход; компонента без val при s2
        stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'factor_itog': {**F, 7: 1.3}}}}, '>= 2022', AssertionError)
        (LOCKED / 'vyr' / 'ref_x_h22_test.parquet').write_bytes((LOCKED / 'R2' / 'L' / 'noga_h22_test.parquet').read_bytes())
        (LOCKED / 'final3').mkdir()
        (LOCKED / 'final3' / 'final_h22_test.parquet').write_bytes((LOCKED / 'R2' / 'L' / 'noga_h22_test.parquet').read_bytes())
        for bad, what in (('R2/val/factors_h22.csv', 'не в test_locked'), ('test_locked/vyr/ref_x_h22_test.parquet', 'замороженные'),
                          ('test_locked/vyr/etalon_d0_h22_test.parquet', 'замороженные'), ('test_locked/final3/final_h22_test.parquet', 'первого раунда'),
                          ('test_locked/R2/final/final_r2_h5_test.parquet', 'выход сборки')):
            stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'komponenty': [dict(name='noga', file=bad)]}}}, what, AssertionError)
        stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 's2': spec['h'][5]['s2']}}}, 'val-части нет', AssertionError)
        # 7. белый список: файла без rv_pred — отказ
        n22[ID].assign(rv_actual=1.0).to_parquet(LOCKED / 'R2' / 'L' / 'bez_pred_h22_test.parquet', index=False)  # R2-guard
        stop({**spec, 'h': {**spec['h'], 22: {**spec['h'][22], 'komponenty': [dict(name='noga', file='test_locked/R2/L/bez_pred_h22_test.parquet')]}}},
             'нет колонок', AssertionError)
        # 8. E1 (+E8 функциями G): h=5 — (hyb₁ + 4·HAR₂₋₅)/5, k = 3 (g по дате из файла G) × halt; h=22 — (hyb₅ + 17·HAR₆₋₂₂)/22, k = 0
        (LOCKED / 'R2' / 'M').mkdir(); (LOCKED / 'R2' / 'G').mkdir()
        SR = {}

        def ser(name, h):
            V, T, _, _, g = _synth_vyr(rng, h, TK, BI)
            pd.concat([T[ID].assign(rv_pred=T.raw.values, rv_actual=-1.0, part='test'),  # R2-guard: цель теста — заглушка
                       V[V.step >= 7][ID].assign(rv_pred=V[V.step >= 7].raw.values, rv_actual=V[V.step >= 7].y.values, part='val')]).to_parquet(  # R2-guard
                LOCKED / 'R2' / 'M' / f'{name}_test.parquet', index=False)
            pv_ = REGEN / 'R2' / 'val' / f'{name}_val.parquet'
            V[V.step <= 6][ID].assign(rv_actual=V[V.step <= 6].y.values, rv_pred=V[V.step <= 6].raw.values).to_parquet(pv_, index=False)  # R2-guard
            SR[name] = (T, g)
            return dict(file=f'test_locked/R2/M/{name}_test.parquet', s2=dict(est='sm', kappa=0.5, embargo='on', val_file=dict(file=str(pv_), sha=sha(pv_))))
        H5, H22 = ser('har_2_5_h5', 5), ser('har_6_22_h22', 22)
        # ворота G по дате: GA.shock_flags на синтетической RV рынка (на тесте этот файл пишет G)
        dts = pd.Series(sorted(SR['har_2_5_h5'][0].date.unique()))
        mrv = pd.Series(np.exp(rng.normal(-8, .5, len(dts))), index=pd.to_datetime(dts))
        gs = GA.shock_flags(mrv, 3, GA.E8_Q, int(GA.CFG['k16']['shock_fresh']['okno']), 'uroven')
        assert (gs == 0).any() and (gs == 1).any()
        pd.DataFrame({'date': gs.index, 'g': gs.values}).to_parquet(LOCKED / 'R2' / 'G' / 'g_h5_test.parquet', index=False)
        ge = {5: dict(phase='E', h=5, pobeditel=dict(pobeditel='E1_E8_h5', V0=False, poryadok=['E1_E8_h5'])),
              22: dict(phase='E', h=22, pobeditel=dict(pobeditel='E1_E8_h22', V0=False, poryadok=['E1_E8_h22']))}
        for h, gd in ge.items():
            (GATES / f'gate_E_h{h}.yaml').write_text(yaml.safe_dump(gd, allow_unicode=True))
        E8_5, E8_22 = dict(lam=0.75, k=3, g_file='test_locked/R2/G/g_h5_test.parquet'), dict(lam=0.5, k=0)
        sp_e = {**spec, 'gate_E_sha': {h: sha(GATES / f'gate_E_h{h}.yaml') for h in HS}, 'h': {
            5: dict(design='E1_E8_h5', pravilo='e1', e1=dict(n_hyb=1, har=H5, e8=E8_5)),
            22: dict(design='E1_E8_h22', pravilo='e1', e1=dict(n_hyb=5, har=H22, e8=E8_22))}}
        OUTDIR = tmp / 'e1' / 'final'
        READ.clear(); WROTE.clear()
        oe = run(putspec(sp_e, tmp / 'spec_e1.yaml'))
        cal = lambda T, h, F: T[ID].assign(rv_pred=T.raw.values * T.step.map(F).values)

        def want_e1(h, nh, hyb, e8p, gser=None):
            """Дословно операции r2_gate_abstain.e8_e1 (f, mk) с воротами G по дате и halt G; нога — эталон (б) d0 сборки."""
            n = 'har_2_5_h5' if h == 5 else 'har_6_22_h22'
            Fh = {k: _ref_emb(SR[n][1], h, k) for k in range(1, 11)}
            m = cal(SR[n][0], h, Fh).rename(columns={'rv_pred': 'har'}).merge(hyb[KEY + ['rv_pred']].rename(columns={'rv_pred': 'hyb'}), on=KEY)
            m = m.merge(oe[('E', h)][KEY + ['har_corr']], on=KEY).sort_values(['ticker', 'date'])
            cl = m[['date', 'ticker']].drop_duplicates().sort_values(['ticker', 'date'])
            cl['te'] = cl.groupby('ticker').date.shift(-h)
            te = m[['date', 'ticker']].merge(cl, on=['date', 'ticker'], how='left').te.values
            hg = GA.halt_g(m.date.values, te, h)
            G = np.ones(len(m)) if gser is None else pd.to_datetime(m.date).map(gser).fillna(1.0).values
            Lv = m.har_corr.values
            lnL = np.log(Lv)
            c = np.log((nh * m.hyb.values + (h - nh) * m.har.values) / h)
            f = lambda lam: lnL + G * hg * lam * (c - lnL)  # r2_gate_abstain.e8_e1, стр. f
            mk = lambda ln: Lv * np.exp(ln - lnL)  # r2_gate_abstain.e8_e1, стр. mk
            return m.assign(want=mk(f(float(e8p['lam']))))[KEY + ['want']], int(((G * hg) == 0).sum()), int((hg == 0).sum())
        w5, ng5, _ = want_e1(5, 1, oe[('F', 1)], E8_5, gs)
        x = oe[5].merge(w5, on=KEY, validate='1:1')
        assert len(x) == len(oe[5]) and np.allclose(x.rv_pred, x.want, rtol=1e-12), 'E1 (+E8) h=5 (s² — независимой сверкой)'
        assert ng5 > 0
        w22, _, nh22 = want_e1(22, 5, oe[5], E8_22)
        x = oe[22].merge(w22, on=KEY, validate='1:1')
        assert len(x) == len(oe[22]) and np.allclose(x.rv_pred, x.want, rtol=1e-12), 'E1 (+E8) h=22 (hyb₅ = E1 h=5)'
        assert nh22 > 0, 'halt G должен обнулять g у окон через остановку торгов'
        # E8 побайтно: на одних входах e8() сборки = операции e8_e1 G (f, mk, ворота по дате, halt_g) — Δ = 0 по всей сетке (λ, k)
        nn = 4000
        dz = pd.Series(pd.to_datetime(rng.choice(dts.values, nn)))
        Lv_, p_ = np.exp(rng.normal(-8, .4, nn)), np.exp(rng.normal(-8, .4, nn))
        te_ = (dz + pd.offsets.BDay(22)).values
        for k in GA.E8_K:
            gk = GA.shock_flags(mrv, k, GA.E8_Q, int(GA.CFG['k16']['shock_fresh']['okno']), 'uroven') if k > 0 else None
            Gk = np.ones(nn) if gk is None else pd.to_datetime(dz).map(gk).fillna(1.0).values
            hg_ = GA.halt_g(dz.values, te_, 22)
            lnL_, c_ = np.log(Lv_), np.log(p_)
            for lam in GA.E8_LAM:
                ref = Lv_ * np.exp((lnL_ + Gk * hg_ * lam * (c_ - lnL_)) - lnL_)
                assert (e8(Lv_, p_, dz.values, te_, 22, lam, k, gk) == ref).all(), f'E8 λ={lam}, k={k}: не побайтно'
        lv0 = e8(np.array([1.5, 2.0]), np.array([3.0, 0.5]), pd.to_datetime(['2019-01-10', '2019-01-11']).values,
                 pd.to_datetime(['2019-02-10', '2019-02-11']).values, 22, 0.0, 0)
        assert (lv0 == np.array([1.5, 2.0])).all(), 'λ = 0 — ровно нога (как mk G)'
        assert e8_ops()[0].__code__.co_filename == __file__ or hasattr(GA, 'e8_ln')
        assert any('цель для s²' in r and 'har_6_22' in r for r, _, _ in READ) and any('g_h5_test' in r for r, _, _ in READ)
        for bad_e8, what in ((dict(E8_5, lam=[0.5, 1.0]), 'константой'), ({k: v for k, v in E8_5.items() if k != 'g_file'}, 'g_file'),
                             (dict(E8_22, g_file='test_locked/R2/G/g_h5_test.parquet'), 'g_file'), (dict(E8_5, w=0.5), 'веса ноги'),
                             (dict(E8_5, k=4), 'k ∈'), (dict(E8_5, lam='ждёт G'), 'константой')):
            stop({**sp_e, 'h': {**sp_e['h'], 5: {**sp_e['h'][5], 'e1': {**sp_e['h'][5]['e1'], 'e8': bad_e8}}}}, what)
        stop({**sp_e, 'h': {**sp_e['h'], 5: {**sp_e['h'][5], 'e1': {**sp_e['h'][5]['e1'], 'n_hyb': 5}}}}, 'n_hyb')
        stop({**sp_e, 'h': {**sp_e['h'], 22: {**sp_e['h'][22], 'e1': {**sp_e['h'][22]['e1'], 'har': {'file': H22['file']}}}}}, 'нужен har')
        # 9. вариант 1 (решение владельца): F_h(d0) на всех h; h=1 — без gate_E, по gate_D без лучшего дизайна; победителей нет
        for h in (5, 22):
            (GATES / f'gate_E_h{h}.yaml').write_text(yaml.safe_dump(dict(phase='E', h=h, pobeditel=dict(pobeditel='F_h(d0)', V0=False, poryadok=[])),
                                                                     allow_unicode=True))
        (GATES / 'gate_E_h1.yaml').unlink()
        (GATES / 'gate_D_h1.yaml').write_text(yaml.safe_dump(dict(phase='D', h=1, luchshiy=None), allow_unicode=True))
        sp1 = dict(status='zamorozheno', s2_pravilo='posledovatelno', s2_etalon=dict(est='sm', kappa=0.5, embargo='on'), h={}, F_h_d0=[1, 5, 22],
                   gate_E_sha={h: sha(GATES / f'gate_E_h{h}.yaml') for h in (5, 22)}, gate_D_sha={1: sha(GATES / 'gate_D_h1.yaml')})
        OUTDIR = tmp / 'v1' / 'final'
        READ.clear(); WROTE.clear()
        o1 = run(putspec(sp1, tmp / 'spec_v1.yaml'))
        assert not list(OUTDIR.glob('final_r2_*')) and len(list(OUTDIR.glob('*_s2emb_h*_test.parquet'))) == 6
        for h in HS:
            assert np.allclose(o1[('E', h)].F_sm, out[('E', h)].F_sm, rtol=0, atol=0), f'h={h}: эталон варианта 1 = основной прогон'
        stop({**sp1, 'gate_D_sha': {}}, 'нет gate_D_h1')
        (GATES / 'gate_D_h1.yaml').write_text(yaml.safe_dump(dict(phase='D', h=1, luchshiy='C_h1_x'), allow_unicode=True))
        stop({**sp1, 'gate_D_sha': {1: sha(GATES / 'gate_D_h1.yaml')}}, 'нужны ворота E')
        OUTDIR = LOCKED / 'R2' / 'final'
    finally:
        REGEN, LOCKED, OUTDIR, GATES, JOURNAL = saved
        READ.clear(); WROTE.clear()
    print('selftest r2_sborka_test: OK (s² последовательно t−1: без эмбарго = R1 / F_sm ВЫР-1, с эмбарго = независимая сверка, у эталона (б) d0, '
          'F_h(d0) и победителя одним правилом; цель — только val шагов 7–10, тестовая — никогда; zamorozka — цель не читается, 7–10 = 6; '
          'ворота E и SHA спецификации до чтения; правила single / ridge_noga; ключи > 6 — отказ; пути; повтор — те же байты; '
          'E1 (+E8): h=5 (hyb₁ + 4·HAR₂₋₅)/5 и h=22 (hyb₅ + 17·HAR₆₋₂₂)/22, s² компонент t−1 с эмбарго, λ·g к ноге (g — файл G × halt, '
          'halt сверен r2_gate_abstain.halt_g), вес ноги w; λ, k, w — только константы):', tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default=str(SPEC))
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--proverka-G', type=int, choices=[5, 22], help='val 2017–2021: E8 сборки = ряды G_E8 кэша побайтно (test_locked не читается)')
    ap.add_argument('--ryad', default='E1', help='--proverka-G: E1 / E1b / E2 (r2_gate_abstain.E8_RYADY_H)')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.proverka_G:
        R = proverka_G(a.proverka_G, a.ryad)
        sys.exit(0 if R.ok.all() else 3)
    try:
        run(a.spec)
    except Stop as e:
        print('СТОП:', e, flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
