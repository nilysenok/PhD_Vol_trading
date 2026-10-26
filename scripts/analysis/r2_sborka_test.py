#!/usr/bin/env python3
"""R2, агент О: тестовая сборка дизайнов-победителей второго раунда (протокол docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md,
раздел «Тестовая часть: кто читает test_locked»). ЗАГОТОВКА: до заморозки не запускается.

Правило доступа:
- test_locked читает только агент О и только этим скриптом; запуск — только после заморозки: файл R2/O/ZAMOROZKA.json (итог ворот,
  пишет r2_loyo после 29.10, 09:00) с временем, победителями по h и SHA-256 спецификации сборки; спецификация после заморозки не меняется
  (SHA сверяется);
- читаются только файлы компонент победителей, перечисленные в спецификации; путь — внутри data/regen/test_locked, не в R2/final;
- из тестовых файлов загружаются только колонки прогноза date, ticker, year, step, rv_pred (белый список); целевая колонка не загружается,
  assert на её отсутствие — строки с меткой R2-guard; матрица признаков не открывается; оценок качества скрипт не считает;
- параметры (множители Б0.1, веса, β) берутся из спецификации: они оценены на val шагов <= 6 тем же правилом, что на валидации;
  тест шага k >= 7 — значения шага 6 (заморозка; ↺ к первому раунду, где s² шага k бралась по val-годам <= k);
- выход: test_locked/R2/final/final_r2_h{h}_test.parquet (date, ticker, year, step, rv_pred, design) и SHA256.txt;
- журнал: прочитанные и записанные файлы с SHA-256 — docs/otchety/dev_log_r2_O.md.
Проверка проверяющего — grep по шаблону из протокола: совпадения допустимы только в строках с меткой R2-guard.

python3 scripts/analysis/r2_sborka_test.py --spec models/designs/r2/r2_O_sborka.yaml

Спецификация (yaml, пишется при заморозке):
  h:
    5:
      design: "<имя победителя, как в ZAMOROZKA.json>"
      pravilo: single | mean | median | weights | ridge_noga
      komponenty:                       # сырые тестовые прогнозы шагов 1-10 (без s²)
        - {name: luchshiy, file: test_locked/R2/B2/<ряд>_h5_s42_test.parquet, factors: {1: F1, ..., 6: F6}}
        - {name: noga, file: test_locked/R2/E/<ряд>_h5_sdet_test.parquet, factors: {...}}
      weights: {1: {luchshiy: w, noga: w}, ..., 6: {...}}     # pravilo weights (w >= 0, сумма 1)
      beta: {1: b1, ..., 6: b6}                              # pravilo ridge_noga: ln p = ln noga + β (ln luchshiy − ln noga)
      factor_itog: {1: ..., 6: ...}                          # необязательно: множитель к итогу
"""
import argparse, hashlib, json, subprocess, sys
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
ZAM = REGEN / 'R2' / 'O' / 'ZAMOROZKA.json'
JOURNAL = REPO / 'docs' / 'otchety' / 'dev_log_r2_O.md'
KEEP = ('date', 'ticker', 'year', 'step', 'rv_pred')  # белый список колонок тестовых файлов
KEY = ['date', 'ticker', 'step']
LAST = 6  # последний шаг разработки (val 2021): параметры шагов >= 7 — его
READ, WROTE = [], []


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def par(d, k):
    """Параметр шага k: шаги 1-6 — свой (оценён по val шагов <= 6), k >= 7 — шага 6 (заморозка)."""
    d = {int(a): b for a, b in d.items()}
    return d[min(int(k), LAST)]


def zamorozka(spec_path):
    """До любого чтения test_locked: заморозка есть, время в прошлом, SHA спецификации совпадает. -> победители {h: дизайн}."""
    if not ZAM.exists():
        sys.exit(f'СТОП: нет {ZAM} — сборка только после заморозки (итог ворот R2)')
    z = json.loads(ZAM.read_text())
    assert z.get('time') and z.get('winners') is not None and z.get('spec_sha'), 'ZAMOROZKA.json: нужны time, winners, spec_sha'
    if z['time'] > now():
        sys.exit('СТОП: время заморозки в будущем')
    if sha(spec_path) != z['spec_sha']:
        sys.exit('СТОП: спецификация сборки изменена после заморозки (SHA не совпал)')
    return {int(h): d for h, d in z['winners'].items()}


def allowed(rel):
    p = (REGEN / rel).resolve()
    assert str(p).startswith(str(LOCKED.resolve()) + '/'), f'{rel}: не в test_locked'
    assert not str(p).startswith(str(OUTDIR.resolve())), f'{rel}: выход сборки не читается'
    assert p.exists(), f'{rel}: нет файла'
    return p


def read_pred(rel):
    """Тестовый прогноз компоненты: только колонки белого списка."""
    p = allowed(rel)
    use = [c for c in KEEP if c in pq.read_schema(p).names]
    assert use == list(KEEP), f'{p.name}: нет колонок {set(KEEP) - set(use)}'
    d = pd.read_parquet(p, columns=use)
    assert 'rv_actual' not in d.columns  # R2-guard: целевая колонка не загружена
    READ.append((str(p.relative_to(MAIN)), sha(p), len(d)))
    assert d.duplicated(KEY).sum() == 0 and d.rv_pred.gt(0).all() and set(d.step.unique()) <= set(range(1, 11)), p.name
    return d


def combine(c, h):
    """Правило дизайна по шагам: каждая компонента × свой множитель Б0.1 шага, затем объединение, затем множитель итога."""
    K = c['komponenty']
    parts = []
    for comp in K:
        d = read_pred(comp['file'])
        f = d.step.map(lambda k: par(comp['factors'], k)) if comp.get('factors') else 1.0
        parts.append(d.assign(rv_pred=d.rv_pred * f)[KEY + ['year', 'rv_pred']].rename(columns={'rv_pred': comp['name']}))
    m = parts[0]
    for p in parts[1:]:
        m = m.merge(p.drop(columns='year'), on=KEY, how='inner', validate='1:1')
    n0 = len(parts[0])
    assert all(len(p) == n0 for p in parts) and len(m) == n0, f'h={h}: строки компонент не совпадают'
    names = [k['name'] for k in K]
    X = m[names].values
    rule = c['pravilo']
    if rule == 'single':
        assert len(names) == 1
        pred = X[:, 0]
    elif rule == 'mean':
        pred = X.mean(axis=1)
    elif rule == 'median':
        pred = np.median(X, axis=1)
    elif rule == 'weights':
        W = np.vstack([[par(c['weights'], k)[nm] for nm in names] for k in m.step.values])
        assert np.all(W >= 0) and np.allclose(W.sum(axis=1), 1), f'h={h}: веса'
        pred = (W * X).sum(axis=1)
    elif rule == 'ridge_noga':
        assert names[1] == 'noga' or names[0] == 'noga', 'ridge_noga: нужна компонента noga'
        ln_n, ln_b = (np.log(m['noga'].values), np.log(m[[x for x in names if x != 'noga'][0]].values))
        b = m.step.map(lambda k: par(c['beta'], k)).values
        pred = np.exp(ln_n + b * (ln_b - ln_n))
    else:
        raise ValueError(f'неизвестное правило {rule}')
    if c.get('factor_itog'):
        pred = pred * m.step.map(lambda k: par(c['factor_itog'], k)).values
    out = m[['date', 'ticker', 'year', 'step']].assign(rv_pred=pred, design=c['design'])
    assert np.isfinite(out.rv_pred).all() and (out.rv_pred > 0).all()
    return out.sort_values(['step', 'date', 'ticker']).reset_index(drop=True)


def journal(spec_path):
    L = [f'\n## Сборка теста R2 ({now()}; r2_sborka_test.py, SHA кода {sha(__file__)[:16]}…, спецификация {spec_path} SHA {sha(spec_path)[:16]}…)',
         '', 'Прочитано (только колонки прогноза):', '', '| файл | SHA-256 | строк |', '|---|---|---|']
    L += [f'| {f} | {s} | {n} |' for f, s, n in READ]
    L += ['', 'Записано:', '', '| файл | SHA-256 | строк |', '|---|---|---|']
    L += [f'| {f} | {s} | {n} |' for f, s, n in WROTE]
    with open(JOURNAL, 'a') as f:
        f.write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default=str(REPO / 'models/designs/r2/r2_O_sborka.yaml'))
    a = ap.parse_args()
    spec_path = Path(a.spec)
    win = zamorozka(spec_path)  # до любого чтения test_locked
    spec = yaml.safe_load(spec_path.read_text())
    hs = {int(h): c for h, c in spec['h'].items()}
    assert set(hs) == set(win), f'спецификация {sorted(hs)} != победители {sorted(win)}'
    for h, c in hs.items():
        assert c['design'] == win[h], f'h={h}: дизайн спецификации не совпадает с заморозкой'
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for h, c in sorted(hs.items()):
        o = combine(c, h)
        f = OUTDIR / f'final_r2_h{h}_test.parquet'
        o.to_parquet(f, index=False)
        WROTE.append((str(f.relative_to(MAIN)), sha(f), len(o)))
        print(f'h={h}: {c["design"]} -> {f.name} ({len(o)} строк, шаги {sorted(o.step.unique())}); оценок качества нет', flush=True)
    (OUTDIR / 'SHA256.txt').write_text(''.join(f'{s}  {Path(p).name}\n' for p, s, _ in WROTE))
    journal(spec_path)


if __name__ == '__main__':
    main()
