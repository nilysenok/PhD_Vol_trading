#!/usr/bin/env python3
"""S0, дополнение к ch4_dev_sigma.py (поручение дирижёра после PR #234): проверка сдвига σ h=5 (гибрид, HAR-J (б)) против h-дневной цели.

Цель строки d: y_d = среднее RV дней d+1…d+5 по строкам тикера (features_pit_v3.rv через r2_loyo.rv_daily — фильтр ≤ 2021-12-31 при
чтении + assert; окна, выходящие за 2021-12-30, — NaN и не участвуют). Спирмен ln σ²_d с ln y_{d+k}, k = −2…+1, медиана по тикерам,
решение r2_loyo.sdvig_ok (K = {0}, симметричное ε = 0,01) — тот же код, что ch4_dev_sigma.shift_row, с y вместо дневной RV.
Модельный ряд решает уровень; профиль приращений — справочно. Периоды 2017–2021 и 2018–2021.
Контроли те же: σ_d := σ_{d−1} (сдвиг −1) и σ_d := σ_{d+1} (утечка +1) — обязаны блокироваться.

Читаются только data/regen/ch4_dev/sigma_{hyb,harj_b}_h5_dev.parquet (SHA сверяется с ch4_dev/SHA256.txt) и RV ≤ 2021.
Строки дописываются в data/regen/ch4_dev/proverka_sdvig.csv (столбец cel; прежние строки — cel = «RV_{d+k} дневная»,
повторный запуск заменяет свои строки), строка SHA proverka_sdvig.csv в SHA256.txt обновляется.

python3 scripts/analysis/ch4_dev_sdvig_h5.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch4_dev_sigma as S  # noqa: E402  read_dev, check_dev, shift_ts, shift_row, sha

OUT = S.OUT
H = 5
CEL_DAY = 'RV_{d+k} дневная'
CEL_H = 'y_{d+k} = среднее RV дней d+k+1…d+k+5 (h=5)'
SERIES = {'hyb_h5': 'sigma_hyb_h5_dev.parquet', 'harj_b_h5': 'sigma_harj_b_h5_dev.parquet'}


def sha_ref():
    ref = {}
    for ln in (OUT / 'SHA256.txt').read_text().splitlines():
        if ln and not ln.startswith('#'):
            h, p = ln.split('  ')[:2]
            ref[p] = h
    return ref


def target_h(rv):
    """y_d = среднее RV строк d+1…d+H тикера; хвост без полного окна — NaN."""
    r = rv.sort_values(['ticker', 'date']).reset_index(drop=True)
    r['rv'] = r.groupby('ticker')['rv'].transform(lambda s: s.rolling(H, min_periods=H).mean().shift(-H))
    return r


def main():
    import r2_loyo as O
    import r2_features as RF
    assert abs(float(O.CFG['sdvig']['eps']) - S.EPS) < 1e-12
    ref = sha_ref()
    series = {}
    for lab, fn in SERIES.items():
        p = OUT / fn
        want = ref[f'data/regen/ch4_dev/{fn}']
        if S.sha(p) != want:
            raise SystemExit(f'СТОП: SHA {fn} ≠ ch4_dev/SHA256.txt')
        d = S.read_dev(p, ['date', 'ticker', 'sigma_pred_h5'])
        series[lab] = (d, 'ряд разработки')
    rv = O.rv_daily()
    rv['date'] = pd.to_datetime(rv['date'])
    S.check_dev(rv, 'rv_daily')
    y = target_h(rv)
    for lab, (d, _) in list(series.items()):
        series[f'kontrol:{lab}_sdvig-1'] = (S.shift_ts(d, 'sigma_pred_h5', -1), 'контроль: σ_d := σ_{d−1} (сдвиг −1)')
        series[f'kontrol:{lab}_utechka+1'] = (S.shift_ts(d, 'sigma_pred_h5', +1), 'контроль: σ_d := σ_{d+1} (утечка +1)')
    rows = []
    for name, (d, rol) in series.items():
        for start in ('2017-01-01', '2018-01-01'):
            r = S.shift_row(O, RF, y, name, d, 'sigma_pred_h5', 'model', rol, start)
            r['cel'] = CEL_H
            r['primechanie'] = ('окно цели y_{d+k} при k < 0 содержит известные дню d RV (y_{d−1} — RV_d, y_{d−2} — RV_{d−1}, RV_d): '
                                'argmax < 0 ожидаем и у ряда без сдвига; различает сдвиг перенос профиля относительно контролей')
            rows.append(r)
            print(f'  {name} {start[:4]}: ' + ' '.join(f'k{k:+d} {r[f"corr_k{k:+d}"]:.4f}' for k in (-2, -1, 0, 1))
                  + f' argmax {r["argmax_uroven"]} ok {r["ok"]}', flush=True)
    new = pd.DataFrame(rows)
    f = OUT / 'proverka_sdvig.csv'
    old = pd.read_csv(f)
    if 'cel' not in old:
        old['cel'] = CEL_DAY
    old = old[old['cel'] != CEL_H]
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)
    lines = (OUT / 'SHA256.txt').read_text().splitlines()
    lines = [ln for ln in lines if 'ch4_dev_sdvig_h5.py' not in ln]
    lines = [f'{S.sha(f)}  data/regen/ch4_dev/proverka_sdvig.csv  (проверка; + цель h=5, ch4_dev_sdvig_h5.py)'
             if 'proverka_sdvig.csv' in ln else ln for ln in lines]
    lines.append(f'{S.sha(HERE / "ch4_dev_sdvig_h5.py")}  scripts/analysis/ch4_dev_sdvig_h5.py  (код: сдвиг h=5 против цели h=5)')
    (OUT / 'SHA256.txt').write_text('\n'.join(lines) + '\n')
    work = ~new['ryad'].str.startswith('kontrol:')
    ok = bool(new.loc[work, 'ok'].all())
    blok = bool((~new.loc[~work, 'ok']).all())
    print(f'ряды h=5 против цели h=5: {"OK" if ok else "СДВИГ"}; контроли блокируются: {blok}')
    return 0 if ok and blok else 2


if __name__ == '__main__':
    os.environ.setdefault('NUMBA_CACHE_DIR', str(S.MAIN / 'data' / 'fastbt' / 'numba_cache'))
    sys.exit(main())
