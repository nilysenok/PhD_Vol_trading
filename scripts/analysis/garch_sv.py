#!/usr/bin/env python3
"""Справочный GJR-GARCH(1,1) и GARCH(1,1) на скользящей валидации (решение владельца 26.10, 00:55: справочно, в FINAL_DESIGN не входит).
Спецификация как в test2019_v3.stage_garch (и moex_volatility/scripts/models/train_garch.py): по тикеру, дневные лог-доходности close x100, mean=Constant,
dist=normal, p=1,o=0|1,q=1; прогноз h-шаговой дисперсии: среднее ожидаемых σ² за [t+1, t+h], делится на 1e4; без масштабирования (как в коде).
Шаг k = 1..10: год оценки vy = 2015+k, тестовый год ty = 2016+k; параметры оцениваются ОДИН раз по расширяющемуся окну (годы < vy), без подбора
на годе оценки; на vy и ty модель фильтрует с теми же параметрами. Один процесс, одно ядро.
Выход: data/regen/GARCH_sv{gjr,garch}_h{h}_val.parquet (год оценки vy, шаги 1-6 = разработочный набор + шаг 1 для s²);
data/regen/test_locked/GARCH_sv/: {model}_h{h}_test.parquet (ty, шаги 1-10) и {model}_h{h}_valconfirm.parquet (vy шагов 7-10) - не открывать, метрик нет.
python3 scripts/analysis/garch_sv.py"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[k] = '1'
import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from arch import arch_model

REG = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
FEAT, TL = REG / 'features_pit_v3.parquet', REG / 'test_locked/GARCH_sv'
HS, STEPS, DEV = (1, 5, 22), range(1, 11), 6
MODELS = {'GARCH_svgjr': 1, 'GARCH_svgarch': 0}  # имя -> o
tcol = lambda h: 'rv_target_h1' if h == 1 else f'rv_target_mean_h{h}'


def main():
    TL.mkdir(parents=True, exist_ok=True)
    a = pd.read_parquet(FEAT, columns=['date', 'ticker', 'close'] + [tcol(h) for h in HS])
    a['date'] = pd.to_datetime(a.date); a['year'] = a.date.dt.year
    out = {(m, h, p): [] for m in MODELS for h in HS for p in ('val', 'conf', 'test')}
    prm = []; t0 = time.time()
    for tk, g in a.groupby('ticker'):
        g = g.sort_values('date').reset_index(drop=True); r = 100 * np.log(g.close).diff()
        ok = r.notna().values; gg, rr = g[ok].reset_index(drop=True), r[ok].values
        for step in STEPS:
            vy, ty = 2015 + step, 2016 + step
            ntr = int((gg.year < vy).sum())  # окно оценки: годы < vy (расширяющееся)
            for name, o in MODELS.items():
                res = arch_model(rr[:ntr], mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='normal').fit(disp='off', show_warning=False)
                pa = res.params; om, al, be, ga = pa['omega'], pa['alpha[1]'], pa['beta[1]'], pa.get('gamma[1]', 0.0)
                fx = arch_model(rr, mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='normal').fix(pa)
                e = rr - pa['mu']; s2 = fx.conditional_volatility ** 2
                nxt = om + (al + ga * (e < 0)) * e ** 2 + be * s2  # σ²_{t+1} по информации до t
                phi = al + ga / 2 + be; lr = om / (1 - phi) if phi < 1 else np.mean(rr[:ntr] ** 2)
                prm.append(dict(ticker=tk, step=step, model=name, n_train=ntr, omega=om, alpha=al, gamma=ga, beta=be, mu=pa['mu'], persistence=phi,
                                converged=int(res.convergence_flag == 0), loglik=res.loglikelihood))
                for h in HS:
                    f = (lr + np.outer(nxt - lr, phi ** np.arange(h))).mean(axis=1) / 1e4; t = tcol(h)
                    for part, yr in (('val', vy), ('test', ty)):
                        d = gg.assign(fc=f)[gg.year == yr]
                        if not (part == 'test' and step == 10): d = d.dropna(subset=[t])  # 2026: цели нет, метрик не считается
                        fr = pd.DataFrame({'date': d.date.values, 'ticker': tk, 'year': yr, 'step': step, 'point': -1,
                                           'rv_actual': d[t].values, 'rv_pred': np.clip(d.fc.values, 1e-10, None)})
                        out[(name, h, 'test' if part == 'test' else ('val' if step <= DEV else 'conf'))].append(fr)
        print(f'{time.time() - t0:.0f} с {tk}', flush=True)
    for (m, h, p), L in out.items():
        d = pd.concat(L, ignore_index=True).sort_values(['step', 'date', 'ticker']).reset_index(drop=True)
        fp = {'val': REG / f'{m}_h{h}_val.parquet', 'conf': TL / f'{m}_h{h}_valconfirm.parquet', 'test': TL / f'{m}_h{h}_test.parquet'}[p]
        d.to_parquet(fp, index=False); print(fp.name, len(d), flush=True)
    pd.DataFrame(prm).to_csv(REG / 'GARCH_sv_params.csv', index=False)


if __name__ == '__main__':
    main()
