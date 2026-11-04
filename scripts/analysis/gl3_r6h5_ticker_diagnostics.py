# -*- coding: utf-8 -*-
"""Owner addendum1g: pooled-vs-ticker audit and saved DEV ticker losses."""
import json
import numpy as np
import pandas as pd
from gl3_r6h5_data import Data,TAB,ROOT,disk
from gl3_r56_diagnostics import render


def main():
    disk('ticker diagnostics');d=Data();ref=d.ref.copy();rows=[]
    for tk,g in ref.groupby('ticker'):
        r=g.rv_actual/g.rv_pred;h=g.rv_actual/g.har_corr
        q=float((r-np.log(r)-1).groupby(g.year).mean().mean());b=float((h-np.log(h)-1).groupby(g.year).mean().mean())
        m=g.date.between('2020-03-01','2020-06-30')
        rows.append(dict(ticker=tk,n=len(g),QLIKE_final=float(q),QLIKE_HAR_b=float(b),delta_final_minus_HAR_b=float(q-b),
                         gain_pct=float(100*(1-q/b)),RV_over_final=float(r.mean()),RV_over_HAR_b=float(h.mean()),
                         ratio_final_stress2020=float(r[m].mean()),ratio_HAR_b_stress2020=float(h[m].mean()),hybrid_loses=bool(q>b)))
    out=pd.DataFrame(rows).sort_values('delta_final_minus_HAR_b',ascending=False);out.to_csv(TAB/'ticker_diagnostics.csv',index=False)
    text=['# Дополнение1г: h5, dev2017–2021,17тикеров',
          '## Архитектура по коду',
          'D12_h5 — одна общая панельная модель на все17тикеров для каждой библиотеки/годового шага, затем среднее уровней XGB/LGB. Не17 отдельных моделей. Вся матрица идёт в один fit: [d12_run.py:24](../../scripts/analysis/d12_run.py#L24), [l2_lib.py:523](../../scripts/analysis/l2_lib.py#L523), [l2_lib.py:611](../../scripts/analysis/l2_lib.py#L611). Единый study на модель×шаг, без тикерного разбиения: [tune_L2_optuna.py:47](../../scripts/analysis/tune_L2_optuna.py#L47), [tune_L2_optuna.py:99](../../scripts/analysis/tune_L2_optuna.py#L99). Гиперпараметры общие для тикеров, меняются по библиотеке/шагу.',
          'В D12 нет явного ticker/sector ID: конфигурация CFG5 толькоv9/harf/huber [vyr_etalony_finaly.py:81](../../scripts/analysis/vyr_etalony_finaly.py#L81); идентификаторы не попадают в FC [l2_lib.py:86](../../scripts/analysis/l2_lib.py#L86). Добавление категорий требует cfg.cat [l2_lib.py:408](../../scripts/analysis/l2_lib.py#L408), у CFG5 его нет. Отраслевые индексы в исходных признаках не равны явному идентификатору сектора. Список замороженных признаков также не содержит ticker/sector/cat_ полей.',
          'HAR-J(а)/(б) используют общие pooled коэффициенты: одна LinearRegression на всех строках прошлого окна [vyr_etalony_finaly.py:214](../../scripts/analysis/vyr_etalony_finaly.py#L214), [vyr_etalony_finaly.py:222](../../scripts/analysis/vyr_etalony_finaly.py#L222). (а) — raw без s²-множителя; (б) — те же коэффициенты плюс общий по17тикерам скаляр exp(s²/2). Потикерных intercept/коэффициентов здесь нет.',
          'Для текущего эталона (б) с эмбарго s² общий скаляр подтверждён прямо assert по step: [r2_calib.py:538](../../scripts/analysis/r2_calib.py#L538), [r2_calib.py:542](../../scripts/analysis/r2_calib.py#L542). Калибровка по прошлым остаткам и конец цели с эмбарго: [r2_calib.py:65](../../scripts/analysis/r2_calib.py#L65), [r2_calib.py:213](../../scripts/analysis/r2_calib.py#L213), [r2_calib.py:245](../../scripts/analysis/r2_calib.py#L245). Возможность sm_by_ticker в библиотеке существует, но это отдельная опция, а не действующий baseline.',
          '## По тикерам',
          'QLIKE — равновесное среднее5годовых потерь каждого тикера, как основная dev-метрика; RV/прогноз — среднее по его наблюдениям2017–2021; Δ>0 означает проигрыш гибрида. RV/прогноз>1 означает среднее занижение. Дополнительно приведено отношение в марте–июне2020.',render(out),
          f"Гибрид проигрывает HAR-J(б) на {int(out.hybrid_loses.sum())}/17 тикерах: "+', '.join(out.loc[out.hybrid_loses,'ticker'])+'. Контроль не читался.']
    (ROOT/'docs/otchety/2026-11-04_gl3_r6_h5_TICKERS.md').write_text('\n\n'.join(text)+'\n')
    print(out.to_string(index=False))


if __name__=='__main__':main()
