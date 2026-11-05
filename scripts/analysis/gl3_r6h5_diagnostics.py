# -*- coding: utf-8 -*-
"""Owner item1: before new candidates, h5 saved forecasts only."""
import json
import numpy as np
import pandas as pd
from gl3_r6h5_data import Data,TAB,REG,ROOT,READS,disk
from gl3_r56_diagnostics import render


def main():
    disk('diagnostics first');d=Data();old=pd.read_csv(ROOT/'docs/tables/gl3_r5_r6/dev_candidates.csv');old=old[old.h==5]
    k9=old[old.candidate.str.startswith('K9')].sort_values('mean_QLIKE').iloc[0].candidate
    series={'D12_h5':d.series(d.ref,'rv_pred'),'HAR-J(б)':d.series(d.ref,'har_corr'),
            'К9 лучший ('+k9+')':d.old(k9),'К10 лучший':d.old('K10'),
            'К9 допущенный r5 (c=0.75)':d.old('K9_lgb_shrink0.75'),
            'HAR-X⁺ напрямую h5':d.saved(REG/'D22/HXP_5_val.parquet','rv_pred_sm')}
    rows=[]
    for label,p in series.items():rows.append(dict(model=label,**d.score5(p)))
    frame=pd.DataFrame(rows);frame.to_csv(TAB/'diagnostics.csv',index=False)
    text=['# Раунд6, только h5: диагностика до кандидатов',
          'Только dev2017–2021, контроль не читался. Новые модели для этой диагностики не обучались. Все прогнозы сопоставлены на одном кадре17тикеров, цели полностью заканчиваются≤2021.',
          '## Аудит D12_h5 по коду',
          'В split() обучение и валидация разделяются только годом строки; проверки конца пятедневной цели, purge≥5 и embargo нет: [l2_lib.py:507](../../scripts/analysis/l2_lib.py#L507). step_data вызывает split без дополнительной очистки: [l2_lib.py:523](../../scripts/analysis/l2_lib.py#L523). Это касается и HAR-смещения: [l2_lib.py:539](../../scripts/analysis/l2_lib.py#L539). Поэтому последние обучающие цели могут перекрываться с началом валидационного года.',
          'Optuna: TPE seed42,30испытаний на модель×шаг, выбор по QLIKE того же val-года: [tune_L2_optuna.py:50](../../scripts/analysis/tune_L2_optuna.py#L50), [tune_L2_optuna.py:85](../../scripts/analysis/tune_L2_optuna.py#L85). Ранняя остановка на этом же val-году: XGB mphe при Huber, иначе rmse; LGB huber либо l2, patience30, потолок1000: [l2_lib.py:620](../../scripts/analysis/l2_lib.py#L620), [l2_lib.py:634](../../scripts/analysis/l2_lib.py#L634), [l2_lib.py:651](../../scripts/analysis/l2_lib.py#L651), [l2_lib.py:670](../../scripts/analysis/l2_lib.py#L670), [l2_lib.py:44](../../scripts/analysis/l2_lib.py#L44). Отдельного, ранее лежащего внутреннего ES-окна перед оцениваемым val-годом здесь нет. D12_run повторяет fit на выбранных точках: [d12_run.py:28](../../scripts/analysis/d12_run.py#L28). Источник численных warmup-параметров2016 сохранён ранее; годовые точки прежнего D12 могли подбираться на оцениваемом dev-году. Это ограничение исходного in-sample эталона, а не новый просмотр контроля.',
          '## QLIKE и RV/прогноз',render(frame),
          'HAR-X⁺: прямой сохранённый прогноз HXP_5_val.rv_pred_sm, а не корректирующая надстройка над D12. Эластичная сеть по прошлому внутреннему году и очистка конца цели описаны в [d22_harxp.py:136](../../scripts/analysis/d22_harxp.py#L136); s²-поправка — по прошлой ступени. В таблице показан лучший К9 по среднему dev; отдельно оставлен допущенный К9 c=.75, чтобы не смешивать два понятия «лучший».',
          'Уточнение аудита архивного HAR-X⁺: основная регрессия имеет target-end embargo, однако i_va внутреннего выбора alpha/l1 [d22_harxp.py:150](../../scripts/analysis/d22_harxp.py#L150) и старая add_bias [harx.py:299](../../scripts/analysis/harx.py#L299) не проверяют конец цели на границе года. Эта строка воспроизводит заранее зафиксированный архивный эксперт, а не доказывает его полную point-in-time чистоту. Новые Г1/калибраторы проверяются отдельно; выбор архивного источника после результатов не менялся.',
          'Знак ответа о занижении следует из ratio_stress2020/ratio_stressRVI: >1 — среднее RV/прогноз выше единицы. См. численные строки; снижение этого отношения само по себе не означает улучшения QLIKE.']
    (ROOT/'docs/otchety/2026-11-04_gl3_r6_h5_DIAGNOSTICS.md').write_text('\n\n'.join(text)+'\n')
    (TAB/'diagnostic_reads.json').write_text(json.dumps(READS,ensure_ascii=False,indent=2))
    print(frame.to_string(index=False))


if __name__=='__main__':main()
