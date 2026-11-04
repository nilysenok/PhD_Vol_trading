# -*- coding: utf-8 -*-
"""Generate owner report/handoff from DEV artefacts only."""
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from gl3_r56_data import ROOT,TAB,OUT,disk_guard

REPORT=ROOT/'docs/otchety/2026-11-04_gl3_r5_r6.md'


def table(a):
    def fmt(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'.replace('.',',')
        if isinstance(v,(bool,np.bool_)):return 'да' if v else 'нет'
        return str(v).replace('|','/')
    return '| '+' | '.join(map(str,a.columns))+' |\n|'+ '|'.join(['---']*len(a.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in a.itertuples(index=False,name=None))


def main():
    disk_guard('render report')
    selection=json.loads((TAB/'selection.json').read_text()) if (TAB/'selection.json').exists() else dict(status='diagnostics_only',choices={})
    dev=pd.read_csv(TAB/'dev_candidates.csv') if (TAB/'dev_candidates.csv').exists() else pd.DataFrame()
    stress=pd.read_csv(TAB/'diag_stress.csv')
    events=[json.loads(x) for x in (TAB/'fit_events.jsonl').read_text().splitlines()] if (TAB/'fit_events.jsonl').exists() else []
    ev=pd.DataFrame(events)
    counts=dict(evaluated_rows=len(dev),candidate_rows=int(dev.eligible.sum()) if len(dev) else 0,actual_fit_calls=len(events),
                inner_ES_fits=int(ev.phase.eq('inner_ES').sum()) if len(ev) else 0,
                full_ML_refits=int(ev.phase.eq('refit').sum()) if len(ev) else 0,
                analytic_or_meta_fits=int(ev.phase.eq('fit').sum()) if len(ev) else 0,
                round3_unique_variants=15,round4_unique_variants=134)
    (TAB/'attempt_counts.json').write_text(json.dumps(counts,indent=2))
    old=pd.read_csv(TAB/'rounds_3_6.csv');summary=old[old['round'].isin([3,4])].copy()
    more=[]
    for rd in (5,6):
        for h in (1,5,22):
            x=dev[(dev['round']==rd)&(dev.h==h)&dev.eligible] if len(dev) else pd.DataFrame()
            best=x.loc[x.mean_QLIKE.idxmin()] if len(x) else None
            passed=bool(x.admitted.any()) if len(x) else False
            more.append(dict(round=rd,h=h,tried='К9–К14: экстраполяция, bias, адаптация, RV, bagging' if rd==5 else 'Р1–Р6: архитектура, стек, панель, эксперты, распределение',
                             best_dev=best.mean_QLIKE if best is not None else np.nan,best_recipe=best.candidate if best is not None else 'не запускался',
                             admission='есть допущенный' if passed else ('нет допущенного' if len(x) else 'не запускался'),N_round_total=int(((dev['round']==rd)&dev.eligible).sum()) if len(dev) else 0))
    summary=pd.concat([summary,pd.DataFrame(more)],ignore_index=True);summary.to_csv(TAB/'rounds_3_6.csv',index=False)
    lines=['# Глава3, раунды5–6: экстраполяция, адаптация и архитектура',
           f"Статус: **{selection['status']}**. DEV2017–2021. Контроль2022–2026 закрыт, открытий2→2. Численные итоги ниже меняются, пока статус running/failed_incomplete; номинации не заморожены.",
           '**Раскрытие:** гипотеза раундов4–6 сформулирована по условному GW на контрольном периоде в раунде3. Последующий контроль не является первым независимым подтверждением.',
           'Действующие D22_h1/D12_h5/HAR-J(б)h22 и глава4 не меняются. Отдельная ветка, handoff дирижёру; официальные журналы и номераЖ не редактируются.',
           '## Диагностика до кандидатов',
           'Полная диагностика: [отдельный отчёт](2026-11-04_gl3_r56_diagnostics.md), [режимы и годы](../tables/gl3_r5_r6/diag_regimes.csv), [границы прогнозов](../tables/gl3_r5_r6/diag_prediction_bounds.csv), [выход за обучающие максимумы](../tables/gl3_r5_r6/diag_outside_training.csv).',
           table(stress),
           'Наh1/5 финал сильнее HAR-J(б) занижает уровень в обоих стрессовых срезах: К9/10 запускаются. Наh22 прогнозы тождественны: К9/10 пропущены; начат К11. Разница в среднем RV/p не равнозначна разнице QLIKE: например,h1 в верхнем терцилеRVI финал имеет меньший QLIKE, несмотря на больший средний RV/p.',
           '## Dev-таблица',
           'LOYO: структура/родитель выбирается по остальным4годам. Основная метрика — среднее5годовых QLIKE; все строки оцениваются на прежнейRV и одном пуле17тикеров. Стресс2020=март–июнь; стрессRVI=верхний терциль предыдущих252дат. Допуск: лучше сохранённого финала в≥3/5лет и не хуже в обеих стрессовых строках. Свежий FIN — диагностический эталон переобучения, его гиперпараметры фиксированы по warmup2016, поэтому численно он не обязан повторять старый FIN с прежней процедурой подбора.',
           table(dev[['h','candidate','2017','2018','2019','2020','2021','mean_QLIKE','stress2020','stressRVI','better_years','stress_pass','admitted']]) if len(dev) else 'Ожидается расчёт.',
           '## Номинации и единственное открытие',
           '```json\n'+json.dumps(selection['choices'],ensure_ascii=False,indent=2)+'\n```',
           'До завершённого протокола, его коммита и отдельного «да» владельца контроль не читается. DM/MCS/Холм/GW на2022+ не рассчитывались. При отсутствии допуска на всехh раунд закрывается без открытия; иначе открытие единое на раунды5–6.',
           '## Счёт попыток',table(pd.DataFrame([counts])),
           'Число строк таблицы и фактических обучений различается: одна строка может включать5годовых/квартальных/месячных переобучений, top5FIN,10seed,5LOYO-родителей, раннюю остановку и полное повторное обучение. Сжатия и условные калибровки учитываются отдельно от новых деревьев. Ledger сохраняет все фактические вызовы, включая warmup, отрицательные результаты, прерванные/повторные технические запуски. Кэш прогнозов не начисляет новое обучение. Счётчики раундов3/4 имеют прежнюю единицу «уникальный вариант»; их нельзя механически складывать с fit-вызовами.',
           '## Раунды3–6',table(summary),
           '## Воспроизводимость и ограничения',
           'Расчёт локально на Маке, без агентов и серверных обучений. Новые прогнозы — parquet/float32; деревья отрицательных кандидатов не сохраняются. Перед этапами df -h /, при свободных<20GiB остановка. Действующие источники/модели не изменены. К13 использует136локальных1минCSV2014–2021; файлы2022+ не читались. Минута становится доступна после её окончания; только до18:50,10сдвигов0…9, без ночных доходностей и подстановки цены из будущего.',
           'Повторные технические запуски: исправлено различение единого рецепта и LOYO-карты рецептов и исключение одного года во всей цепочке выбора; после указания владельца NPZ-кэш прогнозов перенесён в parquet/float32 без новых моделей. Первая запись на заполненном диске не удалась; исходные данные сохранены. Кэши pip/Homebrew очищены по поручению владельца; Time Machine требовал пароль, рабочие копии с несохранёнными файлами не удалены.',
           'Новые производные признаки проверены независимым префиксным пересчётом и сдвигом вперёд. Режимы переобучения проверены эмбарго и мутациями даты доступности цели. Существующие семейства v3 наследуют прежний аудит и не объявляются заново сертифицированными. Детали в validation.json; тесты tests/test_gl3_r56.py.',
           'Команды:\n```sh\nPYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r56_run.py\nPYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r56_validate.py\npython3 -m pytest -q tests/test_gl3_r56.py\nPYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r56_report.py\n```']
    REPORT.write_text('\n\n'.join(lines)+'\n')
    handoff=ROOT/'docs/otchety/gl3_r5_r6_dirizher_handoff.md'
    handoff.write_text('# Handoff дирижёру: глава3, раунды5–6\n\n'+f"Статус: {selection['status']}.\n\n"+'Официальный номерЖ не присвоен; журналы/решения не редактировались. Контроль не открывался; счётчик остаётся2. Финал и глава4 неизменны.\n\nГипотеза раундов4–6 сформулирована по условному GW на контрольном периоде раунда3; раскрытие сохранить.\n\nДиагностика допускает К9/10 наh1/5, h22 пропущен по тождеству. Итоги/номинации и счёт попыток — в отчёте2026-11-04_gl3_r5_r6.md, dev_candidates.csv и attempt_counts.json. Резерв6 разрешён только при отсутствии допуска5 на всехh.\n\n'+json.dumps(counts,ensure_ascii=False,indent=2)+'\n\nДо отдельного «да» после коммита протокола новых контрольных чтений не выполнять. При отсутствии допуска закрыть без открытия,2→2; при разрешённом открытии одна запись2→3 на оба раунда.\n')
    manifest=dict(status=selection['status'],platform=platform.platform(),control_opened=False,files={})
    for p in sorted((ROOT/'scripts/analysis').glob('gl3_r56*.py')):
        manifest['files'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (TAB/'code_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Report updated',selection['status'],counts)


if __name__=='__main__':main()
