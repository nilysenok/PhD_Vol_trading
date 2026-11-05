# -*- coding: utf-8 -*-
"""Audit, report and close h5 DEV without any control opening."""
import json,hashlib,platform,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from gl3_r6h5_data import Data,TAB,OUT,ROOT,READS,disk,load_events
from gl3_r6h5_engine import purged,gap_boundary
from gl3_r56_diagnostics import render


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def finish():
    disk('independent audit and close h5')
    d=Data();s=json.loads((TAB/'selection.json').read_text());assert s['status']=='dev_complete_no_control_opening'
    dev=pd.read_csv(TAB/'dev_candidates.csv');manifest=[]
    # Independent prediction/metric/admission check using pandas year grouping.
    for row in dev.to_dict('records'):
        path=OUT/f"{row['candidate']}_dev.parquet";p=pd.read_parquet(path)
        assert p.prediction.dtype==np.dtype('float32') and p.date.max()<pd.Timestamp('2022-01-01')
        z=d.ref[['date','ticker','year','rv_actual']].merge(p,on=['date','ticker'],validate='1:1')
        assert len(z)==len(d.ref) and np.isfinite(z.prediction).all() and (z.prediction>0).all()
        ratio=z.rv_actual/z.prediction;loss=ratio-np.log(ratio)-1
        years=loss.groupby(z.year).mean();mean=float(years.mean())
        for y,q in years.items():np.testing.assert_allclose(q,row[str(y)],rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(mean,row['mean_QLIKE'],rtol=1e-11,atol=1e-12)
        idx=pd.MultiIndex.from_frame(d.a[['date','ticker']]).get_indexer(pd.MultiIndex.from_frame(z[['date','ticker']]))
        stress=[z.date.between('2020-03-01','2020-06-30').values,d.a.loc[idx,'_stress'].values]
        for mask,qcol,rcol in zip(stress,['stress2020','stressRVI'],['ratio_stress2020','ratio_stressRVI']):
            np.testing.assert_allclose(loss[mask].mean(),row[qcol],rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(ratio[mask].mean(),row[rcol],rtol=1e-11,atol=1e-12)
        n=sum(years[y]<d.base[str(y)]-1e-12 for y in range(2017,2022))
        sp=all(row[c]<=d.base[c]+1e-12 for c in ('stress2020','stressRVI'))
        admitted=bool(row['eligible'] and n>=3 and sp and mean<d.base['mean_QLIKE']-1e-12)
        assert admitted==row['admitted'] and n==row['better_years']
        manifest.append(dict(candidate=row['candidate'],path=str(path.relative_to(ROOT)),sha256=sha(path),rows=len(p),dtype='float32',audit_pass=True))
    pd.DataFrame(manifest).to_csv(TAB/'prediction_manifest.csv',index=False)
    events=load_events()
    for e in events:assert e['h']==5 and pd.Timestamp(e['max_target_end'])<pd.Timestamp(e['cutoff'])<pd.Timestamp('2022-01-01')
    purge=[]
    for year in range(2016,2022):
        cut=pd.Timestamp(f'{year}-01-01');tr,va,full=purged(d.a,cut,year-1)
        purge.append(dict(year=year,n_inner_train=int(tr.sum()),n_validation=int(va.sum()),n_refit=int(full.sum()),
                          inner_train_last_end=str(d.a.loc[tr,'_end5'].max()),inner_embargo_edge=str(gap_boundary(d.a,pd.Timestamp(f'{year-1}-01-01'))),
                          validation_last_end=str(d.a.loc[va,'_end5'].max()),outer_embargo_edge=str(gap_boundary(d.a,cut)),refit_last_end=str(d.a.loc[full,'_end5'].max())))
    pd.DataFrame(purge).to_csv(TAB/'purge_audit.csv',index=False)
    # Causal transforms are recomputed on independent prefixes. Forward-shift negatives must differ.
    b=d.a;mutations=[]
    expected=b.groupby('ticker').rv.transform(lambda v:v.rolling(252,min_periods=252).mean())
    np.testing.assert_allclose(b._scale,expected,equal_nan=True)
    for c in ('_scale','_ln_rv_w','_ln_close_RVI','_ln_rq','_state','_stress'):
        shifted=b[c].groupby(b.ticker).shift(-1)
        caught=not np.allclose(b[c].astype(float),shifted.astype(float),equal_nan=True)
        assert caught;mutations.append(dict(feature=c,forward_day_caught=True))
    rvi=b.groupby('date').close_RVI.first().sort_index()
    for cut in pd.to_datetime(['2017-06-01','2019-06-01','2020-06-01','2021-06-01']):
        pref=b[b.date<=cut]
        scale=pref.groupby('ticker').rv.transform(lambda v:v.rolling(252,min_periods=252).mean())
        np.testing.assert_allclose(scale,b.loc[pref.index,'_scale'],equal_nan=True)
        rv=rvi.loc[:cut];state=rv>rv.expanding(min_periods=22).median().shift(1)
        np.testing.assert_array_equal(pref.date.map(state).values,pref._state.values)
    trial=[e for e in events if e['phase']=='trial_ES' and e['detail'].get('trial') is not None]
    coverage={(e['component'],e['detail']['variant'],e['cutoff'][:4],e['detail']['trial']) for e in trial if e['detail'].get('numerical_contract')=='D12_clean_v3'}
    assert len(coverage)==720,'Both final G1 studies require2variants×2libraries×6years×30trials'
    unique_trials={json.dumps([e['component'],e['detail'].get('numerical_contract','prototype_v1'),e['detail'].get('variant'),e['detail']['params'],e['seed']],sort_keys=True) for e in trial}
    recipes=set();aliases=set()
    def collect(r):
        recipes.add(json.dumps(r,sort_keys=True))
        if r['core']=='G5':
            parent=r['parent']
            if parent['core'] in ('G2','G3') and not parent.get('cat_inputs',False):aliases.add(json.dumps(parent,sort_keys=True))
            else:
                for seed in range(42,52):recipes.add(json.dumps({**parent,'seed':seed},sort_keys=True))
    for recs in s['recipes'].values():
        for r in recs.values():collect(r)
    counts=dict(named_rows=len(dev),eligible_rows=int(dev.eligible.sum()),hyperparameter_weight_rows=int((~dev.eligible).sum()),
                unique_candidate_recipes_including_seeds=len(recipes),deterministic_bagging_alias_parents=len(aliases),
                all_Optuna_trial_fit_calls=len(trial),unique_Optuna_parameter_points_all_contracts=len(unique_trials),
                final_contract_Optuna_trial_fits=sum(e['detail'].get('numerical_contract')=='D12_clean_v3' for e in trial),
                all_fit_calls=len(events),fit_calls_by_phase=pd.Series([e['phase'] for e in events]).value_counts().to_dict(),
                technical_prototypes_included=True)
    failures=json.loads((TAB/'technical_failures.json').read_text()) if (TAB/'technical_failures.json').exists() else []
    counts['logged_completed_parameter_estimations']=len(events)
    counts['unlogged_completed_fits_after_prediction_failure']=sum(r['completed_fit'] for r in failures)
    counts['failed_or_interrupted_before_completion']=sum(not r['completed_fit'] for r in failures)
    counts['all_estimation_attempts_including_disclosed_failures']=len(events)+len(failures)
    counts['all_Optuna_trial_attempts_including_interrupted']=len(trial)+sum(r['optuna_trial'] for r in failures)
    counts['unique_parameter_points_excludes_unlogged_interrupted_trial']=True
    (TAB/'attempt_counts.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2))
    ledger=pd.DataFrame(events);ledger['detail']=ledger.detail.map(lambda z:json.dumps(z,sort_keys=True));ledger.to_csv(TAB/'fit_ledger.csv.gz',index=False)
    pd.DataFrame([dict(recipe=json.loads(r)) for r in sorted(recipes)]).to_json(TAB/'unique_recipes.json',orient='records',indent=2)
    points=pd.DataFrame([json.loads(r) for r in sorted(unique_trials)],columns=['component','contract','variant','params','seed']);points['params']=points.params.map(lambda z:json.dumps(z,sort_keys=True));points.to_csv(TAB/'unique_trial_points.csv.gz',index=False)
    raw_trials=TAB/'optuna_trials.jsonl'
    if raw_trials.exists():
        points=[json.loads(x) for x in raw_trials.read_text().splitlines()]
        assert len(points)==len(trial)
        for pt,event in zip(points,trial):
            assert pt['year']==pd.Timestamp(event['cutoff']).year and pt['trial']==event['detail']['trial'] and pt['variant']==event['detail']['variant']
            pt['contract']=event['detail'].get('numerical_contract','prototype_v1')
        frame=pd.DataFrame(points);frame['params']=frame.params.map(lambda z:json.dumps(z,sort_keys=True));frame.to_csv(TAB/'optuna_trials.csv.gz',index=False)
    validation=dict(control_read=False,all_fit_horizons=[5],fits_checked=len(events),predictions_audited=len(manifest),all_maturity_valid=True,
                    feature_mutations=mutations,prefix_checks=True,models_serialized=0,
                    ticker_id_check='static known key, stable category codes; row-key shift detected, same-ticker temporal shift is an identity, not future information')
    (TAB/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    passed=dev[dev.admitted]
    assert len(passed)==0,'If a winner appears, freeze a control proposal before marking final closure'
    closure=dict(status='closed_without_winner',control_read=False,control_opened=False,opening_count_before=2,opening_count_after=2,
                 winner=None,reason='No candidate satisfies3years + bothstress + meanbetter',dev_table_sha256=sha(TAB/'dev_candidates.csv'),
                 benchmark_QLIKE=d.base['mean_QLIKE'],hypothesis_origin='control GW in round3',old_final_unchanged=True,
                 versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','optuna','lightgbm','xgboost')})
    (TAB/'closure.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2))
    diag=pd.read_csv(TAB/'diagnostics.csv');tick=pd.read_csv(TAB/'ticker_diagnostics.csv')
    np.testing.assert_allclose(tick.QLIKE_final.mean(),d.base['mean_QLIKE'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(tick.QLIKE_HAR_b.mean(),d.score5(d.series(d.ref,'har_corr'))['mean_QLIKE'],rtol=1e-11,atol=1e-12)
    best=dev[dev.eligible].sort_values('mean_QLIKE').iloc[0]
    lines=['# Глава3, раунд6: только h5, закрыт без победителя',
           '**DEV2017–2021 завершён. Победителя по правилу владельца нет. Контроль2022–2026 не читался, открытий2→2. Действующий финал D12_h5, h1/h22 и глава4 неизменны.**',
           f"Лучший средний результат: **{best.candidate}, QLIKE {best.mean_QLIKE:.9f}** против {d.base['mean_QLIKE']:.9f} у финала (выигрыш{100*(1-best.mean_QLIKE/d.base['mean_QLIKE']):.3f}%). Он выигрывает{best.better_years}/5лет, но март–июнь2020: {best.stress2020:.9f} против {d.base['stress2020']:.9f} (хуже{100*(best.stress2020/d.base['stress2020']-1):.3f}%). Допуск не пройден; пороги не ослаблялись.",
           'Гипотеза раундов4–6 сформулирована по условному GW на прежнем контрольном периоде раунда3. Сохранённые финал/эксперты уже выбирались на dev; текущий LOYO условен на этом фиксированном наборе.',
           '## Диагностика1а–в',
           '[Полный аудит D12, ссылки на файл/строку и диагностика](2026-11-04_gl3_r6_h5_DIAGNOSTICS.md). Исходный D12: годовой split без purge/embargo, native ES30 на том же val-году, по которому ранжируется Optuna30. Старый эталон не пересчитывался задним числом.',render(diag),
           'Лучший К9 уменьшает стрессовое RV/прогноз с1.313 до1.281, но остаётся хуже HAR-J(б)1.254; К10 усиливает занижение. Прямой HAR-X⁺ имеет mean0.209604 и меньшие стрессовые потери, однако не является отдельным кандидатом вне закрытого списка; он используется только как разрешённый эксперт/вход.',
           'Ограничение архивных входов: у HAR-X⁺ основной train-mask очищен, но i_va выбора alpha/l1 и старая s² add_bias не имеют end-horizon проверки. Финал/эксперты фиксированы до этого раунда и уже выбирались на dev; они не объявляются новым строго OOS/PIT набором. all_maturity_valid в audit означает все НОВЫЕ fit текущего раунда, не ретроспективную сертификацию старых источников. HAR-CJ архив взят из _emb_d0, там очищены и fit, и s² (vyr_spravochnye.py:119,133,175).',
           '## Дополнение1г: панель и потикерная диагностика',
           '[Аудит общей модели и коэффициентов](2026-11-04_gl3_r6_h5_TICKERS.md). D12 — pooled17, общие гиперпараметры на библиотеку×год, без ticker/sector ID. HAR-J(а)/(б) — общие коэффициенты; (а) без смещения, у (б) общий скалярs², не потикерный. В Г7в добавлен настоящий категориальный ticker, без индивидуального Optuna.',render(tick),
           'QLIKE по тикеру — среднее5годовых потерь; среднее17строк повторяет общий dev-QLIKE финала/HAR-J(б). RV/прогноз — среднее по наблюдениям тикера. Проигрыш гибрида: LSRG,SBER,MGNT,NVTK.',
           '## Dev-таблица Г1–Г7',
           'Допуск = ≥3строго выигранных года + обе стрессовые строки не хуже + средний QLIKE<0.20996539211548085. Г7b grid-точки отображаются, но eligible=False: допуск проверяется для LOYO-выбораw, чтобы не подменить его лучшей in-sample hyper-точкой. 2020stress=01.03–30.06; RVIstress=верхний терциль предыдущих252дат.',render(dev),
           'У G2_logistic март–июнь практически совпадает с финалом: отличие порядка1e−9 вызвано сохранением прогнозов float32. Его всё равно исключает условие3/5лет. У остальных близких кандидатов стрессовое ухудшение существенно больше погрешности хранения; оно не заменялось нулём.',
           'Г5 над детерминированным ridge/смесью фиксированных входов даёт seed-алиасы; независимые бустинги не изображаются десятью разными fit. Если выбран stochastic-родитель, код делает10seed и фиксирует все fit. Г7в переобучает бустинговые компоненты с категорией, общие hyperparams и ES выбираются только на прошлом.',
           '## Очищенная валидация Г1',render(pd.DataFrame(purge)),
           'Для текущего внешнего года G1 использует предыдущий год внутренней валидации. Удаляются пересекающиеся5-дневные цели, дополнительно5наблюдаемых торговых дат зазора; то же для HAR-offset и конца полного refit. Гистограммные бины строятся на текущем очищенном прошлом, поэтому это не изолированная причинная оценка эффекта одного purge.',
           '## Счёт вариантов и технические ограничения',
           '```json\n'+json.dumps(counts,ensure_ascii=False,indent=2)+'\n```',
           'N не скрывает технические прототипы: отдельно technical_failures.json раскрывает3вызова, не попавших вpost-fit ledger (одинпрерванныйtrial, одинуспешныйRidge-fitсошибкойпоследующегоpredict, одинотказтипапараметраLGBдообучения). Суммарные числа приведены в JSON выше: post-fit записи и3 отдельно раскрытых вызова не смешиваются. Параметры прерванного trial не логировались; число полностью логированных уникальных точек не объявляется исчерпывающим для незавершённых конфигураций. сначала уточнялись log-floor/исключения признаков D12, затем исправлено ранжирование trial по исходномуRV вместо RV+1e−10 log-метки. Прототипы не участвуют в выборе; все их фактические fit сохранены в fit_ledger.csv.gz. Финальные Optuna-точки numerical_contract=D12_clean_v3. Именованные строки, уникальные hyperparameter points и fit-вызовы — разные единицы; дирижёру передаются все счётчики, официальные журналы не правились.',
           'После теста масштабной инвариантности исправлена нормировка корневого jump-регрессора в HAR-offset Г4 (sqrt(scale), как в бустинговой части). Старый вариант сохранён только как технический снимок technical_pre_norm_fix.csv и не участвует в выборе. Уточнён контракт кэша: исправление QLIKE trial-score сначала могло не инвалидировать сохранённую строку Г1. Окончательный Study-cache включает h5-clean-contract-v4, а аудит требует полное покрытие720 финальных trial (2варианта×2библиотеки×6лет×30). Все предыдущие fit остаются в счёте. Исправлен технический отказ ridge на последних строках2021 без завершённой пятедневной цели: предсказываются только строки с доступными входами; все строки основной dev-сетки сохранены. При добавлении cat исправлено приведение frozen целочисленных параметров66.0→66; параметры по тикерам не подбирались.',
           '## Проверки и поставка',
           'Метрики/допуск всех21строк пересчитаны из сохранённых parquet/float32; все fit толькоh5 и с концом цели доfit. Проверены purge/embargo, нормы252, режимыRVI, сырая смесь дляG6, потикерноеQLIKE-сжатие и future-label мутации. TickerID — статический известный ключ: сдвиг по строкам панели ловится проверкойключа, временной сдвиг внутри того же тикера тождественен и не является утечкой будущего. Обученные модели на диск не сохранялись.',
           'Контрольного протокола открытия не предлагается: допущенных нет. Дополнительное решение владельца по другим горизонтам остаётся вне этого раунда. Handoff: gl3_r6_h5_dirizher_handoff.md. Ветка/draftPR в kanon-v22-pereschet; зависимость от инфраструктуры предыдущегоPR#931 указана в описании.',
           'Воспроизведение (толькоDEV):\n```sh\nPYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6h5_diagnostics.py\nPYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6h5_ticker_diagnostics.py\nPYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6h5_run.py\npython3 -m pytest -q tests/test_gl3_r6h5.py\nPYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6h5_finish.py\n```']
    (ROOT/'docs/otchety/2026-11-04_gl3_r6_h5.md').write_text('\n\n'.join(lines)+'\n')
    (ROOT/'docs/otchety/gl3_r6_h5_dirizher_handoff.md').write_text('# Handoff дирижёру: раунд6,h5\n\nЗакрыт без победителя. Контроль не читался/не открывался,2→2. Финал D12_h5,h1/h22,глава4 неизменны. НомерЖ не присвоен, официальные журналы не изменены.\n\n'+f'Лучший mean: {best.candidate} {best.mean_QLIKE:.9f},4/5лет, стресс{best.stress2020:.9f}>{d.base["stress2020"]:.9f}; по сильному правилу владельца не допущен. Открытие не предлагается.\n\n'+json.dumps(counts,ensure_ascii=False,indent=2)+'\n\nВсе отрицательные/технические fit включены вledger; отдельно переданы unique_trial_points иunique_recipes. Для общего счёта главы3 не складывать автоматически разные единицы(годовойfit,параметрическаяточка,полныйрецепт). Предыдущий счёт не переписывался.\n\nИсходная гипотеза по контрольномуGWраунда3 раскрыта. ИсходныйD12 не имелpurge/embargoвнутреннейвалидации; panel/общиеHAR и s² подтверждены по коду. Потикернаятаблица: LSRG,SBER,MGNT,NVTK проигрываютHAR-J(б).\n')
    (TAB/'code_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scripts/analysis').glob('gl3_r6h5*.py'))},indent=2))
    print(json.dumps({'closure':closure,'counts':counts},ensure_ascii=False,indent=2))


if __name__=='__main__':finish()
