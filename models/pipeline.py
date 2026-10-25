"""Конвейер дизайна: данные -> признаки -> Dataset -> прогнозы -> HAR-J-семейство -> гибрид -> сводка.
Каждая стадия кэшируется по ключу (входы + параметры + git-версии файлов стадии), пересчитывается только затронутое.
Обучения, HAR-J и гибрид идут в одном пуле процессов (по числу ядер, num_threads=1 на модель); гибрид по h запускается,
как только готовы HAR-J(h) и все обучения h."""
import json, logging, os, time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, Future, wait
import multiprocessing as mp
import pandas as pd
from . import cache as K, config as C, data as D, hybrid as HY, levers as LV, train as TR, v2 as V2

log = logging.getLogger('models')


def import_external(prefix, h, st, T):
    """Внешние прогнозы ({prefix}_h{h}_{val,test}.parquet, формат regen/tune_B) -> кэш прогнозов по шагам (для ворот гибрида)."""
    out = []
    files = {s: f'{prefix}_h{h}_{s}.parquet' for s in ('val', 'test')}
    dig = K.file_digest(files.values())
    for step in st:
        k = K.key('boost_ext', files=dig, step=step)
        if not K.pred_path('val', 'boost', k).exists():
            with T.stage('прогнозы'):
                for s, f in files.items():
                    d = pd.read_parquet(f)
                    K.write_atomic(d[d.step == step].reset_index(drop=True), K.pred_path(s, 'boost', k))
        out.append(k)
    return out


class LazyPool:
    """Пул создаётся только при первом промахе кэша; попадания исполняются сразу (без запуска воркеров)."""

    def __init__(self, workers):
        self.workers, self.P = workers, None

    def go(self, fn, t, hit):
        if hit:
            f = Future()
            f.set_result(fn(t))
            return f
        if self.P is None:
            self.P = pool(self.workers)
        return self.P.submit(fn, t)

    def close(self):
        if self.P is not None:
            self.P.shutdown()


def pool(workers):
    os.environ['OMP_NUM_THREADS'] = '1'  # наследуется воркерами до импорта lightgbm/xgboost
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'))


def run(design, root, workers, confirm=False, smoke=False):
    T, t_start = K.Timer(), time.time()
    st = C.steps(confirm, smoke)
    hs = list(design.get('horizons', C.HS))
    src = D.source(design, root)
    dk = D.stage_data(src, T)
    tr, boost, lv = design['train'], {}, dict(design.get('levers') or {})
    v2 = tr.get('protocol') == 'b_v2'
    pairs = {(h, s) for h in hs for s in st} | ({(h, 1) for h in hs} if v2 else set())
    if lv.get('h1_feature'):
        pairs |= {(1, s) for s in st} | {(1, 1)}
    fk = D.stage_features(dk, sorted(pairs), T)
    fc = K.meta('data', dk)['fc']
    tasks, avg = [], lv.pop('avg_prev_step', False)  # Д8.4 — постобработка, не входит в ключ обучения
    ext = design.get('boost_external')
    for h in hs:
        for m in C.MODELS:
            if ext:
                boost[(m, h)] = import_external(str(C.data_path(ext[m])), h, st, T)
                continue
            boost[(m, h)] = [None] * len(st)
            for i, s in enumerate(st):
                t = dict(dk=dk, fk=fk[(h, s)], model=m, h=h, step=s, params=C.step_params(design, m, h, s), train=tr,
                         seed=int(design['seed']), idx=i)
                if v2:
                    t.update(keep=V2.keep_cols(tr.get('exclude_features'), fc, m, h, s), fk1=fk[(h, 1)])
                if lv:
                    t['levers'] = lv
                if lv.get('h1_feature') and h != 1:
                    t['h1task'] = dict(dk=dk, fk=fk[(1, s)], model=m, h=1, step=s, params=C.step_params(design, m, 1, s), train=tr,
                                       seed=int(design['seed']), keep=V2.keep_cols(tr.get('exclude_features'), fc, m, 1, s), fk1=fk[(1, 1)])
                tasks.append(t)
    tasks.sort(key=lambda t: (t['model'] != 'lgbm', -t['step'], -t['h']))  # длинные первыми: LightGBM раньше XGBoost, поздние шаги раньше (25.10)
    ns, hcfg = len(st), design['hybrid']
    sk = HY.spec_key(dk, ns)
    res, fut, wall = [], {}, {}
    P = LazyPool(workers)
    t0 = time.time()
    fut[P.go(HY.run_spec, dict(dk=dk, nsteps=ns), K.done('har_spec', sk))] = ('har_spec', None)
    for h in hs:
        hkey = HY.har_key(dk, h, ns)
        fut[P.go(HY.run_har, dict(dk=dk, h=h, nsteps=ns), K.pred_path('test', 'har', hkey).exists())] = ('har', h)
    for t in tasks:
        k = TR.task_key(t['fk'], t['model'], t['params'], t['train'], t['seed'])
        fut[P.go(TR.run_task, t, K.pred_path('val', 'boost', k).with_suffix('.json').exists())] = ('boost', t)
    left = {h: sum(1 for t in tasks if t['h'] == h) for h in hs}
    hk, hyb = {}, {}
    pending = set(fut)
    try:
        while pending:
            dn, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in dn:
                kind, info = fut[f]
                r = f.result()
                stg = {'boost': 'прогнозы', 'har': 'HAR-J', 'har_spec': 'HAR-J', 'hybrid': 'гибрид'}[kind]
                wall[stg] = time.time() - t0
                T.add(stg, 0 if r['hit'] else (r.get('sec_train', 0) if kind == 'boost' else r.get('sec', 0)), hit=r['hit'])
                if kind == 'boost':
                    boost[(info['model'], info['h'])][info['idx']] = r['key']
                    left[info['h']] -= 1
                    res.append(r)
                    log.info('boost %s h%s step%s bi=%s q=%.4f %.1fs hit=%s', r['model'], r['h'], r['step'], r['best_iteration'],
                             r['qlike_val'], r.get('sec_total', 0), r['hit'])
                elif kind == 'har':
                    hk[info] = r['key']
                elif kind == 'hybrid':
                    hyb[info] = r['key']
                for h in hs:  # гибрид h — как только готовы HAR-J(h) и все обучения h
                    if h in hk and left[h] == 0 and h not in hyb and ('hybrid', h) not in fut.values():
                        if avg and not ext:
                            for m in C.MODELS:
                                boost[(m, h)] = LV.avg_prev(boost[(m, h)], st)
                        t = dict(h=h, dk=dk, nsteps=ns, har=hk[h], boost={m: boost[(m, h)] for m in C.MODELS}, cfg=hcfg)
                        nf = P.go(HY.run_hybrid, t, K.done('hybrid', HY.hyb_key(h, t['har'], t['boost'], hcfg, dk, ns)))
                        fut[nf] = ('hybrid', h); pending.add(nf)
    finally:
        P.close()
    for r in res:
        if not r['hit']:
            T.add('Dataset', r['sec_dataset'], hit=not r['dataset_built'])
    man = dict(design=design['name'], steps=st, data=dk, features={f'{h}_{s}': k for (h, s), k in fk.items()},
               boost={f'{m}_{h}': v for (m, h), v in boost.items()}, har={str(h): k for h, k in hk.items()}, har_spec=sk,
               hybrid={str(h): k for h, k in hyb.items()}, train=tr, src=src,
               trainings=pd.DataFrame(res).drop(columns=['tag'], errors='ignore').to_dict('records'))
    man['wall'] = {k: round(v, 2) for k, v in wall.items()}
    man['total_sec'] = round(time.time() - t_start, 2)
    man['timer'] = T.table().to_dict('records')
    p = manifest_path(design['name'], confirm, smoke, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(man, default=str, indent=1))
    log.info('manifest %s total %.1fs', p, man['total_sec'])
    return man


def manifest_path(name, confirm=False, smoke=False, root=None):
    tag = 'smoke' if smoke else ('all' if confirm else 'dev')
    if root is not None:  # --data-root (подмена входов, детектор утечек) — отдельный манифест
        tag += '_' + K.key('root', r=os.path.abspath(str(root)))[:8]
    return C.CACHE / 'runs' / f'{name}__{tag}.json'
