#!/usr/bin/env python3
"""Справочные LSTM и GRU на скользящей валидации (СВ), честный протокол (ночной приоритет 2, 26.10; в FINAL не входят).
Шаг k = 1..10: год оценки vy = 2015 + k, следующий год ty = 2016 + k (шаг 10: vy = 2025, ty = янв.-фев. 2026).
Модель — по тикеру (как в коде диссертации), 21 признак 04_train_neural.py из features_pit_v3.parquet, цель ln RV:
h=1 — rv_target_h1 = RV(t+1), h=5/22 — rv_target_mean_h{h} = среднее RV за [t+1, t+h].
Отличия от gru_sv.py (справочный GRU 25.10), из-за которых его протокол переписан:
  1) эмбарго по концу цели: в обучение шага входят только строки i с i + h < первая строка года оценки (по строкам тикера);
  2) на шаге t НИЧЕГО не берётся из года оценки vy_t: гиперпараметры, число эпох и траектория lr — лучшего испытания исследования
     шага t-1 (подбор на vy_{t-1}); модель переобучается на всех строках до vy_t (с эмбарго) ровно на это число эпох, без ранней
     остановки по vy_t (в gru_sv.py --oos-params ранняя остановка шла по году оценки);
  3) поправка Б0.1 одна, объявлена заранее: F = exp(s²/2), s² — дисперсия OOS-остатков ln RV - ln прогноза года оценки шага t-1
     (объединение тикеров, строки с целью внутри года — эмбарго); без выбора среди ord/rob/sm/smrob по метрике;
  4) шаг 1 (нужен только для s² шага 2): фиксированные параметры диссертации (04_train_neural.py: 64 x 2, dropout 0.2, lr 1e-3,
     seq 22, 50 эпох), без подбора на годе оценки;
  5) три seed переобучения (42/43/44), прогноз — среднее ln-прогнозов (ансамбль); подбор — seed 42, как в gru_sv.py;
  6) добавлен шаг 10.
Подбор (как gru_sv.py, не урезан): optuna TPE seed 42, 30 испытаний на (модель, тикер, шаг, h), MedianPruner(5, 10), критерий —
val-QLIKE; пространство hidden {32,64,128}, layers {1,2}, dropout {0.1,0.2,0.3}, lr {3e-4,1e-3}, seq {22,44}; batch 32, до 100 эпох,
ранняя остановка по val-MSE (patience 10, лучшие веса). GRU — Adam без scheduler/clip (как gru_sv.py); LSTM — как
src/models/lstm_model.py: clip_grad_norm 1.0 и ReduceLROnPlateau(0.5, 5) по val-MSE (при переобучении — записанная траектория lr).
Исследования шагов 1..9 (шаг 9 даёт параметры шагу 10). Выход (общий data/):
  data/regen/ref_nn_garch/nn_parts/ — исследования (*.study.json, *.trials.csv) и переобучения (*_seed{s}.parquet);
  data/regen/ref_nn_garch/REF_{GRU,LSTM}_h{h}_val.parquet — годы оценки шагов 1-6 (разработочный набор + шаг 1 для s²);
  data/regen/test_locked/ref_nn_garch/REF_{m}_h{h}_valconfirm.parquet (шаги 7-10) и REF_{m}_h{h}_test.parquet (ty шагов 1-10) — без метрик.
python3 scripts/analysis/ref_nn_sv.py --phase tune --steps 1 2 3 4 5 [--models GRU LSTM] [--workers 5]
python3 scripts/analysis/ref_nn_sv.py --phase refit --steps 1 2 3 4 5 6
python3 scripts/analysis/ref_nn_sv.py --phase assemble"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[_k] = '1'
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
REG = MAIN / 'regen'
FEAT = REG / 'features_pit_v3.parquet'
OUT = REG / 'ref_nn_garch'
PARTS = OUT / 'nn_parts'
TL = REG / 'test_locked/ref_nn_garch'
FC = ['rv', 'rv_d', 'rv_w', 'rv_m', 'bv', 'jump', 'rsv_pos', 'rsv_neg', 'rskew', 'rkurt', 'rv_morning', 'rv_midday', 'rv_evening',
      'rv_IMOEX', 'rv_RVI', 'vix', 'sp500', 'brent', 'gold', 'key_rate_cbr', 'fed_rate']
HS, MODELS, SEEDS, DEV_LAST, NSTEP = (1, 5, 22), ('GRU', 'LSTM'), (42, 43, 44), 6, 10
CFG = dict(ntr=30, epochs=100, pat=10, batch=32)  # детектор утечек уменьшает бюджет через CFG
SPACE = dict(hidden=[32, 64, 128], layers=[1, 2], dropout=[0.1, 0.2, 0.3], lr=[3e-4, 1e-3], seq=[22, 44])
DEFAULT = dict(params=dict(hidden=64, layers=2, dropout=0.2, lr=1e-3, seq=22), best_ep=50, lrs=None)  # шаг 1
tcol = lambda h: 'rv_target_h1' if h == 1 else f'rv_target_mean_h{h}'
vy_of, ty_of = (lambda k: 2015 + k), (lambda k: 2016 + k)


def qlike(y, p):  # = gibrid_schemes.qlike
    p = np.clip(p, 1e-10, None)
    return float(np.mean(y / p - np.log(y / p) - 1))


def load_frames(path=FEAT):
    d = pd.read_parquet(path, columns=['date', 'ticker'] + [tcol(h) for h in HS] + FC)
    d['date'] = pd.to_datetime(d.date)
    return {tk: g.sort_values('date').reset_index(drop=True) for tk, g in d.groupby('ticker')}


_FR = {}


def frames():
    if 'f' not in _FR:
        _FR['f'] = load_frames()
    return _FR['f']


class Prep:
    """Строки тикера для шага: обучение (эмбарго по концу цели), год оценки, следующий год; стандартизация по обучению."""

    def __init__(s, g, h, step):
        vy, ty = vy_of(step), ty_of(step)
        g = g[g.date.dt.year <= ty].reset_index(drop=True)
        F = np.nan_to_num(g[FC].ffill().fillna(0).values.astype(np.float64), nan=0, posinf=0, neginf=0)  # ffill — только прошлое
        y = g[tcol(h)].values.astype(np.float64)
        yr = g.date.dt.year.values
        ok = np.isfinite(y) & (y > 0)
        i = np.arange(len(g))
        i0, i1 = int(np.searchsorted(yr, vy)), int(np.searchsorted(yr, ty))  # первые строки vy и ty
        s.tr = i[ok & (i + h <= i0 - 1)]                      # цель [i+1, i+h] кончается до начала vy
        s.va = i[ok & (yr == vy)]
        s.emb = (s.va + h <= i1 - 1)                          # цель строки vy кончается внутри vy (для s² и калибровки)
        s.te = i[yr == ty]
        last = int(s.tr.max())
        mu, sd = F[:last + 1].mean(0), F[:last + 1].std(0) + 1e-8
        s.Z = ((F - mu) / sd).astype(np.float32)
        s.ly = np.log(np.where(ok, y, 1.0) + 1e-10)
        s.ym, s.ys = float(s.ly[s.tr].mean()), float(s.ly[s.tr].std())
        s.g, s.y, s.yr, s.h, s.step = g, y, yr, h, step
        s.cache = {}

    def tensors(s, L):
        import torch
        if L not in s.cache:
            tr = s.tr[s.tr >= L - 1]
            Xs = lambda ii: torch.tensor(np.stack([s.Z[j - L + 1: j + 1] for j in ii])) if len(ii) else None
            Yv = lambda ii: torch.tensor(((s.ly[ii] - s.ym) / s.ys).astype(np.float32))
            assert len(s.va) == 0 or s.va.min() >= L - 1
            s.cache[L] = (Xs(tr), Yv(tr), Xs(s.va), Yv(s.va), Xs(s.te), len(tr))
        return s.cache[L]


def make_net(model, p, nf):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            R = nn.GRU if model == 'GRU' else nn.LSTM
            s.rnn = R(nf, p['hidden'], p['layers'], dropout=p['dropout'] if p['layers'] > 1 else 0.0, batch_first=True)
            s.fc = nn.Linear(p['hidden'], 1)

        def forward(s, x):
            return s.fc(s.rnn(x)[0][:, -1]).squeeze(-1)
    return Net()


def train(model, p, Xtr, Ytr, seed, Xva=None, Yva=None, epochs=None, lrs=None, trial=None):
    """epochs=None — подбор: ранняя остановка по val-MSE, лучшие веса; LSTM — ReduceLROnPlateau по val.
    epochs=E — переобучение: ровно E эпох, lr по эпохам из lrs (None — постоянный), без val."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    net, mse = make_net(model, p, Xtr.shape[2]), nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=p['lr'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5) if model == 'LSTM' and epochs is None else None
    gen = torch.Generator().manual_seed(seed)
    hist, best, bst, best_ep, wait = [], np.inf, None, 0, 0
    for ep in range(CFG['epochs'] if epochs is None else int(epochs)):
        if lrs is not None:
            for q in opt.param_groups:
                q['lr'] = lrs[ep]
        hist.append(float(opt.param_groups[0]['lr']))
        net.train()
        for b in torch.randperm(len(Xtr), generator=gen).split(CFG['batch']):
            opt.zero_grad(); loss = mse(net(Xtr[b]), Ytr[b]); loss.backward()
            if model == 'LSTM':
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
        if epochs is not None:
            continue
        net.eval()
        with torch.no_grad():
            vl = float(mse(net(Xva), Yva))
        if sch is not None:
            sch.step(vl)
        if vl < best - 1e-9:
            best, bst, best_ep, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1, 0
        else:
            wait += 1
            if wait >= CFG['pat']:
                break
        if trial is not None:
            trial.report(vl, ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    if epochs is None:
        net.load_state_dict(bst)
    net.eval()
    return net, dict(best_ep=best_ep, val_mse=best, lrs=hist[:best_ep])


def logpred(net, X, P):
    import torch
    if X is None:
        return np.array([])
    with torch.no_grad():
        return net(X).numpy().astype(np.float64) * P.ys + P.ym


def study(model, g, h, step):
    """Исследование шага step: обучение — до vy (эмбарго), критерий — val-QLIKE года vy. Его лучшее испытание идёт шагу step+1."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    P = Prep(g, h, step)

    def objective(trial):
        p = {k: trial.suggest_categorical(k, v) for k, v in SPACE.items()}
        Xtr, Ytr, Xva, Yva, _, ntr = P.tensors(p['seq'])
        net, info = train(model, p, Xtr, Ytr, 42, Xva, Yva, trial=trial)
        q = qlike(P.y[P.va], np.clip(np.exp(logpred(net, Xva, P)), 1e-10, None))
        for k, v in (('best_ep', info['best_ep']), ('val_mse_std', info['val_mse']), ('lrs', json.dumps(info['lrs'])), ('n_train', ntr)):
            trial.set_user_attr(k, v)
        return q
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    st.optimize(objective, n_trials=CFG['ntr'])
    bt = st.best_trial
    spec = dict(params=bt.params, best_ep=int(bt.user_attrs['best_ep']), lrs=json.loads(bt.user_attrs['lrs']), value=float(bt.value),
                trial=bt.number, n_pruned=sum(t.state.name == 'PRUNED' for t in st.trials), val_year=vy_of(step), n_train=len(P.tr))
    return spec, st.trials_dataframe()


def refit(model, g, h, step, spec, seed, tk):
    """Шаг step: параметры/эпохи/lr из spec (исследование шага step-1), обучение на строках до vy (эмбарго), прогноз vy и ty."""
    P = Prep(g, h, step)
    p = spec['params']
    Xtr, Ytr, Xva, _, Xte, _ = P.tensors(p['seq'])
    net, _ = train(model, p, Xtr, Ytr, seed, epochs=spec['best_ep'], lrs=spec['lrs'] if model == 'LSTM' else None)
    out = []
    for part, ii, lp, emb in (('val', P.va, logpred(net, Xva, P), P.emb), ('test', P.te, logpred(net, Xte, P), np.zeros(len(P.te), bool))):
        out.append(pd.DataFrame({'date': P.g.date.values[ii], 'ticker': tk, 'year': P.yr[ii], 'step': step, 'part': part, 'seed': seed,
                                 'rv_actual': P.y[ii], 'lp': lp, 'emb': emb}))
    return pd.concat(out, ignore_index=True)


def s2_by_step(E):
    """s² шага k: дисперсия ln RV - lp по строкам года оценки шага k с целью внутри года (объединение тикеров)."""
    v = E[(E.part == 'val') & E.emb & (E.rv_actual > 0)]
    return {k: float(np.var(np.log(g.rv_actual.values + 1e-10) - g.lp.values)) for k, g in v.groupby('step')}


def bias_correct(R):
    """R: строки переобучений (step, part, seed, date, ticker, rv_actual, lp, emb). Ансамбль — среднее lp по seed.
    val шага k: F = exp(s²_{k-1}/2) (шаг 1: F = 1, в метрики не входит); test шага k: F_own = exp(s²_k/2), F_prev = exp(s²_{k-1}/2),
    rv_pred теста = raw * F_own (как gibrid_bias: на тесте шага — оценка шага)."""
    E = R.groupby(['step', 'part', 'date', 'ticker'], as_index=False).agg(year=('year', 'first'), rv_actual=('rv_actual', 'first'),
                                                                         lp=('lp', 'mean'), emb=('emb', 'first'), n_seed=('seed', 'nunique'))
    s2 = s2_by_step(E)
    E['rv_pred_raw'] = np.clip(np.exp(E.lp), 1e-10, None)
    E['F_prev'] = np.exp(E.step.map(lambda k: s2.get(k - 1, 0.0)) / 2)
    E['F_own'] = np.exp(E.step.map(lambda k: s2.get(k, np.nan)) / 2)
    E['rv_pred'] = E.rv_pred_raw * np.where(E.part == 'val', E.F_prev, E.F_own)
    return E.sort_values(['step', 'part', 'date', 'ticker']).reset_index(drop=True), s2


def chain(model, frs, h, steps, seeds=SEEDS):
    """Вся цепочка в памяти (для детектора утечек): исследования шагов 1..max-1, переобучения шагов steps, поправка."""
    R = []
    for tk, g in frs.items():
        spec = {1: DEFAULT}
        for k in range(1, max(steps)):
            spec[k + 1] = study(model, g, h, k)[0]
        R += [refit(model, g, h, k, spec[k], sd, tk) for k in steps for sd in seeds]
    return bias_correct(pd.concat(R, ignore_index=True))[0]


# ---------- прогон по частям (общий data/) ----------
sp = lambda m, tk, s, h: PARTS / f'{m}_{tk}_s{s}_h{h}.study.json'
rp = lambda m, tk, s, h, sd: PARTS / f'{m}_{tk}_s{s}_h{h}_seed{sd}.parquet'


def task_tune(arg):
    m, tk, s, h = arg
    fp = sp(m, tk, s, h)
    if fp.exists():
        return fp.name, 0.0, 'есть'
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    spec, tdf = study(m, frames()[tk], h, s)
    for c, v in (('model', m), ('ticker', tk), ('step', s), ('h', h), ('val_year', vy_of(s))):
        tdf.insert(0, c, v)
    tdf.to_csv(fp.with_suffix('').with_suffix('.trials.csv'), index=False)
    fp.write_text(json.dumps(spec))
    return fp.name, time.time() - t0, (spec['n_pruned'], spec['params'], spec['best_ep'])


def task_refit(arg):
    m, tk, s, h, sd = arg
    fp = rp(m, tk, s, h, sd)
    if fp.exists():
        return fp.name, 0.0, 'есть'
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    spec = DEFAULT if s == 1 else json.loads(sp(m, tk, s - 1, h).read_text())
    refit(m, frames()[tk], h, s, spec, sd, tk).to_parquet(fp, index=False)
    return fp.name, time.time() - t0, spec['best_ep']


def assemble(models, tks):
    TL.mkdir(parents=True, exist_ok=True)
    s2rows = []
    for m in models:
        for h in HS:
            fs = [rp(m, tk, s, h, sd) for s in range(1, NSTEP + 1) for tk in tks for sd in SEEDS]
            have = [f for f in fs if f.exists()]
            R = pd.concat([pd.read_parquet(f) for f in have], ignore_index=True)
            steps = sorted(R.step.unique())
            full = [s for s in steps if R[R.step == s].groupby('ticker').seed.nunique().reindex(tks).fillna(0).min() == len(SEEDS)]
            R = R[R.step.isin(full)]
            E, s2 = bias_correct(R)
            seedc = R.pivot_table(index=['step', 'part', 'date', 'ticker'], columns='seed', values='lp').add_prefix('lp_s').reset_index()
            E = E.merge(seedc, on=['step', 'part', 'date', 'ticker'], how='left')
            dev = E[(E.part == 'val') & (E.step <= DEV_LAST)]
            dev.drop(columns='part').to_parquet(OUT / f'REF_{m}_h{h}_val.parquet', index=False)
            E[(E.part == 'val') & (E.step > DEV_LAST)].drop(columns='part').to_parquet(TL / f'REF_{m}_h{h}_valconfirm.parquet', index=False)
            E[E.part == 'test'].drop(columns='part').to_parquet(TL / f'REF_{m}_h{h}_test.parquet', index=False)
            s2rows += [dict(model=m, h=h, step=k, s2=v) for k, v in s2.items() if k <= DEV_LAST - 1]  # s² шагов 7+ не выводится
            print(m, 'h', h, 'шаги', full, 'dev строк', len(dev), flush=True)
    pd.DataFrame(s2rows).to_csv(OUT / 'REF_nn_s2_dev.csv', index=False)


def run_pool(fn, tasks, workers, tag):
    t0 = time.time()
    print(time.strftime('%F %T'), tag, 'старт', len(tasks), 'задач', flush=True)
    with Pool(workers) as P:
        for i, (name, sec, info) in enumerate(P.imap_unordered(fn, tasks), 1):
            print(f'{time.strftime("%T")} {i}/{len(tasks)} {name} {sec:.0f} с {info} всего {time.time() - t0:.0f} с', flush=True)
    print(time.strftime('%F %T'), tag, 'конец', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['tune', 'refit', 'assemble'], required=True)
    ap.add_argument('--models', nargs='*', default=list(MODELS))
    ap.add_argument('--h', type=int, nargs='*', default=list(HS))
    ap.add_argument('--steps', type=int, nargs='*', default=None)
    ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=5)
    a = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    tks = a.tickers or sorted(frames())
    if a.phase == 'assemble':
        assemble(a.models, tks); return
    if a.phase == 'tune':
        steps = sorted(a.steps or range(1, NSTEP), reverse=True)  # длинные шаги первыми
        tasks = [(m, tk, s, h) for s in steps for h in a.h for m in a.models for tk in tks]
        run_pool(task_tune, tasks, a.workers, 'подбор')
    else:
        steps = sorted(a.steps or range(1, NSTEP + 1), reverse=True)
        tasks = [(m, tk, s, h, sd) for s in steps for h in a.h for m in a.models for tk in tks for sd in SEEDS
                 if s == 1 or sp(m, tk, s - 1, h).exists()]
        run_pool(task_refit, tasks, a.workers, 'переобучение')


if __name__ == '__main__':
    main()
