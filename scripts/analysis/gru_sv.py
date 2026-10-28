#!/usr/bin/env python3
"""GRU на скользящей валидации (9 шагов, тестовые годы 2017-2025) на features_pit_v3 — решение владельца 25.10, 21:40 (цель (4)).
Шаг k: тестовый год ty = 2016 + k, валидационный vy = ty - 1, обучение — все строки с годом < vy (расширяющееся окно, как tune_Bv2.SD).
21 признак — 04_train_neural.py (как test2019_v3_nn.py); модель на тикер x шаг x h (как в коде диссертации — по тикеру);
цель ln RV (h=1 rv_target_h1, h=5/22 rv_target_mean_h{h}), стандартизация X и y по обучению; последовательность строки t — дни [t-L+1, t].
Подбор на каждом (тикер, шаг, h): optuna TPE seed 42, 30 испытаний, MedianPruner(5, 10) по val-MSE эпохи (как OptunaOptimizer диссертации);
пространство — объявленное владельцем (dev_log_gru.md № 1): hidden {32,64,128}, layers {1,2}, dropout {0.1,0.2,0.3}, lr {3e-4,1e-3}, seq {22,44};
batch 32, до 100 эпох, ранняя остановка по val-MSE (ln, станд.) с patience 10 и лучшими весами (табл. 2.7), без scheduler/clip (GRU).
Испытание оценивается по val-QLIKE (уровень RV); прогноз шага — лучшего испытания (val и test от одной модели).
Выход: data/regen/GRU_sv_h{h}_val.parquet (шаги 1-9; формат одиночной модели для gibrid_io/gibrid_v17 --boost GRU_sv),
data/regen/test_locked/GRU_sv/GRU_sv_h{h}_test.parquet (не открывать), data/regen/GRU_sv_trials.csv, части — data/regen/GRU_sv_parts/.
python3 scripts/analysis/gru_sv.py [--workers 6] [--h 1 5 22] [--steps 1..9] [--tickers ...] [--assemble]"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[k] = '1'
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
FEAT, REG = MAIN / 'regen/features_pit_v3.parquet', MAIN / 'regen'
PARTS, TL = REG / 'GRU_sv_parts', REG / 'test_locked/GRU_sv'
PARTS_OOS = REG / 'GRU_svoos_parts'
FC = ['rv', 'rv_d', 'rv_w', 'rv_m', 'bv', 'jump', 'rsv_pos', 'rsv_neg', 'rskew', 'rkurt', 'rv_morning', 'rv_midday', 'rv_evening',
      'rv_IMOEX', 'rv_RVI', 'vix', 'sp500', 'brent', 'gold', 'key_rate_cbr', 'fed_rate']
HS, NTR, BATCH, EPOCHS, PAT = (1, 5, 22), 30, 32, 100, 10
SPACE = dict(hidden=[32, 64, 128], layers=[1, 2], dropout=[0.1, 0.2, 0.3], lr=[3e-4, 1e-3], seq=[22, 44])
tcol = lambda h: 'rv_target_h1' if h == 1 else f'rv_target_mean_h{h}'
_DF = {}


def qlike(y, p):  # = gibrid_schemes.qlike
    p = np.clip(p, 1e-10, None)
    return float(np.mean(y / p - np.log(y / p) - 1))


def data():
    if 'd' not in _DF:
        d = pd.read_parquet(FEAT, columns=['date', 'ticker'] + [tcol(h) for h in HS] + FC)
        d['date'] = pd.to_datetime(d.date)
        _DF['d'] = {tk: g.sort_values('date').reset_index(drop=True) for tk, g in d.groupby('ticker')}
    return _DF['d']


def task(arg):
    tk, step, h = arg[:3]
    fixed = arg[3] if len(arg) > 3 else None  # справка --oos-params: параметры лучшего испытания шага t-1, без подбора
    fp = (PARTS_OOS if fixed else PARTS) / f'{tk}_s{step}_h{h}.parquet'
    if fp.exists():
        return str(fp), 0.0, None
    import torch, torch.nn as nn, optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.set_num_threads(1)
    t0 = time.time(); t = tcol(h); ty = 2016 + step; vy = ty - 1
    g = data()[tk]; g = g[g.date.dt.year <= ty].reset_index(drop=True)
    F = np.nan_to_num(g[FC].ffill().fillna(0).values.astype(np.float64), nan=0, posinf=0, neginf=0)  # ffill — только прошлое
    y = g[t].values; yr = g.date.dt.year.values
    ok = np.isfinite(y) & (y > 0)
    idx = np.where(ok)[0]
    tr_all, va, te = idx[yr[idx] < vy], idx[yr[idx] == vy], idx[yr[idx] == ty]
    mu, sd = F[: tr_all.max() + 1].mean(0), F[: tr_all.max() + 1].std(0) + 1e-8
    Z = ((F - mu) / sd).astype(np.float32)
    ly = np.log(np.where(ok, y, 1.0) + 1e-10); ym, ys = ly[tr_all].mean(), ly[tr_all].std()
    assert va.min() >= max(SPACE['seq']) - 1 and len(va) > 0
    cache = {}

    def tensors(L):
        if L not in cache:
            tr = tr_all[tr_all >= L - 1]  # строки обучения с полной историей L дней
            Xs = lambda ii: torch.tensor(np.stack([Z[i - L + 1: i + 1] for i in ii])) if len(ii) else None
            Yv = lambda ii: torch.tensor(((ly[ii] - ym) / ys).astype(np.float32))
            cache[L] = (Xs(tr), Yv(tr), Xs(va), Yv(va), Xs(te), len(tr))
        return cache[L]

    preds = {}

    def objective(trial):
        p = {k: trial.suggest_categorical(k, v) for k, v in SPACE.items()}
        Xtr, Ytr, Xva, Yva, Xte, ntr = tensors(p['seq'])
        torch.manual_seed(42)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.rnn = nn.GRU(len(FC), p['hidden'], p['layers'], dropout=p['dropout'] if p['layers'] > 1 else 0.0, batch_first=True)
                s.fc = nn.Linear(p['hidden'], 1)

            def forward(s, x): return s.fc(s.rnn(x)[0][:, -1]).squeeze(-1)
        net, mse = Net(), nn.MSELoss(); opt = torch.optim.Adam(net.parameters(), lr=p['lr'])
        gen = torch.Generator().manual_seed(42)
        best, bst, wait, ep = np.inf, None, 0, 0
        for ep in range(EPOCHS):
            net.train()
            for b in torch.randperm(len(Xtr), generator=gen).split(BATCH):
                opt.zero_grad(); loss = mse(net(Xtr[b]), Ytr[b]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): vl = float(mse(net(Xva), Yva))
            if vl < best - 1e-9: best, bst, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= PAT: break
            trial.report(vl, ep)
            if trial.should_prune():
                trial.set_user_attr('epochs', ep + 1); raise optuna.TrialPruned()
        net.load_state_dict(bst); net.eval()
        with torch.no_grad():
            pv = np.clip(np.exp(net(Xva).numpy().astype(np.float64) * ys + ym), 1e-10, None)
            pt = np.clip(np.exp(net(Xte).numpy().astype(np.float64) * ys + ym), 1e-10, None) if Xte is not None else np.array([])
        q = qlike(y[va], pv)
        preds[trial.number] = (pv, pt)
        trial.set_user_attr('epochs', ep + 1); trial.set_user_attr('val_mse_std', best); trial.set_user_attr('n_train', ntr)
        return q

    if fixed:
        q = objective(optuna.trial.FixedTrial(fixed)); pv, pt = preds[0]
        meta = dict(step=step, point=-1, qlike_val_ticker=q, params=json.dumps(fixed), epochs=-1)
        pd.DataFrame({'date': g.date.values[va], 'ticker': tk, 'year': yr[va], **meta, 'part': 'val', 'rv_actual': y[va], 'rv_pred': pv}).to_parquet(fp, index=False)
        return str(fp), time.time() - t0, fixed
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    st.optimize(objective, n_trials=NTR)
    tdf = st.trials_dataframe()
    tdf.insert(0, 'ticker', tk); tdf.insert(1, 'step', step); tdf.insert(2, 'h', h); tdf.insert(3, 'val_year', vy)
    bt = st.best_trial; pv, pt = preds[bt.number]
    meta = dict(step=step, point=bt.number, qlike_val_ticker=bt.value, params=json.dumps(bt.params), epochs=bt.user_attrs['epochs'])
    out = [pd.DataFrame({'date': g.date.values[ii], 'ticker': tk, 'year': yr[ii], **meta, 'part': part, 'rv_actual': y[ii], 'rv_pred': pp})
           for ii, pp, part in ((va, pv, 'val'), (te, pt, 'test'))]
    pd.concat(out, ignore_index=True).to_parquet(fp, index=False)
    tdf.to_csv(fp.with_suffix('.trials.csv'), index=False)
    return str(fp), time.time() - t0, (len(tdf[tdf.state == 'PRUNED']), bt.params)


def assemble(tks, steps):
    TL.mkdir(parents=True, exist_ok=True)
    trials = []
    for h in HS:
        fs = [PARTS / f'{tk}_s{s}_h{h}.parquet' for s in steps for tk in tks]
        if not all(f.exists() for f in fs):
            print('h', h, 'не собран: нет', sum(not f.exists() for f in fs), 'частей'); continue
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        cols = ['date', 'ticker', 'year', 'step', 'point', 'rv_actual', 'rv_pred', 'qlike_val_ticker', 'params', 'epochs']
        d[d.part == 'val'][cols].sort_values(['step', 'date', 'ticker']).to_parquet(REG / f'GRU_sv_h{h}_val.parquet', index=False)
        d[d.part == 'test'][cols].sort_values(['step', 'date', 'ticker']).to_parquet(TL / f'GRU_sv_h{h}_test.parquet', index=False)  # не открывать
        trials += [pd.read_csv(f.with_suffix('.trials.csv')) for f in fs]
        print('h', h, 'собран:', len(d[d.part == 'val']), 'строк val')
    if trials:
        pd.concat(trials, ignore_index=True).to_csv(REG / 'GRU_sv_trials.csv', index=False)


def oos_params(tks, hs, workers):
    """Справка к смещению выбора: на шаге t (2-6) GRU обучается один раз с параметрами лучшего испытания (тикер, t-1, h);
    ранняя остановка — по val шага t, как в основном прогоне. Шаг 1 (нужен только для s² шага 2) берётся из основного прогона."""
    PARTS_OOS.mkdir(parents=True, exist_ok=True)
    prm = lambda tk, s, h: json.loads(pd.read_parquet(PARTS / f'{tk}_s{s}_h{h}.parquet', columns=['params']).params.iloc[0])
    tasks = [(tk, s, h, prm(tk, s - 1, h)) for h in hs for s in range(6, 1, -1) for tk in tks]
    with Pool(workers) as P:
        for i, (fp, sec, info) in enumerate(P.imap_unordered(task, tasks), 1):
            print(f'{time.strftime("%T")} {i}/{len(tasks)} {Path(fp).stem} {sec:.0f} с {info}', flush=True)
    for h in hs:
        cols = ['date', 'ticker', 'year', 'step', 'point', 'rv_actual', 'rv_pred', 'qlike_val_ticker', 'params', 'epochs']
        s1 = pd.read_parquet(REG / f'GRU_sv_h{h}_val.parquet'); s1 = s1[s1.step == 1]
        d = pd.concat([s1] + [pd.read_parquet(PARTS_OOS / f'{tk}_s{s}_h{h}.parquet')[cols] for s in range(2, 7) for tk in tks], ignore_index=True)
        d.sort_values(['step', 'date', 'ticker']).to_parquet(REG / f'GRU_svoos_h{h}_val.parquet', index=False)
        print('h', h, 'GRU_svoos собран:', len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--h', type=int, nargs='*', default=list(HS))
    ap.add_argument('--steps', type=int, nargs='*', default=list(range(1, 10))); ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--oos-params', action='store_true', help='справка: шаги 2-6, параметры лучшего испытания (тикер, шаг t-1, h), без подбора -> GRU_svoos_h{h}_val.parquet')
    a = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    tks = a.tickers or sorted(data())
    if a.assemble:
        assemble(tks, a.steps); return
    if a.oos_params:
        oos_params(tks, a.h, a.workers); return
    # порядок: разработочные шаги 1-6 по h (h=1, 5, 22; внутри — длинные шаги первыми), затем шаги 7-9
    dev_ = [(tk, s, h) for h in a.h for s in sorted([s for s in a.steps if s <= 6], reverse=True) for tk in tks]
    tasks = dev_ + [(tk, s, h) for s in sorted([s for s in a.steps if s > 6], reverse=True) for h in a.h for tk in tks]
    print(time.strftime('%F %T'), 'старт', len(tasks), 'задач x', NTR, 'испытаний', flush=True)
    t0 = time.time()
    with Pool(a.workers) as P:
        for i, (fp, sec, info) in enumerate(P.imap_unordered(task, tasks), 1):
            print(f'{time.strftime("%T")} {i}/{len(tasks)} {Path(fp).stem} {sec:.0f} с, отсечено/лучшее {info}, всего {time.time() - t0:.0f} с', flush=True)
    print(time.strftime('%F %T'), 'конец', flush=True)
    if a.tickers is None and sorted(a.h) == list(HS) and sorted(a.steps) == list(range(1, 10)):
        assemble(tks, a.steps)


if __name__ == '__main__':
    main()
