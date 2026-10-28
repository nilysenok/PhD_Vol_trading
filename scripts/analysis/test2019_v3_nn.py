#!/usr/bin/env python3
"""LSTM / GRU для теста 2019 на features_pit_v3 (табл. 2.7): hidden 64, 2 слоя, dropout 0,2, последовательность 22 дня, batch 32, Adam 1e-3,
до 100 эпох, ранняя остановка по val-MSE (patience 10, лучшие веса); LSTM — ReduceLROnPlateau (patience 5, factor 0,5) и clip_grad_norm 1,0, GRU — без.
21 признак — состав 04_train_neural.py (rv, rv_d, rv_w, rv_m, bv, jump, rsv_pos, rsv_neg, rskew, rkurt, rv_morning/midday/evening, rv_IMOEX, rv_RVI,
vix, sp500, brent, gold, key_rate_cbr, fed_rate). Модель на тикер x h; цель ln RV (h=1 rv_target_h1, h=5/22 rv_target_mean_h{h}), стандартизация X и y
по обучению; последовательность строки t — признаки дней [t−21, t] (всё известно к 18:50 дня t); разбиение по дате строки t: <= 2017 / 2018 / 2019.
python3 scripts/analysis/test2019_v3_nn.py [--workers 2]  -> data/regen/test2019_v3/{lstm,gru}_h{h}.parquet (части — nn_parts/, продолжение с места)."""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[k] = '1'
import argparse, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
FEAT, OUT = MAIN / 'regen/features_pit_v3.parquet', MAIN / 'regen/test2019_v3'
FC = ['rv', 'rv_d', 'rv_w', 'rv_m', 'bv', 'jump', 'rsv_pos', 'rsv_neg', 'rskew', 'rkurt', 'rv_morning', 'rv_midday', 'rv_evening',
      'rv_IMOEX', 'rv_RVI', 'vix', 'sp500', 'brent', 'gold', 'key_rate_cbr', 'fed_rate']
SEQ, HS = 22, (1, 5, 22)
tcol = lambda h: 'rv_target_h1' if h == 1 else f'rv_target_mean_h{h}'


def task(arg):
    arch, tk, h = arg
    fp = OUT / 'nn_parts' / f'{arch}_{tk}_h{h}.parquet'
    if fp.exists():
        return str(fp), 0.0, -1
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(42); np.random.seed(42)
    t0 = time.time(); t = tcol(h)
    g = pd.read_parquet(FEAT, columns=['date', 'ticker', t] + FC)
    g = g[g.ticker == tk].sort_values('date').reset_index(drop=True); g['date'] = pd.to_datetime(g.date); g = g[g.date.dt.year <= 2019].reset_index(drop=True)
    F = np.nan_to_num(g[FC].ffill().fillna(0).values.astype(np.float64), nan=0, posinf=0, neginf=0)
    idx = np.array([i for i in range(SEQ - 1, len(g)) if np.isfinite(g[t].iat[i]) and g[t].iat[i] > 0])
    yr = g.date.dt.year.values[idx]; tr, va, te = idx[yr <= 2017], idx[yr == 2018], idx[yr == 2019]
    mu, sd = F[: tr.max() + 1].mean(0), F[: tr.max() + 1].std(0) + 1e-8  # стандартизация по строкам обучения
    Z = ((F - mu) / sd).astype(np.float32)
    Xs = lambda ii: torch.tensor(np.stack([Z[i - SEQ + 1: i + 1] for i in ii]))
    ly = np.log(g[t].values + 1e-10); ym, ys = ly[tr].mean(), ly[tr].std()
    Y = lambda ii: torch.tensor(((ly[ii] - ym) / ys).astype(np.float32))
    Xtr, Ytr, Xva, Yva, Xte = Xs(tr), Y(tr), Xs(va), Y(va), Xs(te)

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.rnn = (nn.LSTM if arch == 'lstm' else nn.GRU)(len(FC), 64, 2, dropout=0.2, batch_first=True); s.fc = nn.Linear(64, 1)

        def forward(s, x): return s.fc(s.rnn(x)[0][:, -1]).squeeze(-1)
    net, mse = Net(), nn.MSELoss(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5) if arch == 'lstm' else None
    best, bst, wait, ep = np.inf, None, 0, 0
    gen = torch.Generator().manual_seed(42)
    for ep in range(100):
        net.train()
        for b in torch.randperm(len(Xtr), generator=gen).split(32):
            opt.zero_grad(); loss = mse(net(Xtr[b]), Ytr[b]); loss.backward()
            if arch == 'lstm': nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad(): vl = float(mse(net(Xva), Yva))
        if sch: sch.step(vl)
        if vl < best - 1e-9: best, bst, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 10: break
    net.load_state_dict(bst); net.eval()
    rows = []
    with torch.no_grad():
        for ii, X, part in ((va, Xva, 'val'), (te, Xte, 'test')):
            p = np.exp(net(X).numpy() * ys + ym)
            rows.append(pd.DataFrame({'date': g.date.values[ii], 'ticker': tk, 'year': g.date.dt.year.values[ii], 'part': part,
                                      'rv_actual': g[t].values[ii], 'rv_pred': np.clip(p, 1e-10, None), 'epochs': ep + 1, 'val_mse_std': best}))
    pd.concat(rows).to_parquet(fp, index=False)
    return str(fp), time.time() - t0, ep + 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); a = ap.parse_args()
    (OUT / 'nn_parts').mkdir(parents=True, exist_ok=True)
    tks = sorted(pd.read_parquet(FEAT, columns=['ticker']).ticker.unique())
    tasks = [(ar, tk, h) for h in HS for ar in ('lstm', 'gru') for tk in tks]
    print(time.strftime('%F %T'), 'старт', len(tasks), 'моделей', flush=True)
    with Pool(a.workers) as P:
        for i, (fp, sec, ep) in enumerate(P.imap_unordered(task, tasks), 1):
            print(f'{i}/{len(tasks)} {Path(fp).stem} {sec:.0f} с, эпох {ep}', flush=True)
    for h in HS:
        for ar in ('lstm', 'gru'):
            pd.concat([pd.read_parquet(OUT / 'nn_parts' / f'{ar}_{tk}_h{h}.parquet') for tk in tks]).to_parquet(OUT / f'{ar}_h{h}.parquet', index=False)
    print(time.strftime('%F %T'), 'конец', flush=True)


if __name__ == '__main__':
    main()
