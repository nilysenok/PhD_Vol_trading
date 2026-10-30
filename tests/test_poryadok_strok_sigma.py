"""Расследование (К-5, PR #805): σ̂ шагов f1/f5 меняется при перестановке строк матрицы внутри даты.

Причина (воспроизведена на старой матрице features_pit_v3_rrv / _d7_h5, шаг 1, seed 42 — scratchpad rassl_poryadok/repro.py):
  1) l2_lib.load_base сортирует только по дате (`sort_values('date')`, quicksort — неустойчивая): порядок тикеров внутри даты зависит от
     порядка строк в файле и от размера массива, т. е. канонического порядка строк нет;
  2) признаки шага от порядка НЕ зависят (матрица X, цель, смещение — совпадают побайтно после выравнивания по (date, ticker));
     позиционного лага без groupby('ticker') и выравнивания по позиции между таблицами нет;
  3) подвыборка строк бустинга — XGBoost `subsample`, LightGBM `bagging_fraction` — выбирает строки по ПОЗИЦИИ при фиксированном seed.
     Другая позиция тикера внутри даты = другая подвыборка = другая модель (эффект того же порядка, что смена seed).
     Без подвыборки (subsample = 1, bagging_fraction = 1) прогноз от порядка не зависит.
Вторичный источник, синтетикой здесь не покрыт: у f5 нога HAR-J HuberRegressor (L-BFGS) на старой матрице h=5 при перестановке
даёт смещение/признак har_ln с расхождением до 1e-4 по ln (у f1 — МНК/HAR-X⁺, расхождения 0).

Тест `test_prognoz_ne_zavisit_ot_poryadka_strok_v_fajle` падает на нынешнем коде (это и есть дефект: прогноз зависит от
порядка строк в файле) и помечен xfail(strict=True): после починки он пройдёт, strict-метка уронит прогон, и починка снимет метку.
Остальные проходят и показывают, что причина — в п. 1 и 3, а не в признаках.
Синтетика, сеть и data/ не читаются."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import l2_lib as L  # noqa: E402

TICKERS = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF']
FEATS = ['rv', 'rv_w', 'rv_m', 'x1', 'x2']
XGB_P = dict(max_depth=4, learning_rate=0.1, subsample=0.6, colsample_bytree=1.0, min_child_weight=1.0, reg_alpha=0.0, reg_lambda=1.0)
LGB_P = dict(num_leaves=15, learning_rate=0.1, bagging_fraction=0.6, feature_fraction=1.0, min_data_in_leaf=20, reg_alpha=0.0, reg_lambda=0.0)


def _matrica():
    """6 тикеров × будни 2012–2017; rv — лог-AR(1) по тикеру, цель rv_target_h1 = rv следующего дня тикера (через groupby)."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range('2012-01-02', '2017-12-29')
    fr = []
    for i, tk in enumerate(TICKERS):
        e = rng.normal(0, 0.35, len(days))
        z = np.zeros(len(days))
        for t in range(1, len(days)):
            z[t] = 0.9 * z[t - 1] + e[t]
        rv = np.exp(-8 + 0.2 * i + z)
        d = pd.DataFrame({'date': days, 'ticker': tk, 'rv': rv})
        d['rv_w'] = d.rv.rolling(5, min_periods=1).mean()
        d['rv_m'] = d.rv.rolling(22, min_periods=1).mean()
        d['x1'] = rng.normal(size=len(d))
        d['x2'] = np.log(d.rv) + rng.normal(0, 0.5, len(d))
        d['rv_target_h1'] = d.rv.shift(-1)
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


@pytest.fixture(scope='module')
def dve_matricy(tmp_path_factory):
    """Один и тот же набор строк, записанный в двух порядках (исходный и перемешанный); конфиг признаков."""
    d = tmp_path_factory.mktemp('poryadok')
    A = _matrica()
    p0, p1, cfg = d / 'm0.parquet', d / 'm1.parquet', d / 'm.config.json'
    A.to_parquet(p0, index=False)
    A.sample(frac=1.0, random_state=1).to_parquet(p1, index=False)
    cfg.write_text(json.dumps({'feature_cols': FEATS}))
    return p0, p1, cfg


def _shag(path, cfg, monkeypatch, model):
    monkeypatch.setattr(L, 'FEAT_CFG', cfg)
    A, fc = L.load_base(path)
    return A, L.step_data(A, fc, 1, 1, model, {}, 'net_takogo_tega')


def _po_klyuchu(D, part, arr):
    d = D['parts'][part]
    k = pd.DataFrame({'date': pd.to_datetime(d['date'].values), 'ticker': d['ticker'].values})
    o = np.lexsort((k.ticker.values, k.date.values))
    return k.iloc[o].reset_index(drop=True), np.asarray(arr)[o]


def _fit(model, D, bez_podvyborki=False):
    p = dict(XGB_P if model == 'xgb' else LGB_P)
    if bez_podvyborki:
        p['subsample' if model == 'xgb' else 'bagging_fraction'] = 1.0
    pv, pt, bi, _, _ = L.fit(model, p, D, {}, seed=42, nmax=200)
    return _po_klyuchu(D, 'va', pv)[1], _po_klyuchu(D, 'te', pt)[1], bi


def test_net_dubley_klyucha(dve_matricy, monkeypatch):
    """Гипотеза 5 отброшена: дублей (date, ticker) после load_base нет; даты идут одинаково в обоих порядках файла."""
    p0, p1, cfg = dve_matricy
    a, _ = _shag(p0, cfg, monkeypatch, 'xgb')
    b, _ = _shag(p1, cfg, monkeypatch, 'xgb')
    assert a.duplicated(['date', 'ticker']).sum() == 0 and b.duplicated(['date', 'ticker']).sum() == 0
    assert (a.date.values == b.date.values).all()


@pytest.mark.xfail(strict=True, reason='дефект порядка строк (К-5, #805): load_base сортирует только по дате, неустойчиво')
def test_load_base_kanonicheskij_poryadok(dve_matricy, monkeypatch):
    """Причина 1 (падает на нынешнем коде): один набор строк в двух порядках файла -> после load_base тикеры внутри даты идут
    по-разному. После починки (канонический порядок) — одинаково."""
    p0, p1, cfg = dve_matricy
    a, _ = _shag(p0, cfg, monkeypatch, 'xgb')
    b, _ = _shag(p1, cfg, monkeypatch, 'xgb')
    assert (a.ticker.values == b.ticker.values).all(), 'порядок тикеров внутри даты зависит от порядка строк в файле'


@pytest.mark.parametrize('model', ['xgb', 'lgbm'])
def test_priznaki_shaga_ne_zavisyat_ot_poryadka(dve_matricy, monkeypatch, model):
    """Гипотезы 1, 2, 4 отброшены: X, цель обучения и смещение шага по ключу (date, ticker) совпадают побайтно; составы частей тоже."""
    p0, p1, cfg = dve_matricy
    _, D0 = _shag(p0, cfg, monkeypatch, model)
    _, D1 = _shag(p1, cfg, monkeypatch, model)
    for part in ('tr', 'va', 'te'):
        k0, x0 = _po_klyuchu(D0, part, D0['X'][part])
        k1, x1 = _po_klyuchu(D1, part, D1['X'][part])
        assert k0.equals(k1)
        assert np.array_equal(x0, x1, equal_nan=True)
    assert np.array_equal(_po_klyuchu(D0, 'tr', D0['ytr_ln'])[1], _po_klyuchu(D1, 'tr', D1['ytr_ln'])[1])
    assert np.array_equal(_po_klyuchu(D0, 'tr', D0['off']['tr'])[1], _po_klyuchu(D1, 'tr', D1['off']['tr'])[1])


@pytest.mark.parametrize('model', ['xgb', 'lgbm'])
def test_bez_podvyborki_prognoz_ne_zavisit_ot_poryadka(dve_matricy, monkeypatch, model):
    """Контроль причины 3: без подвыборки строк прогноз от порядка не зависит (до точности суммирования float)."""
    p0, p1, cfg = dve_matricy
    _, D0 = _shag(p0, cfg, monkeypatch, model)
    _, D1 = _shag(p1, cfg, monkeypatch, model)
    v0, t0, b0 = _fit(model, D0, bez_podvyborki=True)
    v1, t1, b1 = _fit(model, D1, bez_podvyborki=True)
    assert b0 == b1
    assert np.abs(v0 - v1).max() <= 1e-6 and np.abs(t0 - t1).max() <= 1e-6


@pytest.mark.xfail(strict=True, reason='дефект порядка строк (К-5, #805): падает до починки load_base; починка снимает метку (как #249 → #250)')
@pytest.mark.parametrize('model', ['xgb', 'lgbm'])
def test_prognoz_ne_zavisit_ot_poryadka_strok_v_fajle(dve_matricy, monkeypatch, model):
    """ДЕФЕКТ (падает на нынешнем коде): точка с подвыборкой строк (subsample / bagging_fraction = 0,6), seed 42, один и тот же набор
    строк в двух порядках файла -> разные ln-прогнозы по ключу. Должно проходить после канонического порядка строк в load_base."""
    p0, p1, cfg = dve_matricy
    _, D0 = _shag(p0, cfg, monkeypatch, model)
    _, D1 = _shag(p1, cfg, monkeypatch, model)
    v0, t0, _ = _fit(model, D0)
    v1, t1, _ = _fit(model, D1)
    d = max(np.abs(v0 - v1).max(), np.abs(t0 - t1).max())
    assert d <= 1e-6, f'прогноз зависит от порядка строк в файле: max|Δln σ̂| = {d:.3e} (подвыборка строк по позиции)'
