"""Ист-v6: боевой загрузчик рядов оценщика v6 (fastbt/istochniki_v6.py). Синтетика, без сети и без файлов данных.

Известный ответ: σ̂ rv5 дня d = √(среднее RV за d−4…d) по 10-минуткам; point-in-time: бары после d не меняют ряды ≤ d
(мутация «ряд с днём d+1» — сдвиг `_al` на −1 — роняет эту проверку); дыра остаётся дырой (Д4): NaN в Dannye, ValueError в td_rel.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850 as d18, istochniki_v6 as iv, period_v2, sigma_real, vneshnie_1850 as v18
from fastbt.eval_v6_osi import rel_td_ind

DAYS = pd.bdate_range("2019-01-02", periods=90)
MIN = [9 * 60 + 50 + 10 * k for k in range(54)]          # бары 09:50 … 18:40 (до 18:50)


def _bars(seed: int, days=DAYS, s0: float = 100.0) -> pd.DataFrame:
    rng, rows, p = np.random.default_rng(seed), [], s0
    for d in days:
        for m in MIN:
            q = p * float(np.exp(rng.normal(0, 0.002)))
            rows.append((d + pd.Timedelta(minutes=m), p, max(p, q) * 1.0005, min(p, q) * 0.9995, q, 1000.0))
            p = q
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


class Fake(iv.Fajly):
    def __init__(self, akcii: dict, indeksy: dict):
        self.end, self.main, self.a, self.i = pd.Timestamp("2020-01-01"), None, akcii, indeksy

    def akcii(self, t):
        return self.a[t]

    def indeks(self, name):
        return self.i.get(name.lower())

    def zakrytie(self, t):
        return v18.close_1850(self.a[t], t)

    def brent(self):
        return {}, pd.DataFrame({"secid": [], "lsttrade": [], "roll": []})

    def kalendar(self):
        return pd.DataFrame({"date": [], "known_from": []}), pd.DataFrame()


def _td(bars, t="AAA"):
    d = d18.aggregate(bars, t)
    c = lambda k: d[k].to_numpy(float)       # noqa: E731
    return SimpleNamespace(ticker=t, dates=d["date"].to_numpy(), n=len(d), sigma={}, close=c("close"), high=c("high"),
                           low=c("low"), volume=c("volume"))


def _sbor(a=None, i=None):
    f = Fake(a or {"AAA": _bars(1), "BBB": _bars(2)}, i or {"imoex": _bars(3)})
    return iv._Sbor(f, list(f.a))


def test_sigma_rv5_izvestnyj_otvet():
    b = _bars(1)
    td, rv = _td(b), sigma_real.rv_1850(b)
    d = _sbor().dannye("AAA", td)
    ожид = np.sqrt(rv.rolling(5).mean().reindex(pd.DatetimeIndex(td.dates)).to_numpy())
    assert np.isnan(d.sig5[:4]).all() and np.allclose(d.sig5[4:], ожид[4:]) and np.isfinite(d.sig5[4:]).all()
    assert np.isnan(d.sig66[:65]).all() and np.isfinite(d.sig66[65:]).all()
    assert np.isnan(d.Z[0]) and np.isfinite(d.Z[70])


def _sreda_do(s: int):
    """(ряды как есть, ряды при испорченных барах после дня DAYS[s])."""
    a, i = {"AAA": _bars(1), "BBB": _bars(2)}, {"imoex": _bars(3)}
    base = _sbor(a, i).dannye("AAA", _td(a["AAA"]))
    bad = {k: v.copy() for k, v in a.items()}
    bi = {k: v.copy() for k, v in i.items()}
    for df in (bad["AAA"], bad["BBB"], bi["imoex"]):
        later = df["timestamp"] >= DAYS[s + 1]
        for c in ("open", "high", "low", "close"):
            df.loc[later, c] = df.loc[later, c] * np.linspace(1.0, 3.0, later.sum())
    return base, _sbor(bad, bi).dannye("AAA", _td(a["AAA"]))


def _pit_ok() -> bool:
    s = 70
    x, y = _sreda_do(s)
    pole = ("sig5", "sig22", "sig66", "Z")
    return all(np.array_equal(getattr(x, p)[:s + 1], getattr(y, p)[:s + 1], equal_nan=True) for p in pole) \
        and all(np.array_equal(x.rezhim[k][:s + 1], y.rezhim[k][:s + 1]) for k in x.rezhim) and not \
        np.array_equal(x.sig5[s + 1:], y.sig5[s + 1:], equal_nan=True)


def test_point_in_time_bary_posle_d_ne_menyayut_ryady_do_d():
    assert _pit_ok()


def test_mutaciya_ryad_s_dnyom_d_plus_1_lovitsya(monkeypatch):
    monkeypatch.setattr(iv, "_al", lambda s, ix: s.shift(-1).reindex(ix).to_numpy(float))
    assert not _pit_ok()


def test_dyra_v_imoex_ostayotsya_nan_bez_ffill():
    i = _bars(3)
    day = DAYS[60]
    i = i[i["timestamp"].dt.normalize() != day]
    d = _sbor(i={"imoex": i}).dannye("AAA", _td(_bars(1)))
    k = list(DAYS).index(day)
    assert np.isnan(d.Z[k]) and np.isnan(d.Z[k + 1])      # z_d = r_d/σ_{d−1}: и день дыры, и следующий — без значения
    assert not d.rezhim["ma200"][k]                       # нет значения режима — вход запрещён, не «как вчера»


def test_dyra_v_td_rel_otkaz_ocenshchika_ne_ffill():
    a = _bars(1)
    tick = a[a["timestamp"].dt.normalize() != DAYS[40]]
    s = _sbor({"AAA": tick}, {"imoex": _bars(3)})
    df = s.rel("AAA", "rynok")
    assert DAYS[40] not in set(pd.to_datetime(df["date"]))
    with pytest.raises(ValueError, match="нет 1 дней сетки"):
        rel_td_ind(df, _td(a))
    assert len(rel_td_ind(s.rel("AAA", "rynok"), _td(tick))[0].dates) == len(DAYS) - 1      # без дыры в сетке — считается


def test_progrev_td_rel_tolko_nachalo_i_tolko_po_flagu():
    a = _bars(1)
    td, df = _td(a), _sbor().rel("AAA", "rynok")
    with pytest.raises(ValueError, match="нет 1 дней"):
        rel_td_ind(df.iloc[1:], td)                                    # по умолчанию начало без td_rel — отказ
    t2, _ = rel_td_ind(df.iloc[1:], td, progrev=True)                   # прогрев: NaN, не заполнено
    assert np.isnan(t2.close[0]) and np.allclose(t2.close[1:], df["close"].to_numpy()[1:])
    with pytest.raises(ValueError, match="нет 1 дней"):
        rel_td_ind(df.drop(df.index[30]), td, progrev=True)              # дыра внутри — отказ и при прогреве


def test_rel_rezhim_syroy_i_sektor_bez_indeksa():
    s = _sbor()
    assert np.allclose(s.rel("AAA", "syroy")["close"], d18.aggregate(_bars(1), "AAA")["close"])
    assert len(s.rel("AAA", "sektor")) == len(DAYS)       # тикера нет в PIT-составе → откат на IMOEX (с флагом), не пусто


def test_vn_ishodnye_otkazy():
    s = _sbor()
    with pytest.raises(ValueError, match="vn_faktor"):
        s.vn("AAA", "gold")
    with pytest.raises(FileNotFoundError, match="rvi"):
        s.vn("AAA", "rvi")


def test_vn_z_beta_izvestnyj_otvet():
    u = _bars(4, s0=60.0)
    s = _sbor(i={"imoex": _bars(3), "usdrub": u})
    z, be = s.vn("AAA", "usdrub")
    dx = np.log(v18.close_1850(u, "u")).diff()
    sg = sigma_real.sigma_rv(sigma_real.rv_1850(u), 22)
    k = 50
    assert z.iloc[k] == pytest.approx(dx.iloc[k] / sg.iloc[k - 1]) and np.isnan(be.iloc[:126].to_numpy()).all()


def test_progon_sobol_podayot_istochniki_tolko_ocenshchiku_v6(monkeypatch):
    from fastbt import eval_v6, progon_sobol as ps
    kw = {}
    monkeypatch.setattr(eval_v6, "EvaluatorV6", lambda *a, **k: kw.update(k) or "evr")
    monkeypatch.setattr(iv, "zagruzit", lambda per, tik: ("ist", per, tuple(tik)))
    monkeypatch.setattr(ps, "OCENSHCHIK_V5", ("fastbt.eval_v6", "EvaluatorV6"))
    inp = SimpleNamespace(sk=SimpleNamespace(tickers=["A", "B"]))
    assert ps.ocenshchik("S1", inp, ps.Rezhim("v6", "train4", "seed", False, "t0d")) == "evr"
    assert kw["ist"] == ("ist", "train4", ("A", "B")) and kw["fdr"] == "t0d"


def _sha(a=None, i=None):
    f = Fake(a or {"AAA": _bars(1)}, i or {"imoex": _bars(3)})
    return iv.sha_ryadov(f, ["AAA"])


def test_sha_menyaetsya_pri_podmene_lyubogo_ryada():
    base = _sha()
    assert base == _sha() and len(base) == 64
    x = _bars(1)
    x.loc[100, "close"] *= 1.0001
    assert _sha({"AAA": x}) != base                                    # бары тикера
    y = _bars(3)
    y.loc[7, "high"] += 1e-6
    assert _sha(i={"imoex": y}) != base                                # IMOEX
    assert _sha(i={"imoex": _bars(3), "rvi": _bars(5)}) != base        # появился ряд фактора


def test_mutaciya_sha_postoyannyj_lovitsya(monkeypatch):
    monkeypatch.setattr(iv, "sha_ryadov", lambda f, t: "0" * 64)
    x = _bars(1)
    x.loc[100, "close"] *= 1.0001
    assert iv.sha_ryadov(None, None) == iv.sha_ryadov(None, None)
    with pytest.raises(AssertionError):                                # мутация «ряд подменён, sha тот же» — проверка выше упала бы
        assert _sha({"AAA": x}) != _sha()


def test_sha_v_klyuche_tochki(monkeypatch):
    from fastbt.eval_v6 import EvaluatorV6, IstochnikiV6
    from fastbt.etap1_eval_v2 import EvaluatorV2
    keys = []
    monkeypatch.setattr(EvaluatorV2, "evaluate_v2", lambda self, pt, period, vozm=None: keys.append(self._fkey))
    monkeypatch.setattr(EvaluatorV6, "__init__", lambda self, sid, ist: self.__dict__.update(sid=sid, ist=ist, inp=SimpleNamespace(sigma={})))
    monkeypatch.setattr("fastbt.eval_v6.razdelit", lambda sid, pt: ({}, {"rezhim": "imoex_ma200"}))
    monkeypatch.setattr("fastbt.eval_v6.razbor", lambda *a: SimpleNamespace(filtr={}, nov={}, pt4={}, sp={}, ext={}))
    monkeypatch.setattr("fastbt.eval_v6.osi6.primenit", lambda *a: None)
    monkeypatch.setattr("fastbt.eval_v6.tip", lambda sid: "kontr")
    for sha in ("aa", "bb"):
        EvaluatorV6.evaluate_v2(EvaluatorV6("S1", IstochnikiV6(sha=sha)), {}, "train4")
    assert keys[0] != keys[1] and ("ist", "aa") in keys[0]


def test_konec_perioda_i_zamok(tmp_path, monkeypatch):
    assert iv.konec("train4") == pd.Timestamp("2022-01-01") == iv.konec("v2")      # сетка td оценщика идёт до конца dev-ряда
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "net.json")
    with pytest.raises(period_v2.ZamokVladeltsa):
        iv.konec("wf")
    (tmp_path / "net.json").write_text('{"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "' + "a" * 64
                                       + '", "data_snyatiya": "2026-10-30"}')
    assert iv.konec("wf") == pd.Timestamp("2026-03-01")      # с отметкой (wf-L1b): конец периода + 1 день, исключительно
    assert iv.konec("train4") == pd.Timestamp("2022-01-01")  # dev-периоды — как было
