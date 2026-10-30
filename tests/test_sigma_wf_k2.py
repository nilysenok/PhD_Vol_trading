"""К-2: RRV и Д7 без ДСВД (scripts/dannye/wf_rrv_d7_bez_dsvd.py). Синтетика, без сети и без данных проекта; календарь — из репозитория."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts/dannye"), str(REPO / "scripts/analysis")]
import wf_baza_bez_dsvd as K1  # noqa: E402
import wf_rrv_d7_bez_dsvd as K2  # noqa: E402
import d7_block as D7  # noqa: E402

SAT = pd.Timestamp("2025-01-11")  # условная дата ДСВД: суббота между пятницей и понедельником
DSVD = {SAT}


def _rrv():
    d = pd.to_datetime(["2025-01-09", "2025-01-10", "2025-01-11", "2025-01-13"])
    return pd.DataFrame({"date": d, "ticker": "SBER", "rrv_pk": [1.0, 2.0, 100.0, 3.0], "rrv_gk": [1.0, 2.0, 100.0, 3.0]})


def _keys():
    return pd.DataFrame({"date": pd.to_datetime(["2025-01-09", "2025-01-10", "2025-01-13"]), "ticker": "SBER"})


def test_rrv_okno_ne_zahvatyvaet_dsvd():
    rrv = K2.rrv_bez_dsvd(_rrv(), DSVD, pd.Timestamp("2026-02-03"))
    assert SAT not in set(rrv.date)
    out = K2.sobrat_rrv(_keys(), rrv)
    assert out.rrv_pk_w.iloc[2] == pytest.approx(np.log((1 + 2 + 3) / 3))  # Чт, Пт, Пн


def test_mutatsiya_dsvd_bar_v_rrv_lovitsya():
    # без фильтра окно понедельника захватывает бар субботы -> значение иное, и тест выше упал бы
    mut = K2.sobrat_rrv(_keys(), _rrv())
    assert mut.rrv_pk_w.iloc[2] == pytest.approx(np.log((1 + 2 + 100 + 3) / 4))
    assert mut.rrv_pk_w.iloc[2] != pytest.approx(np.log((1 + 2 + 3) / 3))


def test_rrv_obrezka_po_konec():
    r = _rrv()
    r.loc[3, "date"] = pd.Timestamp("2026-02-04")
    assert pd.Timestamp("2026-02-04") not in set(K2.rrv_bez_dsvd(r, DSVD, pd.Timestamp("2026-02-03")).date)


def test_matritsa_s_dsvd_otvergaetsya():
    base = _keys()
    K2.proverit_bez_dsvd(base, DSVD)  # без ДСВД — молча
    with pytest.raises(ValueError):
        K2.proverit_bez_dsvd(pd.concat([base, pd.DataFrame({"date": [SAT], "ticker": "SBER"})]), DSVD)


def test_mutatsiya_okno_d7_cherez_dsvd_lovitsya():
    # окно Д7 идёт по строкам матрицы: лишняя строка (ДСВД) меняет годовой перцентиль RVI -> поэтому вход с ДСВД отвергается
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=300)
    v = pd.DataFrame({"close_RVI": 20 + rng.normal(size=300).cumsum() * 0.1, "rv_IMOEX": 0.2}, index=idx)
    a = D7._rvi(v)["d7_rvi_pctl252"]
    t = idx[200] + pd.Timedelta(hours=12)  # «воскресенье» с RVI, равным минимуму ряда
    extra = pd.DataFrame({"close_RVI": float(v.close_RVI.min()) - 1, "rv_IMOEX": 0.2}, index=[t])
    b = D7._rvi(pd.concat([v, extra]).sort_index())["d7_rvi_pctl252"].drop(t)
    assert not np.allclose(a.iloc[-90:].to_numpy(), b.iloc[-90:].to_numpy(), equal_nan=True)


def _tab(n=30):
    return pd.DataFrame({"date": pd.bdate_range("2024-12-01", periods=n), "ticker": "A", "x": np.arange(n, dtype="float64")})


def test_granitsa_2025_lovit_pravku_v_2024():
    old = _tab()
    new = old.copy()
    i = new.index[new.date == pd.Timestamp("2024-12-31")][0]
    new.loc[i, "x"] += 1e-9
    assert K2.GRANICA == pd.Timestamp("2025-01-01") == K1.GRANICA
    assert K2.razlichiya(new, old) == ["x"]
    new2 = old.copy()
    new2.loc[new2.index[new2.date == pd.Timestamp("2025-01-02")][0], "x"] += 1
    assert K2.razlichiya(new2, old) == []  # правки 2025 сверкой не ловятся — так и задумано


def test_mutatsiya_granitsa_2024_propuskaet_pravku():
    old = _tab()
    new = old.copy()
    new.loc[new.index[new.date == pd.Timestamp("2024-12-31")][0], "x"] += 1
    assert K2.razlichiya(new, old, pd.Timestamp("2024-01-01")) == []  # мутация «2025 → 2024» слепа; реальная граница видит
    assert K2.razlichiya(new, old) == ["x"]


def test_razlichiya_dlina_i_dtype():
    old = _tab()
    assert K2.razlichiya(old.iloc[1:], old) == ["__len__"]
    assert K2.razlichiya(old.assign(x=old.x.astype("float32")), old) == ["x"]


def test_kalendar_dsvd_subbota_ostayotsya():
    d = K1.daty_dsvd()
    assert len(d) > 50 and K2.SUBBOTA not in d and pd.Timestamp("2025-03-01") in d


def _rv_frame(sdvig):
    rng = np.random.default_rng(3)
    n = 400
    lr = rng.normal(size=n)
    f = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=n), "ticker": "A", "rv": np.exp(lr)})
    f["x"] = pd.Series(lr).shift(-sdvig).to_numpy()  # sdvig=1: признак знает завтра
    return f


def test_sdvig_pik_na_nule_i_utechka_lovitsya():
    ot, do = pd.Timestamp("2025-01-01"), pd.Timestamp("2030-01-01")
    assert K2.maks_na_nule(K2.profil(_rv_frame(0), "x", ot, do))
    assert not K2.maks_na_nule(K2.profil(_rv_frame(1), "x", ot, do))  # утечка +1 день
    assert not K2.maks_na_nule({-2: np.nan, -1: 0.1, 0: 0.5, 1: 0.1})
