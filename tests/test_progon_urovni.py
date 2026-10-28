"""Л-11 (Ж-078): уровни 2–3, A′ из рядов по парам. Синтетика + мутации + IG-4 на данных ≤ 2019.
python3 -m pytest tests/test_progon_urovni.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import progon_urovni as pu

DEF = {"ves": "ravnyj", "lam": "net", "rho": "net", "sostav": "vse6", "tsel_vol": "net", "okno_vol": 63, "tormoz_mdd": "net"}
T = 600
DATY = (np.datetime64("2018-01-01") + np.arange(T)).astype("datetime64[D]")


def synth(seed=0, k=3, n=3):
    r = np.random.default_rng(seed)
    kom = {f"S{i + 1}": r.normal(0, 0.01, (n, T)) for i in range(k)}
    kom["S1"][0, :50] = np.nan                       # разное число присутствующих пар на дату
    kom["S2"][:, :30] = np.nan
    return kom


def sd(c=None, **kw):
    return {**DEF, **(c or {}), **kw}


def test_umolchaniya_srednee_po_prisutstvuyushchim_parm():
    kom = synth()
    st = np.stack(list(kom.values())).reshape(-1, T)
    ref = np.nanmean(st, axis=0)
    got = pu.a_shtrih(kom, sd(), DATY)
    assert np.abs(got - ref).max() <= 1e-15


def test_izvestnye_ryady_dve_strategii_tri_tikera():
    kom = {"S1": np.array([[1., 2.], [3., 4.], [5., 6.]]), "S2": np.array([[7., np.nan], [9., 10.], [11., 12.]])}
    got = pu.a_shtrih(kom, sd(), DATY[:2])
    assert np.allclose(got, [(1 + 3 + 5 + 7 + 9 + 11) / 6, (2 + 4 + 6 + 10 + 12) / 5])


def ortho(k=3, t=T):
    """Ряды с нулевой взаимной корреляцией на любом окне, кратном 4."""
    base = [[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]
    return np.array([np.tile(b, t // 4) * 0.01 for b in base[:k]])


def test_obr_korr_pri_rho_nul_raven_ravnyj():
    S = ortho()
    for ves in ("obr_korr_126", "obr_korr_252"):
        a = pu._vesa_strategij(S, DATY, ves, None, [0, 1, 2])
        i = 400
        assert np.allclose(a[:, i], 1 / 3)
    kom = {f"S{i + 1}": s[None, :] for i, s in enumerate(ortho())}
    d0 = pu.a_shtrih(kom, sd(), DATY)
    d1 = pu.a_shtrih(kom, sd(ves="obr_korr_252"), DATY)
    assert np.abs(d0[300:] - d1[300:]).max() <= 1e-12


def corr_set():
    r = np.random.default_rng(1)
    a, b = r.normal(0, 0.01, T), r.normal(0, 0.01, T)
    return np.array([a, a, b])                         # S1 == S2: дубль, ρ = 1


def test_obr_korr_nakazyvaet_korrelirovannye():
    a = pu._vesa_strategij(corr_set(), DATY, "obr_korr_252", None, [0, 1, 2])[:, 400]
    assert a[2] > a[0] == pytest.approx(a[1]) and a.sum() == pytest.approx(1)


def test_dubl_odna_komponenta_odin_ves():
    S = corr_set()
    a = pu._vesa_strategij(S, DATY, "ravnyj", 0.7, [0, 1, 2])[:, 400]
    assert np.allclose(a, [0.25, 0.25, 0.5])           # дубль делит один вес, а не получает два
    assert np.allclose(pu._vesa_strategij(S, DATY, "ravnyj", None, [0, 1, 2])[:, 400], 1 / 3)


def test_vesa_pit_budushchee_i_den_d_ne_vliyayut():
    S = np.random.default_rng(3).normal(0, 0.01, (3, T))
    base = pu._vesa_strategij(S, DATY, "obr_korr_126", None, [0, 1, 2])
    starts = pu._starty(DATY)
    i = starts[10]
    S2 = S.copy()
    S2[:, i:] += np.random.default_rng(4).normal(0, 0.05, (3, T - i))     # возмущение с дня d и позже
    S2[0, i:] = S[0, i:] * 3 + S2[1, i:]
    new = pu._vesa_strategij(S2, DATY, "obr_korr_126", None, [0, 1, 2])
    assert np.array_equal(base[:, :starts[11]], new[:, :starts[11]])      # вес в день пересмотра не видит день d
    assert not np.array_equal(base[:, starts[11]:], new[:, starts[11]:])  # а следующий месяц — видит


def poz_sovp():
    """Три стратегии, один тикер: сигналы совпадают в первые 100 дней, дальше только одна в рынке."""
    p = {f"S{i}": np.ones((1, T)) for i in (1, 2, 3)}
    for s in ("S2", "S3"):
        p[s][0, 100:] = 0.0
    return p


def test_limit_lambda_pri_sovpadenii_signalov():
    kom = {f"S{i}": np.full((1, T), 0.01) for i in (1, 2, 3)}
    base = pu.a_shtrih(kom, sd(), DATY)
    lim = pu.a_shtrih(kom, sd(lam="1"), DATY, poz_sovp())
    # λ = 1: |Σ a·pos| = 1 > a_max = 1/3 -> множитель 1/3 в первые 100 дней; дальше Σ = a_max -> без изменений
    assert np.allclose(lim[:100], base[:100] / 3) and np.allclose(lim[100:], base[100:])
    lim2 = pu.a_shtrih(kom, sd(lam="2"), DATY, poz_sovp())
    assert np.allclose(lim2[:100], base[:100] * 2 / 3)
    with pytest.raises(ValueError):
        pu.a_shtrih(kom, sd(lam="1"), DATY)


def test_sostav_proshedshie_trebuet_spisok():
    kom = synth()
    r = pu.a_shtrih(kom, sd(sostav="proshedshie"), DATY, proshli=["S3"])
    assert np.allclose(r, np.nanmean(kom["S3"], axis=0), equal_nan=True)
    with pytest.raises(ValueError):
        pu.a_shtrih(kom, sd(sostav="proshedshie"), DATY)


def test_tsel_vol_i_tormoz_bez_dnya_d():
    r = np.random.default_rng(5).normal(0, 0.01, T)
    A2 = r.copy()
    for kw in (dict(tsel=5.0, tormoz=None), dict(tsel=None, tormoz=300.0)):    # 300 %: без дня d не срабатывает, обвал дня d — срабатывает
        i = 300
        A3 = A2.copy()
        A3[i] = -5.0                                                    # обвал в день d
        s0 = pu._uroven3(A2, kw["tsel"], 63, kw["tormoz"], 0.5)
        s1 = pu._uroven3(A3, kw["tsel"], 63, kw["tormoz"], 0.5)
        assert s0[i] / A2[i] == pytest.approx(s1[i] / A3[i])            # масштаб дня d не видит день d
    s = pu._uroven3(A2, 5.0, 63, None, 0.5)
    assert 0 < np.abs(s[200:]).mean() < np.abs(A2[200:]).mean() and (np.abs(s) <= np.abs(A2) + 1e-15).all()


def test_tormoz_sreza_posle_prosadki():
    A2 = np.r_[np.full(30, 0.01), np.full(30, -0.02), np.full(30, 0.01)]
    out = pu._uroven3(A2, None, 63, 10.0, 0.5)
    assert out[40] == pytest.approx(-0.01) and out[10] == pytest.approx(0.01) and out[-1] == pytest.approx(0.005)   # просадка > 10 % -> ×0,5


def test_setka_54_na_24_odna_osnovnaya():
    cs = pu.konfiguracii()
    assert len(cs) == 1296 and sum(c["osnovnaya"] for c in cs) == 1
    assert len({tuple(sorted((k, str(v)) for k, v in c.items() if k != "osnovnaya")) for c in cs}) == 1296
    assert [c for c in cs if c["osnovnaya"]][0] == {**DEF, "osnovnaya": True}


def test_tablica_bez_rangirovaniya(tmp_path):
    kom, poz = synth(), {s: np.ones((3, T)) for s in synth()}
    sk = pu.setka()
    sk["uroven2"].update(ves=["ravnyj"], rho=["net"])        # урезанная сетка для скорости
    import fastbt.progon_urovni as m
    orig = m.konfiguracii
    m.konfiguracii = lambda s=None: orig(sk)
    try:
        df = pu.tablica_1296(kom, DATY, poz, ["S1", "S2"], out=tmp_path / "u.csv")
    finally:
        m.konfiguracii = orig
    assert df["osnovnaya"].sum() == 1 and set(df["status"]) == {"osnovnaya", "chuvstvitelnost"}
    assert "rang" not in df.columns and (tmp_path / "u.csv").exists()
    assert df.loc[df["osnovnaya"], "sharpe"].iloc[0] == pytest.approx(
        pu.metriki(pu.a_shtrih(kom, DEF, DATY))["sharpe"])


# ---------------- мутации: каждая обязана ронять соответствующий тест ----------------
def lovitsya(test):
    with pytest.raises(AssertionError):
        test()


def test_mut_umolchaniya_s_vesom_obr_korr_lovitsya(monkeypatch):
    orig = pu._vesa_strategij
    monkeypatch.setattr(pu, "_vesa_strategij", lambda S, d, ves, p, act, mini=20: orig(S, d, "obr_korr_126", p, act, mini))
    lovitsya(test_umolchaniya_srednee_po_prisutstvuyushchim_parm)


def test_mut_rho_s_dnem_d_lovitsya(monkeypatch):
    monkeypatch.setattr(pu, "_istoriya", lambda x, i, w: x[..., max(0, i - w + 1):i + 1])
    lovitsya(test_vesa_pit_budushchee_i_den_d_ne_vliyayut)


def test_mut_ves_po_vsemu_periodu_lovitsya(monkeypatch):
    monkeypatch.setattr(pu, "_istoriya", lambda x, i, w: x[..., :])
    lovitsya(test_vesa_pit_budushchee_i_den_d_ne_vliyayut)


def test_mut_dubl_s_dvumya_vesami_lovitsya(monkeypatch):
    monkeypatch.setattr(pu, "_klastery", lambda R, p: [[k] for k in range(R.shape[0])])
    lovitsya(test_dubl_odna_komponenta_odin_ves)


def test_mut_tormoz_s_dnem_d_lovitsya(monkeypatch):
    monkeypatch.setattr(pu, "_prosadka", lambda cum, i: float(cum[:i + 1].max() - cum[i]))
    lovitsya(test_tsel_vol_i_tormoz_bez_dnya_d)


def test_mut_limit_po_pozitsiyam_d_plus_1_lovitsya(monkeypatch):
    orig = pu._limit
    monkeypatch.setattr(pu, "_limit", lambda X, a, poz, lam: orig(X, a, {s: np.roll(p, -1, axis=-1) for s, p in poz.items()}, lam))
    lovitsya(test_limit_lambda_pri_sovpadenii_signalov)


# ---------------- IG-4: A′(умолчания) = A на данных ≤ 2019 (реальные кэши; нет данных — падение, не пропуск) ----------------
def a_engine():
    from fastbt import ekviv_s9, s9
    with s9.period_ctx("train4"):
        res = s9.load_cache()
    cache = {t: dict(dates=c["dates"], lr=c["lr"], A={s: dict(v) for s, v in c["A"].items()}) for t, c in res.items()}
    return ekviv_s9.s9_engine(cache, ekviv_s9.windows("train4")[2019], "train4")


def test_a_shtrih_ravno_a():
    eng = a_engine()
    comm = 0.0004
    ref = eng.port(eng.A_row, comm=comm)["excess"]
    kom, poz, daty = pu.pary_iz_dvizhka(eng, comm)
    mine = pu.a_shtrih(kom, sd(), daty)
    assert len(mine) == len(ref) > 900 and np.abs(ref.to_numpy()).max() > 0
    assert np.abs(mine - ref.to_numpy()).max() <= 1e-12
    assert daty.max() <= np.datetime64("2019-12-31")


def test_a_shtrih_ravno_a_mut_vesa_lovitsya():
    eng = a_engine()
    kom, _, daty = pu.pary_iz_dvizhka(eng, 0.0004)
    ref = eng.port(eng.A_row, comm=0.0004)["excess"].to_numpy()
    assert np.abs(pu.a_shtrih(kom, sd(ves="obr_korr_126"), daty) - ref).max() > 1e-6
