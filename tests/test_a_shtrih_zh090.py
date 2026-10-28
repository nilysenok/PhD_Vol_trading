"""Ж-090 п. 2: основной портфель A′ = обратная волатильность 63 (до d−1) + таргет 10 % (окно 63, L = 1), без тормоза, N = 1.
Синтетика; независимая эталонная реализация в тесте; пять мутаций (каждая роняет проверку).
python3 -m pytest tests/test_a_shtrih_zh090.py -q"""
import numpy as np
import pytest

from fastbt import progon_urovni as pu

T = 600
DATY = (np.datetime64("2018-01-01") + np.arange(T)).astype("datetime64[D]")
SIG = (0.01, 0.02, 0.04)                     # дневные σ трёх стратегий; σ портфеля ≈ 16 % годовых, таргет 10 % кусается
ZH090 = {"ves": "obr_vol_63", "lam": "net", "rho": "net", "sostav": "vse6", "tsel_vol": 10, "okno_vol": 63,
         "tormoz_mdd": "net", "L": 1}
RAVNYJ = {"ves": "ravnyj", "lam": "net", "rho": "net", "sostav": "vse6", "tsel_vol": "net", "okno_vol": 63,
          "tormoz_mdd": "net"}


def synth(scale=1.0, seed=3):
    r = np.random.default_rng(seed)
    return {f"S{i + 1}": r.normal(0, s * scale, (1, T)) for i, s in enumerate(SIG)}


def ref_a_shtrih(kom, okno=63, L=1.0, tsel=0.10):
    """Эталон: вес_k ∝ 1/σ_k по окну до первого дня месяца; s = min(L, цель/σ_портф(до d−1))."""
    S = np.stack([v[0] for v in kom.values()])
    st = pu._starty(DATY) + [T]
    A2 = np.zeros(T)
    for i, e in zip(st[:-1], st[1:]):
        h = S[:, max(0, i - okno):i]
        a = np.full(len(S), 1 / len(S)) if h.shape[1] < 20 else (1 / h.std(axis=1, ddof=1)) / (1 / h.std(axis=1, ddof=1)).sum()
        A2[i:e] = a @ S[:, i:e]
    out = A2.copy()
    for i in range(T):
        w = A2[max(0, i - okno):i]
        if len(w) >= 20 and w.std(ddof=1) > 0:
            out[i] = A2[i] * min(L, tsel / (w.std(ddof=1) * np.sqrt(252)))
    return out, A2


def vesa(kom):
    return pu._vesa_strategij(np.stack([v[0] for v in kom.values()]), DATY, "obr_vol_63", None, [0, 1, 2])


def chk_ves_do_d_minus_1():
    kom = synth()
    i = pu._starty(DATY)[10]
    a = vesa(kom)[:, i]
    S = np.stack([v[0] for v in kom.values()])
    sg = S[:, i - 63:i].std(axis=1, ddof=1)
    assert np.abs(a - (1 / sg) / (1 / sg).sum()).max() <= 1e-12 and a[0] > a[1] > a[2]
    S2 = {k: v.copy() for k, v in kom.items()}
    S2["S3"][0, i:] *= 50                            # будущее (день d и позже) не должно менять вес на d
    assert np.abs(vesa(S2)[:, i] - a).max() <= 1e-15


def chk_ves_okno_63():
    kom = synth()
    i = pu._starty(DATY)[10]
    S = np.stack([v[0] for v in kom.values()])
    sg = S[:, i - 63:i].std(axis=1, ddof=1)
    assert np.abs(vesa(kom)[:, i] - (1 / sg) / (1 / sg).sum()).max() <= 1e-12


def chk_portfel_raven_etalonu():
    kom = synth()
    got = pu.a_shtrih(kom, ZH090, DATY)
    ref, A2 = ref_a_shtrih(kom)
    assert np.abs(got - ref).max() <= 1e-12
    assert np.abs(ref[100:]).sum() < np.abs(A2[100:]).sum()         # таргет действительно снижает экспозицию


def chk_plecho_ne_bolshe_1():
    kom = synth(scale=0.1)                                           # σ_портф ≈ 1–3 % годовых: цель 10 % недостижима без плеча
    got = pu.a_shtrih(kom, {**ZH090, "L": 3}, DATY)                  # даже при L = 3 в конфиге плечо режется до 1
    _, A2 = ref_a_shtrih(kom, L=1.0)
    assert np.abs(got - A2).max() <= 1e-12 and np.all(np.abs(got) <= np.abs(A2) + 1e-18)


def chk_osnovnaya_zh090():
    assert pu.osnovnaya_konf() == {**ZH090, "osnovnaya": True, "rol": "osnovnaya"}
    cs = pu.konfiguracii()
    assert len(cs) == 1296 and not any(c["osnovnaya"] for c in cs)
    assert [c["rol"] for c in cs].count("ravnyj_bez_targeta") == 1
    assert all(c["ves"] != "obr_vol_63" for c in cs)                # в сетке основной нет: N = 1, выбирать не из чего


def chk_tablica_bez_vybora(tmp_path):
    kom = synth()
    sk = pu.setka()
    sk["uroven2"].update(ves=["ravnyj"], rho=["net"], lam=["net"], sostav=["vse6"])
    sk["uroven3"].update(tsel_vol=["net", 10], tormoz_mdd=["net"])
    orig = pu.konfiguracii
    pu.konfiguracii = lambda s=None: orig(sk)
    try:
        df = pu.tablica_1296(kom, DATY, None, None, out=tmp_path / "u.csv")
    finally:
        pu.konfiguracii = orig
    assert df.iloc[0]["status"] == "osnovnaya" and df.iloc[0]["ves"] == "obr_vol_63" and df.iloc[0]["tsel_vol"] == 10
    assert int(df["osnovnaya"].sum()) == 1 and list(df["status"]).count("ravnyj_bez_targeta") == 1
    ref, _ = ref_a_shtrih(kom)
    assert df.iloc[0]["sharpe"] == pytest.approx(pu.metriki(ref)["sharpe"], abs=1e-12)


def test_ves_obratnaya_volatilnost_do_d_minus_1():
    chk_ves_do_d_minus_1()


def test_ves_okno_63():
    chk_ves_okno_63()


def test_portfel_raven_etalonu_i_tarhet_snizhaet():
    chk_portfel_raven_etalonu()


def test_plecho_ne_bolshe_1():
    chk_plecho_ne_bolshe_1()


def test_osnovnaya_zh090_i_setka_spravochnaya():
    chk_osnovnaya_zh090()


def test_tablica_osnovnaya_pervoj_bez_vybora(tmp_path):
    chk_tablica_bez_vybora(tmp_path)


def test_ig4_ravnyj_bez_targeta_raven_A():
    """Справочная строка: A′(ravnyj, без таргета) = A (среднее по присутствующим парам) ≤ 1e-12."""
    r = np.random.default_rng(0)
    kom = {f"S{i + 1}": r.normal(0, 0.01, (3, T)) for i in range(3)}
    kom["S1"][0, :50] = np.nan
    ref = np.nanmean(np.stack(list(kom.values())).reshape(-1, T), axis=0)
    assert np.abs(pu.a_shtrih(kom, RAVNYJ, DATY) - ref).max() <= 1e-12
    rows = [c for c in pu.konfiguracii() if c["rol"] == "ravnyj_bez_targeta"]
    assert {k: rows[0][k] for k in RAVNYJ} == RAVNYJ


def lovitsya(chk, *a):
    with pytest.raises(AssertionError):
        chk(*a)


def test_mut_vesa_po_sigma_s_dnem_d(monkeypatch):
    monkeypatch.setattr(pu, "_istoriya", lambda x, i, w: x[..., max(0, i - w + 1):i + 1])
    lovitsya(chk_ves_do_d_minus_1)


def test_mut_plecho_bolshe_1_dopuskaetsya(monkeypatch):
    orig = pu._uroven3
    monkeypatch.setattr(pu, "_uroven3", lambda A2, ts, ok, tm, mn, mini=20, L=1.0: orig(A2, ts, ok, tm, mn, mini, 5.0))
    lovitsya(chk_plecho_ne_bolshe_1)


def test_mut_okno_126_vmesto_63(monkeypatch):
    orig3, origv = pu._uroven3, pu._vesa_strategij
    monkeypatch.setattr(pu, "_uroven3", lambda A2, ts, ok, tm, mn, mini=20, L=1.0: orig3(A2, ts, 126, tm, mn, mini, L))
    lovitsya(chk_portfel_raven_etalonu)
    monkeypatch.setattr(pu, "_vesa_strategij", lambda S, d, ves, p, act, mini=20: origv(S, d, "obr_vol_126", p, act, mini))
    lovitsya(chk_ves_okno_63)


def test_mut_osnovnaya_ravnyj_ves(monkeypatch):
    monkeypatch.setattr(pu, "osnovnaya_konf", lambda sk=None: {**RAVNYJ, "osnovnaya": True, "rol": "osnovnaya"})
    lovitsya(chk_osnovnaya_zh090)


def test_mut_vybor_iz_1295(monkeypatch, tmp_path):
    orig = pu.tablica_1296

    def vybor(*a, **k):
        df = orig(*a, **k)
        best = df.iloc[1:].sort_values("sharpe", ascending=False).iloc[0].copy()   # «лучшая» из справочных становится основной
        best["status"], best["osnovnaya"] = "osnovnaya", True
        df.iloc[0] = best
        return df
    monkeypatch.setattr(pu, "tablica_1296", vybor)
    lovitsya(chk_tablica_bez_vybora, tmp_path)
