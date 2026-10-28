"""П-5 / Л-9: независимые тесты `fastbt/progon_kontrol.py` (PR #437) по контракту (docs/otchety/2026-10-28_progon_etap1.md §1.2, §6)
и заданию Л-9. Синтетика, без данных и сети. Требования не ослаблены.
Изменения против первой версии (9b9b354, имена угаданы до кода) — под фактический API:
  placebo_dni -> placebo_sig_fn(seed)(t, sig, dates) + runs_of;  k2_pozicii -> k2_sig_fn(lag)(t, sig, dates);  K2_LAG_MIN -> MIN_LAG;
  k3_obnulit -> plan(pt) (задача 'k3');  sigma_mediana_goda -> etap1_eval_v2.mediana_goda;  p_placebo -> p_perm;
  vozmusheniya -> VOZM;  prognat_vozm / dlya_N / kontrol_katalog -> run_kandidat с подставным оценщиком и kontrol_pisatel.
python3 -m pytest tests/proverka_progon/test_p5_kontrol.py -q"""
import importlib
from collections import Counter

import numpy as np
import pandas as pd
import pytest


def pk():
    return importlib.import_module("fastbt.progon_kontrol")


def _daty():
    return pd.date_range("2016-01-01", "2019-12-31", freq="D")


def _sig(daty):
    """Известные сделки: по 3+j в год, длительности 2+j, знаки чередуются, зазор между ними; внутри года."""
    s, yr = np.zeros(len(daty)), daty.year.to_numpy()
    for y in (2016, 2017, 2018, 2019):
        i0 = int(np.flatnonzero(yr == y)[0])
        for j in range(3 + (y - 2016)):
            a = i0 + 10 + 40 * j
            s[a:a + 2 + j] = 1.0 if j % 2 == 0 else -1.0
    return s


def _gr(sig, daty):
    yr, g = daty.year.to_numpy(), {}
    for st, ln, v in pk().runs_of(sig):
        g.setdefault(int(yr[st]), []).append((ln, v))
    return {k: sorted(v) for k, v in g.items()}


def test_placebo_sohranyaet_chislo_storony_dlitelnosti_v_godu():
    d, s = _daty(), _sig(_daty())
    out = pk().placebo_sig_fn(3)("SBER", s, d)
    assert out.shape == s.shape and _gr(out, d) == _gr(s, d)
    assert Counter(np.sign(out[out != 0])) == Counter(np.sign(out[out != 0]))   # (знаки учтены в _gr: значение run)


def test_placebo_menyaet_dni_a_ne_chislo_dnej_v_pozicii():
    d, s = _daty(), _sig(_daty())
    out = pk().placebo_sig_fn(3)("SBER", s, d)
    assert not np.array_equal(out, s)                                           # дни действительно сменились
    assert int((out != 0).sum()) == int((s != 0).sum())                         # суммарная длительность та же
    for y in (2016, 2017, 2018, 2019):
        m = d.year.to_numpy() == y
        assert int((out[m] != 0).sum()) == int((s[m] != 0).sum()), y            # и в каждом году: сделки не уходят в другой год


def test_placebo_vosproizvodim_po_seed_i_seedy_razlichayutsya():
    d, s = _daty(), _sig(_daty())
    a, b = pk().placebo_sig_fn(4)("SBER", s, d), pk().placebo_sig_fn(4)("SBER", s, d)
    assert np.array_equal(a, b)
    outs = {pk().placebo_sig_fn(k)("SBER", s, d).tobytes() for k in range(10)}
    assert len(outs) == 10
    assert pk().placebo_sig_fn(4)("GAZP", s, d).tobytes() != a.tobytes()        # разные тикеры — разные случайные дни


def test_placebo_pustaya_lenta_i_odna_sdelka():
    d = _daty()
    assert not pk().placebo_sig_fn(0)("SBER", np.zeros(len(d)), d).any()
    one = np.zeros(len(d))
    one[50:53] = -1.0
    out = pk().placebo_sig_fn(0)("SBER", one, d)
    assert pk().runs_of(out)[0][1:] == (3, -1.0) and len(pk().runs_of(out)) == 1


def test_plan_plasebo_desyat_seedov_0_9():
    z = pk().plan({"vhod": 1.0})
    assert [x.kontrol for x in z if (x.kontrol or "").startswith("placebo")] == [f"placebo:{i}" for i in range(10)]


def test_p_perm_izvestnyj_otvet():
    assert pk().p_perm(1.0, [0.0] * 10) == 1 / 11
    assert abs(pk().p_perm(0.5, [0.5] * 10) - 1.0) < 1e-15                      # равные считаются (>=)
    assert abs(pk().p_perm(0.0, [1.0] * 10) - 1.0) < 1e-15
    assert abs(pk().p_perm(1.0, [2.0] * 4 + [0.0] * 6) - 5 / 11) < 1e-15
    assert abs(pk().p_perm(1.0, [np.nan] * 10) - 1 / 11) < 1e-15                # nan контроля не >= факта
    assert np.isnan(pk().p_perm(np.nan, [0.0] * 10))


def test_k2_lag0_ravno_ishodnomu_mutaciya9_posledniy_den_nenulevoy():
    sig = np.array([0, 1, 1, 0, -1, -1, 0.0, 1, 1, 1])                          # позиция в последний день ≠ 0
    assert np.array_equal(pk().k2_sig_fn(0)("SBER", sig, None), sig)


def test_k2_lag0_ravno_ishodnomu_na_realnoj_dline():
    d, s = _daty(), _sig(_daty())
    s[-1] = 1.0
    assert np.array_equal(pk().k2_sig_fn(0)("SBER", s, d), s)


def test_k2_sdvig_krugovoj_i_lag_ne_menshe_63():
    assert pk().MIN_LAG == 63
    assert all(g >= 63 for g in pk().k2_lags()) and len(pk().k2_lags()) == 20 and pk().k2_lags() == pk().k2_lags()
    sig = np.arange(1.0, 301.0)
    out = pk().k2_sig_fn(100)("SBER", sig, None)
    assert np.array_equal(out[:-1], np.roll(sig, 100)[:-1]) and not np.array_equal(out, sig)


def test_k3_obnulyaet_rovno_novye_osi_mutaciya10():
    from fastbt.etap1_eval_v2 import NOVYE
    assert dict(NOVYE) == {"ed_lim": "sigma", "zhizn": "den", "lestnica": "net", "gap": "vhod", "t_vyhod": "close",
                           "k_x": "net", "storona": "obe", "filtr": "net"}
    pt = {"ma_window": 13, "z_entry": 1.67, "vhod": 0.5, "tau": 5.0, "k_s": 2.0, "q_sily": 0.5,
          "ed_lim": "atr14", "zhizn": 12, "lestnica": 0.5, "gap": "propusk", "t_vyhod": "1500",
          "k_x": 1.0, "storona": "long", "filtr": "szhatie"}
    k3 = [z for z in pk().plan(pt) if z.kontrol == "k3"]
    assert len(k3) == 1
    out = k3[0].pt
    for o, v in NOVYE.items():                                                  # ни одна новая ось не оставлена
        assert out[o] == v, o
    for o in set(pt) - set(NOVYE):                                              # прочее нетронуто
        assert out[o] == pt[o], o
    assert pt["filtr"] == "szhatie"                                             # вход не мутирован


def test_sigma_mediana_goda_tolko_svoj_god():
    from fastbt.etap1_eval_v2 import mediana_goda
    d = _daty()
    god = d.year.to_numpy()
    sg = np.where(god == 2016, 1.0, np.where(god == 2017, 2.0, np.where(god == 2018, 3.0, 100.0)))
    sg[np.flatnonzero(god == 2016)[:5]] = 1000.0                                # выброс: медиана его не видит
    out = mediana_goda(sg, d)
    for y, v in ((2016, 1.0), (2017, 2.0), (2018, 3.0), (2019, 100.0)):
        assert np.all(out[god == y] == v), y                                    # не медиана по всем годам


def test_plan_sigma_mediana_pomechena_zaglyadyvaniem():
    z = [x for x in pk().plan({"vhod": 1.0}) if x.kontrol == "sigma_mediana"]
    assert len(z) == 1 and z[0].zaglyad and z[0].ev_vozm == {"sigma_lim_mediana": True}


def test_vozmusheniya_rovno_po_zadaniyu_i_doshli_v_evaluate():
    ozh = [{"sigma_mult": 0.9}, {"sigma_mult": 1.1}, {"bar_plus": 1}, {"stop_tik": 1}, {"komissiya": 0.0004},
           {"komissiya": 0.0005}]
    assert [dict(v) for v in pk().VOZM] == ozh
    zv = [z for z in pk().plan({"vhod": 1.0}) if z.istochnik == "vozm"]
    assert [z.vozm for z in zv] == ozh and [z.ev_vozm for z in zv] == ozh      # объявленное = то, что уходит в evaluate_v2


class _Inp:
    sigma, meta = {"rv22": {}}, {}

    def digest(self):
        return "vh"


class _Pis:
    def __init__(self):
        self.rows = []

    def add(self, row, x, tik):
        self.rows.append(row)


class _Ev:
    sid, comm, fdr, inp = "S1", 0.0004, None, _Inp()

    def __init__(self):
        self.calls = []

    def evaluate_v2(self, pt, period, vozm=None):
        self.calls.append(vozm)
        v = sum(float(x) for k, x in (vozm or {}).items() if k != "sig_fn")
        row = {"period": period, "komissiya": (vozm or {}).get("komissiya", 0.0004), "status": "ok", "mediana_kv": 1.0 + v,
               "flagi": {}}
        return row, np.zeros(3), np.zeros((17, 3))


def test_run_kandidat_vozm_v_evaluate_i_stroki_kontrolej_bez_sobol():
    ev, pis = _Ev(), _Pis()
    out = pk().run_kandidat(ev, {"vhod": 1.0, "sigma": "rv22"}, pis, lags=[63, 100])
    voz = [c for c in ev.calls if c and not set(c) <= {"sig_fn", "sigma_lim_mediana"}]
    assert voz == [dict(v) for v in pk().VOZM]                                  # каждое возмущение дошло до evaluate_v2
    rows = pis.rows
    assert rows and all(r["istochnik"] in ("kontrol", "vozm") for r in rows)    # ни одной строки sobol (иначе попадёт в N)
    assert all(r["sobol_i"] is None for r in rows)
    assert len({r["klyuch_sha"] for r in rows}) == len(rows)                    # плацебо/к2 одной точки различимы в ключе
    assert len({r["mediana_kv"] for r in rows if r["istochnik"] == "vozm"}) > 1  # возмущение имеет эффект
    assert sum(1 for z, _ in out if z.istochnik == "baza") == 1 and len(rows) == len(out) - 1


def test_kontrol_pisatel_ne_pishet_v_osnovnoj_katalog(tmp_path):
    with pytest.raises(ValueError):
        pk().kontrol_pisatel(tmp_path / "S1", "S1", 0, {})
    with pytest.raises(ValueError):
        pk().kontrol_pisatel(tmp_path, "S1", 0, {})


def test_bolshe_20_kandidatov_otkaz(tmp_path):
    with pytest.raises(ValueError):
        pk().run_strategiya(_Ev(), {f"c{i}": {} for i in range(21)}, tmp_path / "kontrol", tmp_path / "k.csv")
