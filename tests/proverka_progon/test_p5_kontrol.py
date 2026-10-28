"""П-5 / Л-9: независимые тесты `fastbt/progon_kontrol.py` по контракту (docs/otchety/2026-10-28_progon_etap1.md, §1.2, §6)
и заданию Л-9. Синтетика, без данных и сети. Кода на момент написания нет -> тесты красные (ImportError), без skip/xfail (Ж-081).
Ожидаемый API (проверяющий фиксирует его здесь; расхождение имён — вопрос к автору Л-9, а не повод ослаблять тест):
  placebo_dni(sdelki, daty, seed) -> list;  sdelki: [(tiker, den_vhoda, storona, dlit)], den_vhoda — индекс в daty
  k2_pozicii(pos, lag) -> ndarray;  K2_LAG_MIN = 63;  k3_obnulit(pt) -> dict;  NOVYE_OSI: dict ось -> умолчание
  sigma_mediana_goda(sigma, daty) -> ndarray;  p_placebo(fakt, placebo) -> float
  vozmusheniya() -> list[dict];  prognat_vozm(evaluate_v2, pt, period) -> list[(vozm, rezultat)]
  dlya_N(rows), dlya_pbo(rows) -> list;  kontrol_katalog(root: Path, sid) -> Path
python3 -m pytest tests/proverka_progon/test_p5_kontrol.py -q"""
import importlib
from collections import Counter
from pathlib import Path

import numpy as np


def pk():
    return importlib.import_module("fastbt.progon_kontrol")


def _daty():
    return np.arange(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))


def _sdelki(daty):
    """Известные сделки: 2 тикера x 4 года, число/стороны/длительности разные по тикеро-годам."""
    god = daty.astype("datetime64[Y]").astype(int) + 1970
    out = []
    for ti, t in enumerate(("SBER", "GAZP")):
        for y in (2016, 2017, 2018, 2019):
            idx = np.flatnonzero(god == y)
            for j in range(3 + ti + (y - 2016)):
                out.append((t, int(idx[10 + 25 * j]), 1 if (j + ti) % 2 == 0 else -1, 2 + j))
    return out


def _gr(daty, sdelki):
    god = daty.astype("datetime64[Y]").astype(int) + 1970
    g = {}
    for t, d, s, n in sdelki:
        g.setdefault((t, int(god[d])), []).append((s, n))
    return {k: sorted(v) for k, v in g.items()}


def test_placebo_sohranyaet_chislo_storony_dlitelnosti():
    daty, sd = _daty(), _sdelki(_daty())
    pl = pk().placebo_dni(sd, daty, seed=3)
    assert len(pl) == len(sd)
    assert _gr(daty, pl) == _gr(daty, sd)          # число, стороны, длительности в каждом тикеро-году
    assert [x[0] for x in pl] == [x[0] for x in sd] or Counter(x[0] for x in pl) == Counter(x[0] for x in sd)


def test_placebo_menyaet_tolko_dni_i_ostayotsya_v_godu():
    daty, sd = _daty(), _sdelki(_daty())
    pl = pk().placebo_dni(sd, daty, seed=3)
    assert sorted(x[1] for x in pl) != sorted(x[1] for x in sd)    # дни действительно сменились
    assert all(0 <= x[1] < len(daty) for x in pl)


def test_placebo_vosproizvodim_po_seed_i_seedy_razlichayutsya():
    daty, sd = _daty(), _sdelki(_daty())
    a, b, c = (pk().placebo_dni(sd, daty, seed=s) for s in (4, 4, 5))
    assert a == b and a != c
    assert len({tuple(pk().placebo_dni(sd, daty, seed=s)) for s in range(10)}) == 10


def test_placebo_pustaya_lenta_i_odna_sdelka():
    daty = _daty()
    assert pk().placebo_dni([], daty, seed=0) == []
    one = [("SBER", 40, -1, 3)]
    out = pk().placebo_dni(one, daty, seed=0)
    assert len(out) == 1 and out[0][2:] == (-1, 3)


def test_p_placebo_izvestnyj_otvet():
    assert pk().p_placebo(1.0, [0.0] * 10) == 1 / 11
    assert abs(pk().p_placebo(0.5, [0.5] * 10) - 1.0) < 1e-15           # >= : равные считаются
    assert abs(pk().p_placebo(0.0, [1.0] * 10) - 1.0) < 1e-15
    assert abs(pk().p_placebo(1.0, [2.0] * 4 + [0.0] * 6) - 5 / 11) < 1e-15


def test_k2_lag0_ravno_ishodnomu_mutaciya9():
    pos = np.array([0, 1, 1, 0, -1, -1, 0.0, 1])
    assert np.array_equal(pk().k2_pozicii(pos, 0), pos)
    sh = pk().k2_pozicii(pos, 3)
    assert np.array_equal(sh, np.roll(pos, 3)) and not np.array_equal(sh, pos)


def test_k2_lag_ne_menshe_63_i_sdvig_krugovoj():
    assert pk().K2_LAG_MIN == 63
    pos = np.arange(200.0)
    assert sorted(pk().k2_pozicii(pos, 64)) == sorted(pos)             # круговой: ничего не теряется


NOVYE = {"ed_lim": "sigma", "zhizn": "den", "lestnica": "net", "gap": "vhod", "t_vyhod": "close",
         "k_x": "net", "storona": "obe", "filtr": "net"}


def test_novye_osi_umolchaniya_po_kontraktu():
    assert dict(pk().NOVYE_OSI) == NOVYE


def test_k3_obnulyaet_rovno_novye_osi_mutaciya10():
    pt = {"ma_window": 13, "z_entry": 1.67, "vhod": 0.5, "tau": 5.0, "k_s": 2.0, "q_sily": 0.5,
          "ed_lim": "atr14", "zhizn": 12, "lestnica": 0.5, "gap": "propusk", "t_vyhod": "1500",
          "k_x": 1.0, "storona": "long", "filtr": "szhatie"}
    out = pk().k3_obnulit(pt)
    for o, v in NOVYE.items():                                          # все новые оси = умолчанию (ни одна не оставлена)
        assert out[o] == v, o
    for o in set(pt) - set(NOVYE):                                      # прочее нетронуто
        assert out[o] == pt[o], o
    assert pt["filtr"] == "szhatie"                                     # вход не мутирован


def test_k3_na_tochke_s_umolchaniyami_tozhdestvenno():
    pt = {"ma_window": 13, "vhod": 1.0, **NOVYE}
    assert pk().k3_obnulit(pt) == pt


def test_sigma_mediana_goda_tolko_svoj_god():
    daty = _daty()
    god = daty.astype("datetime64[Y]").astype(int) + 1970
    sg = np.where(god == 2016, 1.0, np.where(god == 2017, 2.0, np.where(god == 2018, 3.0, 100.0)))
    sg[np.flatnonzero(god == 2016)[:5]] = 1000.0                        # выброс: медиана его не видит
    out = pk().sigma_mediana_goda(sg, daty)
    assert out.shape == sg.shape
    for y, v in ((2016, 1.0), (2017, 2.0), (2018, 3.0), (2019, 100.0)):
        assert np.all(out[god == y] == v), y                            # не медиана по всем годам


def test_vozmusheniya_rovno_po_zadaniyu():
    v = pk().vozmusheniya()
    ozh = [{"sigma_mult": 0.9}, {"sigma_mult": 1.1}, {"bar_plus": 1}, {"stop_tik": 1}, {"komissiya": 0.0004},
           {"komissiya": 0.0005}]
    assert len(v) == 6 and all(x in v for x in ozh)


def test_vozm_dohodit_do_evaluate_i_menyaet_rezultat():
    seen = []

    def fake(pt, period, vozm=None):                                    # подставной evaluate_v2 с откликом на vozm
        seen.append(vozm)
        return 1.0 + sum(float(x) for x in (vozm or {}).values())

    res = pk().prognat_vozm(fake, {"vhod": 1.0}, "train4")
    assert [r[0] for r in res] == pk().vozmusheniya() and seen == pk().vozmusheniya()
    assert len({r[1] for r in res}) > 1                                 # возмущение имеет эффект, а не тихо отброшено


def test_kontroli_ne_v_N_i_ne_v_pbo():
    osn = [{"klyuch_sha": f"a{i}", "istochnik": "sobol1", "kontrol": None, "vozm": None, "status": "ok"} for i in range(3)]
    kon = [{"klyuch_sha": "k1", "istochnik": "kontrol", "kontrol": "placebo:3", "vozm": None, "status": "ok"},
           {"klyuch_sha": "k2", "istochnik": "kontrol", "kontrol": "k2:lag63", "vozm": None, "status": "ok"},
           {"klyuch_sha": "v1", "istochnik": "vozm", "kontrol": None, "vozm": {"bar_plus": 1}, "status": "ok"}]
    assert pk().dlya_N(osn + kon) == osn and pk().dlya_pbo(osn + kon) == osn
    assert pk().dlya_N(kon) == [] and pk().dlya_pbo([]) == []


def test_kontrol_katalog_otdelnyj_podkatalog():
    root = Path("/x/_ledger_v2/S1")
    k = pk().kontrol_katalog(root, "S1")
    assert k != root and k.name == "kontrol" and root in k.parents
