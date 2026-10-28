"""П-5 / Л-10: независимые тесты `fastbt/progon_ustojch.py` (контракт §1.4, §5 вопрос 2; задание Л-10). Синтетика.
Кода нет -> красные (ImportError), без skip/xfail (Ж-081). Ожидаемый API (фиксирует проверяющий):
  loto_portfel(tik[N,17,T], drop) -> [N,T]  nanmean по 16 оставшимся, знаменатель = число присутствующих на дату
  loyo_trenirovka(by_quarter_keys, god) -> list[str]  12 кварталов без года god
  peresemple_kvartaly(x, daty, seed) -> ndarray  ряд, склеенный из целых блоков-кварталов
  bootstrap_kvartaly(by_quarter, seed=20261028, B=10000) -> dict  mediana/dolya: {"p05","p50","p95"}
  vybor_popravki(vyigrysh, gody4) -> bool;  PORIG_VYIGRYSH = 0.2;  MIN_GODOV = 3
  dlya_kandidatov(rows) -> list;  dlya_A_shtrih(rows) -> list
python3 -m pytest tests/proverka_progon/test_p5_ustojch.py -q"""
import importlib

import numpy as np


def pu():
    return importlib.import_module("fastbt.progon_ustojch")


def _tik():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 17, 30)).astype(np.float64)
    x[0, 3, 5] = np.nan
    x[1, 0, :] = np.nan                       # тикер 0 вовсе без данных у точки 1
    x[2, 16, 10] = np.nan
    return x


def test_loto_znamenatel_chislo_prisutstvuyushchih():
    x = _tik()
    for drop in (0, 3, 16):
        keep = [i for i in range(17) if i != drop]
        ozh = np.nanmean(x[:, keep, :], axis=1)
        got = pu().loto_portfel(x, drop)
        assert got.shape == (4, 30)
        assert np.allclose(got, ozh, atol=1e-15, equal_nan=True), drop


def test_loto_ne_delit_na_16_i_ne_na_17_pri_nan():
    x = np.zeros((1, 17, 3))
    x[0, :, :] = 1.0
    x[0, 5, 1] = np.nan                       # на дату 1 присутствуют 16 -> без тикера 2 остаётся 15
    got = pu().loto_portfel(x, 2)
    assert np.allclose(got, 1.0)              # средние единиц = 1; деление на фиксированное 16 дало бы 15/16


def test_loto_tiker_edinstvennyj_na_date_daet_nan():
    x = np.full((1, 17, 2), np.nan)
    x[0, 4, 0] = 0.5
    x[0, 7, 1] = 0.1
    got = pu().loto_portfel(x, 4)
    assert np.isnan(got[0, 0]) and got[0, 1] == 0.1


def test_loto_ne_menyaet_vhod():
    x = _tik()
    c = x.copy()
    pu().loto_portfel(x, 5)
    assert np.array_equal(x, c, equal_nan=True)


def test_loyo_dvenadtsat_kvartalov_bez_goda():
    keys = [f"{y}Q{q}" for y in range(2016, 2020) for q in range(1, 5)]
    for y in range(2016, 2020):
        tr = pu().loyo_trenirovka(keys, y)
        assert len(tr) == 12 and not any(k.startswith(str(y)) for k in tr)
        assert set(tr) | {k for k in keys if k.startswith(str(y))} == set(keys)


def _daty():
    return np.arange(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))


def _kv(daty):
    m = daty.astype("datetime64[M]").astype(int)
    return (daty.astype("datetime64[Y]").astype(int) + 1970) * 10 + ((m % 12) // 3 + 1)


def test_bootstrap_blokami_kvartalami_a_ne_dnyami():
    daty, x = _daty(), np.arange(len(_daty()), dtype=float)
    kv = _kv(daty)
    bl = {k: x[kv == k] for k in np.unique(kv)}
    out = pu().peresemple_kvartaly(x, daty, seed=1)
    i, uzhe = 0, 0
    while i < len(out):                                    # режем на блоки: каждый обязан быть целым исходным кварталом
        for k, b in bl.items():
            if np.array_equal(out[i:i + len(b)], b):
                i += len(b)
                uzhe += 1
                break
        else:
            raise AssertionError(f"позиция {i}: не целый квартал (бутстрап по дням?)")
    assert uzhe == 16


def test_bootstrap_perestanovki_vozvratom_neskolko_raz_odin_kvartal():
    daty, x = _daty(), np.arange(len(_daty()), dtype=float)
    kv = _kv(daty)
    povtory = 0
    for s in range(30):
        out = pu().peresemple_kvartaly(x, daty, seed=s)
        povtory += len(out) != len(set(out.tolist()))
    assert povtory > 0                                      # выборка С возвращением


def test_bootstrap_seed_vosproizvodim_i_porjadok_kvantilej():
    bq = {f"{y}Q{q}": s for (y, q), s in zip(((y, q) for y in range(2016, 2020) for q in range(1, 5)),
                                             np.linspace(-1.0, 2.0, 16))}
    a = pu().bootstrap_kvartaly(bq, seed=20261028, B=500)
    b = pu().bootstrap_kvartaly(bq, seed=20261028, B=500)
    c = pu().bootstrap_kvartaly(bq, seed=1, B=500)
    assert a == b and a != c
    for k in ("mediana", "dolya"):
        assert a[k]["p05"] <= a[k]["p50"] <= a[k]["p95"]
    assert 0.0 <= a["dolya"]["p05"] and a["dolya"]["p95"] <= 1.0
    assert -1.0 <= a["mediana"]["p05"] and a["mediana"]["p95"] <= 2.0


def test_bootstrap_po_umolchaniyu_seed_i_B_po_zadaniyu():
    import inspect
    sg = inspect.signature(pu().bootstrap_kvartaly)
    assert sg.parameters["seed"].default == 20261028 and sg.parameters["B"].default == 10000


def test_popravka_2_iz_4_let_ne_beretsya():
    assert pu().vybor_popravki(0.25, [0.1, 0.1, -0.1, -0.1]) is False


def test_popravka_3_iz_4_beretsya_i_znak_vazhen():
    assert pu().vybor_popravki(0.25, [0.1, 0.1, 0.1, -0.1]) is True
    assert pu().vybor_popravki(0.25, [0.1, 0.1, 0.0, -0.1]) is False        # ноль — не «тот же знак»
    assert pu().vybor_popravki(-0.25, [-0.1, -0.1, -0.1, 0.1]) is False     # выигрыш — только положительный


def test_popravka_porog_nestrogij():
    assert pu().PORIG_VYIGRYSH == 0.2 and pu().MIN_GODOV == 3
    assert pu().vybor_popravki(0.2, [0.1] * 4) is True                       # >= 0,2
    assert pu().vybor_popravki(0.2 - 1e-9, [0.1] * 4) is False


def test_popravka_ne_v_kandidaty_i_ne_v_A_shtrih_no_v_N():
    osn = [{"klyuch_sha": "a", "istochnik": "sobol1", "status": "ok"}]
    tk = [{"klyuch_sha": "t", "istochnik": "tiker", "status": "ok"}]
    assert pu().dlya_kandidatov(osn + tk) == osn and pu().dlya_A_shtrih(osn + tk) == osn
    assert pu().dlya_kandidatov(tk) == []
