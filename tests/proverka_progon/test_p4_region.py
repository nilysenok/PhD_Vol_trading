"""П-4 (проверяющий), Л-7: select_fn региона на синтетике с известным ответом. Контракт: docs/zadaniya/2026-10-28_progon_L7_region.md.
Допущения о форме входа (контракт Chitatel §1.5): M — T × N; rows[i]["klyuch"] — токены осей (repr(float)), by_quarter, mediana_kv,
dolya_pol_kv; daty — datetime64[D]. Модуль progon_region импортируется внутри тестов: до кода каждый тест красный."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATY = pd.bdate_range("2016-01-01", "2019-12-31").values.astype("datetime64[D]")
QI = pd.PeriodIndex(DATY, freq="Q").asi8 - pd.Period("2016Q1").ordinal       # номер квартала 0..15 по дню
KEYS = [f"{y}Q{q}" for y in range(2016, 2020) for q in range(1, 5)]
SIG = ["s1", "s2", "s3", "s4"]


def sr(x):
    s = np.nanstd(x, ddof=1)
    return 0.0 if s < 1e-15 else float(np.nanmean(x) / s * np.sqrt(252))


def kub(nvals, nnoise, mu_fn, seed=1, sigma=0.01):
    """Полный факториал: 4 сигнальные оси по nvals значений + 1 шумовая по nnoise. mu_fn(idx: dict, q: int) -> дневной дрейф."""
    names = SIG + ["n1"]
    pts = list(itertools.product(*([range(nvals)] * 4 + [range(nnoise)])))
    mu = np.array([[mu_fn(dict(zip(names, p)), q) for q in range(16)] for p in pts])         # N × 16
    M = mu[:, QI].T + np.random.default_rng(seed).normal(0, sigma, (len(DATY), len(pts)))
    return M, rows_of(M, pts, names), pts, names


def rows_of(M, pts, names):
    rows = []
    for j, p in enumerate(pts):
        bq = {k: sr(M[QI == i, j]) for i, k in enumerate(KEYS)}
        v = np.array(list(bq.values()))
        rows.append(dict(status="ok", klyuch={a: repr(float(i)) for a, i in zip(names, p)}, by_quarter=bq,
                         mediana_kv=float(np.median(v)), dolya_pol_kv=float((v > 0).sum() / 16),
                         sdelok_y={str(y): 60 for y in range(2016, 2020)}, storony_y={str(y): 40 for y in range(2016, 2020)}))
    return rows


def plato(c, w=1):
    return lambda i, q: 0.0008 if all(abs(i[a] - c) <= w for a in SIG) else -0.0005


def test_region_vybiraet_plato_a_ne_pik():
    from fastbt import progon_region as R
    spike = (0, 0, 0, 0, 0)

    def mu(i, q):
        return 0.01 if tuple(i[a] for a in SIG + ["n1"]) == spike else plato(2)(i, q)

    M, rows, pts, names = kub(5, 2, mu)
    j_peak = int(np.argmax([r["mediana_kv"] for r in rows]))
    assert pts[j_peak] == spike, "синтетика: argmax обязан указывать на шумовой пик"
    res = R.vybor_region(M, rows, DATY)
    assert res is not None and j_peak not in set(res["chleny"]) and len(res["chleny"]) >= R.MIN_CHLENOV
    assert {a: c for a, c in zip(res["osi"], res["centr"])} == {a: 2 for a in SIG}          # центр плато (мутация argmax: 1 член, не плато)
    assert all(all(abs(pts[j][k] - 2) <= 1 for k in range(4)) for j in res["chleny"])


def test_ryad_regiona_srednee_a_ne_mediana():
    from fastbt import progon_region as R
    rng = np.random.default_rng(3)
    M = rng.normal(0, 1, (50, 5))
    M[:, 0] += 5                                        # асимметричный столбец: среднее и медиана по членам различаются
    r = R.ryad_regiona(M, [0, 1, 2, 3, 4])
    assert np.allclose(r, M.mean(axis=1)) and not np.allclose(r, np.median(M, axis=1))
    assert np.allclose(R.ryad_regiona(M, [1, 2]), M[:, [1, 2]].mean(axis=1))


def _polovina_kub(k_dop):
    """Плато положительно в k_dop кварталах из 16, в остальных сильно отрицательно; прочая сетка — минус."""
    def mu(i, q):
        base = plato(2)(i, q)
        return base if base < 0 or q < k_dop else -0.002
    return kub(5, 2, mu, seed=5)


def test_dolya_075_ne_dopustima_a_13_dopustima():
    from fastbt import progon_region as R
    M, rows, *_ = _polovina_kub(12)
    assert R.vybor_region(M, rows, DATY) is None, "доля + кварталов 12/16 = 0,75 < 0,8: допустимых регионов нет"
    M, rows, *_ = _polovina_kub(13)
    res = R.vybor_region(M, rows, DATY)
    assert res is not None and len(res["chleny"]) >= R.MIN_CHLENOV                          # 13/16 = 0,8125 ≥ 0,8


def test_nan_kvartal_ne_polozhitelnyj():
    """Мутация: nan-квартал ряда региона считается положительным. 12 положительных + 4 кварталов без дней (SR = nan) = 0,75
    -> не допустим. tsel="dolya": при "mediana" целевая с nan сама nan и мутацию не выдаёт."""
    from fastbt import progon_region as R
    M, rows, *_ = _polovina_kub(16)
    keep = QI < 12
    assert R.vybor_region(M[keep], rows, DATY[keep], tsel="dolya") is None


def test_osi_ranzhiruyutsya_tolko_na_perednyh_kvartalah():
    """A–D работают в кварталах 0–7 (убывая по силе), E,F (n1, s5-аналоги) — только в 8–15, сильнее. На половине осей E,F нет в топ-4."""
    from fastbt import progon_region as R
    names = ["a", "b", "c", "d", "e", "f"]
    pts = list(itertools.product(range(4), repeat=6))
    w = dict(a=4, b=3, c=2, d=1.5, e=0, f=0)
    w2 = dict(a=0, b=0, c=0, d=0, e=12, f=12)
    mu = np.array([[0.0006 + (sum(w[n] * p[k] for k, n in enumerate(names)) if q < 8 else
                    sum(w2[n] * p[k] for k, n in enumerate(names))) * 1e-4 / 3 for q in range(16)] for p in pts])
    M = mu[:, QI].T + np.random.default_rng(7).normal(0, 0.01, (len(DATY), len(pts)))
    rows = rows_of(M, pts, names)
    top_half = R.vybor_region(M, rows, DATY, kvartaly_idx=list(range(8)))["osi"]
    top_all = R.vybor_region(M, rows, DATY)["osi"]
    assert set(top_half) == {"a", "b", "c", "d"}                                            # мутация «все 16» вносит e, f
    assert {"e", "f"} <= set(top_all)


def test_pbo_regiona_cherez_pravilo_i_argmax_menyaet_pbo():
    """Ж-080: PBO региона = pbo_cscv_pravilo(select_fn_region); подстановка argmax внутри обязана менять PBO."""
    from fastbt import progon_region as R
    from fastbt.pbo_pravilo import pbo_cscv_pravilo, vybor_argmax

    def mu(i, q):
        if all(i[a] == 3 for a in SIG) and i["n1"] == 0:
            return 0.03 if q < 8 else -0.03                                                  # пик: знак по блокам CSCV
        return plato(1)(i, q)

    M, rows, pts, names = kub(4, 2, mu, seed=9)
    coords = [{a: repr(float(i)) for a, i in zip(names, p)} for p in pts]
    p_reg = pbo_cscv_pravilo(M.T, coords, R.select_fn_region(rows, DATY, coords), nblk=8, dates=DATY)
    p_arg = pbo_cscv_pravilo(M.T, coords, vybor_argmax, nblk=8, dates=DATY)
    assert p_reg <= 0.1, f"регион выбирает плато: PBO региона {p_reg}"
    assert p_arg - p_reg >= 0.1, f"мутация «argmax внутри» обязана менять PBO: {p_arg} vs {p_reg}"
