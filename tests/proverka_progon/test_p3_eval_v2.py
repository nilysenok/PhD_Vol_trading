"""П-3 (проверяющий), лист Л-5: `EvaluatorV2` по контракту etap1 §1.1–1.4, §3 (IG-1…IG-4) и заданию Л-5. Пишется до кода.
Свой IG-2: 20 случайных точек на стратегию, seed 7 (кодер — 20261028), сравнение sha256 `excess_d` и полей v1.0;
значения метрик не печатаются (только булевы сравнения; запускать с --tb=no). Нужны данные <= 2019 (FASTBT_MAIN, FASTBT_BARS1M).
Правка IG-теста (28.10, П-3б): первая версия требовала равенства ключей v2 (ключи-годы int против str, tselevaya/comm_pct/
vyrozhdena переехали в mediana_kv/komissiya/flagi) — сверяются excess_d и значения полей v1.0, ключи v2 не сверяются.
A в `Evaluator` отсутствует (только S1–S6; якорь — режим точки, не стратегия A): 3 точки A не набираются, отступление.
Без данных тест красный, не пропущен. python3 -m pytest tests/proverka_progon/test_p3_eval_v2.py -q --tb=no"""
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from fastbt import bars5m, cache

_MAIN_BARS = cache.MAIN / "data" / "regen" / "D25" / "bars1m"       # 1-минутки лежат в основной копии (только чтение)
if not bars5m.SRC.exists() and _MAIN_BARS.exists():
    bars5m.SRC = _MAIN_BARS

PER, COMM, SEED, NPT ="train4", 0.0004, 7, 20
SIDS = [f"S{k}" for k in range(1, 7)]
NOVYE = dict(ed_lim="sigma", zhizn="den", lestnica="net", gap="vhod", t_vyhod="close", k_x="net", storona="obe", filtr="net")
GRID = Path("fastbt/designs/sweep")


@pytest.fixture(scope="module")
def evs():
    from fastbt import etap1
    from fastbt.etap1_eval import Evaluator
    from fastbt.etap1_eval_v2 import EvaluatorV2
    cache = {}

    def get(sid):
        if sid not in cache:
            inp = etap1.build_inputs(sid, period=PER)
            cache[sid] = (Evaluator(sid, inp, etap1.base_point(sid), COMM), EvaluatorV2(sid, inp, etap1.base_point(sid), COMM))
        return cache[sid]
    return get


def points(sid, n=NPT, seed=SEED):
    y = yaml.safe_load((GRID / f"zh047_{sid}.yaml").read_text(encoding="utf-8"))
    rng = random.Random(f"{seed}:{sid}")
    ob = {k: v["setka"] for k, v in y["obshchie"].items()}
    ob["sobytiya"] = ["net", "cbr"]                  # div/cbr_div в v1 — ne_schitana
    ob["rezhim"] = ["net", "imoex_ma200"]            # rank08 — только если не тормозит; достаточно двух
    sg = {k: v["setka"] for k, v in y["signal"].items()}
    return [{k: rng.choice(v) for k, v in {**sg, **ob}.items()} for _ in range(n)]


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b


PEREIMEN = ("v", "excess_d")                        # v2: версия 2.0 и ряд excess_d отдельным массивом; остальные поля v1.0 — как есть


def norm(x):
    """Ключи-годы v1.0 (int) в v2 — строки: сверяются значения, а не тип ключа."""
    return {str(k): norm(v) for k, v in x.items()} if isinstance(x, dict) else x


def flat(v1):
    return np.concatenate([np.asarray(v1["excess_d"][y], float) for y in sorted(v1["excess_d"])])


@pytest.mark.parametrize("sid", SIDS)
def test_umolchaniya_ig2_svoj_nabor_seed7(sid, evs):
    """IG-2: при умолчаниях новых осей EvaluatorV2 == Evaluator: sha256(excess_d) и все поля v1.0 (кроме `v`)."""
    e1, e2 = evs(sid)
    bad = []
    for i, pt in enumerate(points(sid)):
        a = e1.evaluate(pt, PER)
        row, ex_port, ex_tik = e2.evaluate_v2({**pt, **NOVYE}, PER)
        ok = sha(flat(a)) == sha(ex_port) and all(eq(norm(a[k]), norm(row[k])) for k in a if k not in PEREIMEN)
        ok = ok and ex_tik.shape[0] == 17 and ex_tik.dtype == np.float64
        if not ok:
            bad.append(i)
    assert bad == [], f"{sid}: расхождения в точках {bad}"


def test_umolchaniya_ig2_kratnost_120():
    assert sum(len(points(s)) for s in SIDS) == 120
    assert all(len({repr(sorted(p.items())) for p in points(s)}) > 10 for s in SIDS)


def test_tozhdestvo_ig3_nanmean_tikerov_ravno_portfelyu(evs):
    for sid in ("S1", "S3", "S5"):
        _, e2 = evs(sid)
        for pt in points(sid, 5):
            pt = {**e2.full(pt), "razmer": "ravnyy", **NOVYE}
            _, p, t = e2.evaluate_v2(pt, PER)
            assert np.nanmax(np.abs(np.nanmean(t, axis=0) - p)) <= 1e-12
            assert np.nanmax(np.abs(np.nanmean(t.astype(np.float32), axis=0) - p)) <= 1e-8


def test_mutaciya_ex_tikera_bez_rf_lovitsya(evs):
    """Если из ex тикера убрать `u·rf·cal`, тождество рвётся: портфель платит ставку на экспозицию."""
    _, e2 = evs("S1")
    _, p, t = e2.evaluate_v2({**e2.full({}), **NOVYE}, PER)
    assert np.nanmax(np.abs(np.nanmean(t, axis=0) - p)) <= 1e-12
    assert np.nanmax(np.abs(np.nanmean(t + 1e-4, axis=0) - p)) > 1e-12


def test_ig4_a_shtrih_pary_ravny_srednemu_portfelej(evs):
    """Слой пар: nanmean по 6×17 рядам якорей == среднее 6 портфелей (при cap = 1, равные доли). Сверка с боевой A — IG-4 Л-11."""
    ps, ts = [], []
    for sid in SIDS:
        _, e2 = evs(sid)
        _, p, t = e2.evaluate_v2({**e2.full({}), **NOVYE}, PER)
        ps.append(p), ts.append(t)
    assert np.nanmax(np.abs(np.nanmean(np.concatenate(ts), axis=0) - np.mean(ps, axis=0))) <= 1e-12


def spy_run(monkeypatch):
    from fastbt import exec_e
    calls, orig = [], exec_e.run

    def spy(ticker, days, sig, sigma, p, **kw):
        calls.append((ticker, np.asarray(sig).copy(), p))
        return orig(ticker, days, sig, sigma, p, **kw)
    monkeypatch.setattr(exec_e, "run", spy)
    return calls


def test_kx_idet_v_params_a_pri_b_none(evs, monkeypatch):
    _, e2 = evs("S1")
    calls = spy_run(monkeypatch)
    e2.evaluate_v2({**e2.full({}), **NOVYE, "k_x": 1.5, "k_s": "net", "R": "net", "tau": 5}, PER)
    assert calls and all(p.a == 1.5 and p.b is None for _, _, p in calls)
    calls.clear()
    e2.evaluate_v2({**e2.full({}), **NOVYE}, PER)
    assert all(p.a is None for _, _, p in calls)


def test_mutaciya_kx_pri_ks_ne_net_lovitsya(evs, monkeypatch):
    """canon_v2 п. 2: k_s ≠ net -> k_x = net; тейк задаёт R. Заданный k_x не должен менять Params.a при стопе."""
    _, e2 = evs("S1")
    calls = spy_run(monkeypatch)
    e2.evaluate_v2({**e2.full({}), **NOVYE, "k_x": 1.5, "k_s": 2.0, "R": 3.0}, PER)
    assert calls and all(p.a != 1.5 and p.b == 2.0 for _, _, p in calls)


def test_storona_long_net_shortov(evs, monkeypatch):
    _, e2 = evs("S1")
    calls = spy_run(monkeypatch)
    e2.evaluate_v2({**e2.full({}), **NOVYE, "storona": "long", "vhod": 0.5}, PER)
    assert calls and all((s >= 0).all() for _, s, _ in calls)      # мутация «long пропускает шорт» роняет это
    calls.clear()
    e2.evaluate_v2({**e2.full({}), **NOVYE, "vhod": 0.5}, PER)
    assert any((s < 0).any() for _, s, _ in calls)                 # у obe шорты есть: тест не тривиален


def test_lestnica_dve_poloviny_i_predel(evs, monkeypatch):
    """x -> ∞: вторая половина не исполняется, ряд тикера = ровно половина ряда без лестницы (линейность по весу)."""
    _, e2 = evs("S1")
    base = {**e2.full({}), **NOVYE, "vhod": 0.5}
    _, _, t0 = e2.evaluate_v2(base, PER)
    calls = spy_run(monkeypatch)
    _, _, t1 = e2.evaluate_v2({**base, "lestnica": 1e9}, PER)
    ks = {round(p.k, 6) for _, _, p in calls if p.k is not None}
    assert len(ks) == 2 and min(ks) == 0.5                          # два run: k и k + x
    assert np.nanmax(np.abs(t1 - 0.5 * t0)) <= 1e-9                  # вес лестницы не w/2 — падает (мутация)
    assert np.isnan(t1).tolist() == np.isnan(t0).tolist()


def test_zaem_tolko_na_shorty(evs):
    _, e2 = evs("S1")
    pt = {**e2.full({}), **NOVYE}
    rl, _, _ = e2.evaluate_v2({**pt, "storona": "long"}, PER)
    ro, _, _ = e2.evaluate_v2(pt, PER)
    for s in ("5", "10"):
        assert rl["by_quarter_zaem"][s].keys() == rl["by_quarter"].keys() and len(rl["by_quarter_zaem"][s]) == 16
        assert eq(rl["by_quarter_zaem"][s], rl["by_quarter"])       # заём на лонгах — мутация ловится здесь
    assert not eq(ro["by_quarter_zaem"]["10"], ro["by_quarter"])    # на шортах плата есть
    assert not eq(ro["by_quarter_zaem"]["10"], ro["by_quarter_zaem"]["5"])


def test_ne_schitana_pri_none(evs, monkeypatch):
    """filtr, вернувший None, -> status ne_schitana, метрики nan; мутация «None как net» даёт status ok."""
    from fastbt import filtr_e1_rynok
    _, e2 = evs("S1")
    monkeypatch.setattr(filtr_e1_rynok, "razresheno", lambda *a, **k: None)
    row, *_ = e2.evaluate_v2({**e2.full({}), **NOVYE, "filtr": "div"}, PER)
    assert row["status"] == "ne_schitana" and math.isnan(row["sharpe"])


def test_filtr_cbr_raven_sobytiya_cbr(evs):
    e1, e2 = evs("S1")
    a = e1.evaluate({**e1.full({}), "sobytiya": "cbr"}, PER)
    _, p, _ = e2.evaluate_v2({**e2.full({}), **NOVYE, "filtr": "cbr", "sobytiya": "net"}, PER)
    assert sha(flat(a)) == sha(p)


def test_vozm_sigma_mult_edinica_toždestvenna(evs):
    _, e2 = evs("S1")
    pt = {**e2.full({}), **NOVYE, "sigma": "rv22", "vhod": 0.5}
    _, p0, _ = e2.evaluate_v2(pt, PER)
    _, p1, _ = e2.evaluate_v2(pt, PER, {"sigma_mult": 1.0})
    _, p2, _ = e2.evaluate_v2(pt, PER, {"sigma_mult": 1.1})
    _, p3, _ = e2.evaluate_v2(pt, PER, {"komissiya": COMM})
    assert sha(p0) == sha(p1) == sha(p3) and sha(p0) != sha(p2)


def test_period_vne_train4_otkaz(evs):
    _, e2 = evs("S1")
    with pytest.raises(PermissionError):
        e2.evaluate_v2({**e2.full({}), **NOVYE}, "test")
