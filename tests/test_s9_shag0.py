"""Шаг 0 (v3): fastbt.shag0.build на синтетическом движке. Верный build = независимый счёт через Engine.port;
каждая подложенная ошибка (не тот Sk, к3 с невыключенной S5, LOO без комиссии, сдвиг года, нет комиссии
выхода на границе) делает таблицу неравной эталону — то есть тест на эталон её ловит."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_s9_kontrol import _engine  # noqa: E402

from fastbt import fast_kontrol as fk  # noqa: E402
from fastbt import s9, shag0  # noqa: E402
from fastbt.msp import sharpe  # noqa: E402

COMM = 0.0004
SIDS = shag0.SIDS


def _world(seed=3):
    ns = _engine(Ds=(1000, 1000), drift=0.003, seed=seed)
    ns.pairs = [(f"T{i // 6}", SIDS[i % 6]) for i in range(12)]
    ns.tickers = ["T0", "T1"]
    ns.r["tick"] = ns.r["pair"] // 6
    ns.port = lambda pos, comm=None, cap=1.0, wgt=None: s9.Engine.port(ns, pos, comm=comm, cap=cap, wgt=wgt)
    return ns


def _sh(ns, pos):
    return sharpe(s9.Engine.port(ns, pos, comm=COMM)["excess"].values)


def _ref_pos(ns, drop):
    """Эталонная маска «руками»: обнулить пары с данным sid (по номерам пар синтетики)."""
    out = ns.A_row.copy()
    for pid, (_, s) in enumerate(ns.pairs):
        if s in drop:
            out[ns.r["pair"] == pid] = 0.0
    return out


def _vals(res):
    """{метка: Шарп} для A, к3, LOO и по годам A."""
    d = {r["row"]: r["sharpe"] for r in res["k3"] + res["loo"]}
    d.update({f"A {r['god']}": r["sharpe"] for r in res["godam"] if r["row"] == "A"})
    return d


def _ref(ns):
    d = {"A": _sh(ns, ns.A_row), "к3: A без S5/S6": _sh(ns, _ref_pos(ns, ("S5", "S6")))}
    for s in SIDS:
        d[f"A без {s}"] = _sh(ns, _ref_pos(ns, (s,)))
        d[f"A только {s}"] = _sh(ns, ns.A_row - _ref_pos(ns, (s,)))
    p = s9.Engine.port(ns, ns.A_row, comm=COMM)
    for y in shag0.YEARS:
        d[f"A {y}"] = sharpe(p["excess"].values[p.index.year == y])
    return d


def _same(got, ref):
    return all(k in got and np.isclose(got[k], v, rtol=0, atol=1e-10) for k, v in ref.items())


def test_build_equals_independent_reference():
    ns = _world()
    got = _vals(shag0.build(ns, COMM, reps=50))
    assert _same(got, _ref(ns))
    assert len({round(v, 6) for v in got.values()}) > 10          # ряды различаются: тест не пустой


def test_mutation_wrong_sk_zeroed(monkeypatch):
    ns = _world()
    orig = shag0.zero_sids
    monkeypatch.setattr(shag0, "zero_sids", lambda pos, sidr, sids: orig(pos, sidr, ["S1" if sids[0] != "S1" else "S2"])
                        if len(sids) == 1 else orig(pos, sidr, sids))
    assert not _same(_vals(shag0.build(ns, COMM, reps=50)), _ref(ns))


def test_mutation_k3_with_s5_left(monkeypatch):
    ns = _world()
    monkeypatch.setattr(shag0, "k3_sids", lambda: ("S6",))
    k3_row = shag0.build(ns, COMM, reps=50)["k3"][1]
    assert not np.isclose(k3_row["sharpe"], _ref(ns)["к3: A без S5/S6"], rtol=0, atol=1e-10)
    assert k3_row["isklyucheno"] == "S6"                             # и список исключённого в таблице это показывает


def test_mutation_loo_without_commission():
    ns = _world()
    real = ns.port
    ns.port = lambda pos, comm=None, cap=1.0, wgt=None: real(
        pos, comm=0.0 if (wgt is None and not np.array_equal(pos, ns.A_row) and np.any(pos == 0)) else comm, cap=cap, wgt=wgt)
    got = _vals(shag0.build(ns, COMM, reps=50))
    assert not _same(got, _ref(ns))
    assert np.isclose(got["A"], _ref(ns)["A"], atol=1e-10)           # A цела — ошибка именно в LOO


def test_mutation_year_shift(monkeypatch):
    ns = _world()
    monkeypatch.setattr(shag0, "year_mask", lambda p, y: p.index.year == y + 1)
    assert not _same(_vals(shag0.build(ns, COMM, reps=50)), _ref(ns))


def test_mutation_exit_commission_dropped():
    ns = _world()
    real = ns.port

    def no_exit(pos, comm=None, cap=1.0, wgt=None):
        if wgt is not None:
            return real(pos, comm=comm, cap=cap, wgt=wgt)
        comm = COMM if comm is None else comm
        dp = np.diff(pos, prepend=0.0)
        dp[ns.first] = pos[ns.first]
        di, nd = ns.r["date"], len(ns.dates)
        sides = np.abs(dp)                                           # нет |pos| на последней строке пары
        r = np.bincount(di, pos * ns.lr_row - sides * comm, nd) / ns.cnt
        return ns._finish(r, np.bincount(di, pos * ns.lr_row, nd) / ns.cnt,
                          np.bincount(di, np.abs(pos), nd) / ns.cnt, np.bincount(di, sides, nd) / ns.cnt, cap)
    ns.port = no_exit
    assert not _same(_vals(shag0.build(ns, COMM, reps=50)), _ref(ns))


def test_k3_is_config_flag(tmp_path):
    assert fk.RANGE_SIDS == fk.k3_sids() == ("S5", "S6")
    y = tmp_path / "k3.yaml"
    y.write_text("isklyuchit_semyi: [S5]\n", encoding="utf-8")
    assert fk.k3_sids(y) == ("S5",)
    ns = _world()
    pos = shag0.zero_sids(ns.A_row, shag0.sid_row(ns), fk.k3_sids(y))
    assert np.any(pos[ns.r["pair"] == 5] != 0) and np.all(pos[ns.r["pair"] == 4] == 0)      # S6 жива, S5 снята
    assert np.array_equal(shag0.zero_sids(ns.A_row, shag0.sid_row(ns), fk.k3_sids()),
                          fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs))


def test_scale_half_keeps_sharpe_and_semyi_sums():
    ns = _world()
    res = shag0.build(ns, COMM, reps=50)
    sp = {r["row"]: r for r in res["spravochno"]}
    assert abs(sp["A × 0,5 (масштаб позиций)"]["sharpe"] - sp["A"]["sharpe"]) < 1e-9
    assert abs(sp["A × 0,5 (масштаб позиций)"]["expo_pct"] * 2 - sp["A"]["expo_pct"]) < 1e-9
    sm = shag0.semyi(ns)
    ov = [r["tikerdni_pct"] for r in sm if r["rezhim"] == "пересечение" and r["semya"].startswith("открыто семей")]
    assert abs(sum(ov) - 100) < 1e-9 and len(res["tikeram"]) == 2


@pytest.mark.parametrize("pos_scale", [0.0])
def test_empty_a_gives_nan_not_crash(pos_scale):
    p = s9.Engine.port(_world(), np.zeros(12000), comm=COMM)
    assert np.isfinite(shag0.stats(p, np.ones(len(p), bool))["expo_pct"])


def _bad_s3_world():
    """Семья S3 заведомо убыточна: её выключение улучшает A, значит вклад S3 = Шарп A − Шарп «A без S3» < 0."""
    ns = _world()
    s3 = shag0.sid_row(ns) == "S3"
    ns.lr_row = np.where(s3, -np.abs(ns.lr_row) - 0.01, ns.lr_row)
    return ns


def _vklad_s3(mod, ns):
    res = mod.build(ns, COMM, reps=50)
    return next(r["vklad_sharpe"] for r in res["loo"] if r["row"] == "A без S3")


def test_vklad_sharpe_sign_negative_for_harmful_family():
    ns = _bad_s3_world()
    sh_a, sh_wo = _sh(ns, ns.A_row), _sh(ns, _ref_pos(ns, ("S3",)))
    assert sh_wo > sh_a                                              # синтетика: без S3 лучше
    assert _vklad_s3(shag0, ns) == pytest.approx(sh_a - sh_wo, abs=1e-10) and _vklad_s3(shag0, ns) < 0


def test_mutation_vklad_sharpe_sign_flipped(monkeypatch):
    """Мутация в памяти: знак вклада наоборот → тест знака ловит (файл не тронут)."""
    import types
    src = Path(shag0.__file__).read_text(encoding="utf-8")
    old = 'wo["vklad_sharpe"] = sh_a - wo["sharpe"]'
    assert src.count(old) == 1
    mod = types.ModuleType("fastbt.shag0")
    mod.__file__, mod.__package__ = shag0.__file__, "fastbt"
    monkeypatch.setitem(sys.modules, "fastbt.shag0", mod)
    exec(compile(src.replace(old, 'wo["vklad_sharpe"] = wo["sharpe"] - sh_a'), shag0.__file__, "exec"), mod.__dict__)
    ns = _bad_s3_world()
    assert _vklad_s3(mod, ns) > 0                                    # мутант даёт неверный знак
    assert _vklad_s3(shag0, ns) < 0                                  # оригинал — верный
