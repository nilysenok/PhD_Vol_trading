"""Скорость этапа 1 (Ж-083, лист «тестировщик»): быстрый слой `fastbt.fast_etap1.patched()` ≡ медленный путь
(`etap1_eval.Evaluator` — эталон) ПОБАЙТНО на train4; мутации обязаны ловиться.

Контракт: `fast_etap1.patched()` — контекст; внутри `Evaluator.evaluate/.excess` считают быстро, результат тот же.
Без ветки `skorost-etap1` тесты красные (без skip/xfail, Ж-081). Мутация (а)–(г): эталон считается медленным путём
с подложенной ошибкой (monkeypatch), быстрый — честный; сверка обязана разойтись, и сам тест проверяет, что мутант
действует. (а) лимит по касанию — mutations № 2; (б) стоп/тейк в баре — № 3; (в) выход в день входа (τ = 0);
(г) гэп по уровню, не по open — № 4; (д) ключ кэша быстрого слоя константа — № 7 (Д5). Данные: FASTBT_BARS1M, FASTBT_MAIN.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from fastbt import cache, etap1, exec_e_run

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "fastbt_equiv_etap1.py"
spec = importlib.util.spec_from_file_location("fastbt_equiv_etap1", SCRIPT)
eq = importlib.util.module_from_spec(spec)
sys.modules["fastbt_equiv_etap1"] = eq
spec.loader.exec_module(eq)

SIDS = ("S1", "S4")


def fast_patched():
    try:
        return eq.load_fast()[1]
    except SystemExit as e:
        pytest.fail(str(e))


@pytest.fixture(scope="module")
def inputs():
    return {s: etap1.build_inputs(s, period=eq.PERIOD) for s in SIDS}


def mk(inputs, sid, inp=None):
    return etap1.Evaluator(sid, inp or inputs[sid], etap1.base_point(sid))


def slow_run(inputs, sid, pts, inp=None):
    ev = mk(inputs, sid, inp)
    return [eq.run_point(ev, p) for p in pts]


def fast_run(inputs, sid, pts, inp=None):
    ev, patched = mk(inputs, sid, inp), fast_patched()
    with patched():
        return [eq.run_point(ev, p) for p in pts]


# --- мутации медленного пути (подкладываются monkeypatch'ем) ---
def m_touch(mp):                                     # (а) лимит исполняется по касанию: l ≤ L вместо l ≤ L − тик
    mp.setattr(exec_e_run, "limit_mask", lambda sd, h, l, L, tick: (l <= L) if sd > 0 else (h >= L))


def m_take_first(mp):                                # (б) в баре сначала тейк, потом стоп
    def scan(s, i, j, d, b0):
        if s.zak:
            return s._scan_close(i, j, d)
        sd, dyn = s.tr.side, s.p.k_tr is not None or s.p.be
        S, T = s._levels(j)
        for b in range(b0, len(d.c)):
            o, h, l = d.o[b], d.h[b], d.l[b]
            if T is not None:
                hit, px = exec_e_run.take_hit(sd, o, h, l, T, s.p.tick)
                if hit:
                    return s._exit(i, j, b, px, "take")
            if S is not None:
                hit, px = exec_e_run.stop_hit(sd, o, h, l, S, s.p.stop_touch)
                if hit:
                    return s._exit(i, j, b, px, s.sr)
            if dyn:
                s._after_bar(j, h, l)
                S, T = s._levels(j)
    mp.setattr(exec_e_run.Run, "_scan", scan)


def m_tau0(mp):                                      # (в) временной выход в день входа (τ = 0)
    orig = exec_e_run.Run.step

    def step(s, j, i):
        orig(s, j, i)
        if s.tr is not None and s.p.tau is not None and j == s.ej:
            s._leave(i, j, len(s.days[i].c), s.close_of(i), "vremya")
    mp.setattr(exec_e_run.Run, "step", step)


def m_gap(mp):                                       # (г) гэп через уровень — исполнение по уровню, не по open
    st, tk = exec_e_run.stop_hit, exec_e_run.take_hit
    mp.setattr(exec_e_run, "stop_hit", lambda sd, o, h, l, S, touch: (st(sd, o, h, l, S, touch)[0], S))
    mp.setattr(exec_e_run, "take_hit", lambda sd, o, h, l, T, tick: (tk(sd, o, h, l, T, tick)[0], T))
    mp.setattr(exec_e_run, "limit_px", lambda sd, o, L: L)


def P(**kw):                                         # точка: якорные общие оси + заданное (полная — через Evaluator.full)
    return {**etap1.YAKOR, "vyhod": "net", **kw}


# Точки, на которых мутант действует (подобраны по медленному пути; условие проверяется в тесте).
BARS = dict(proverka="bars5m")
POINTS = {
    "touch": [P(vhod=v, **BARS) for v in (0.25, 0.5, 0.75, 1.0)] + [P(vhod=0.5, k_s=2.0, **BARS)],
    "take_first": [P(k_s=k, R=r, **BARS) for k in (1.0, 1.5) for r in (1.0, 1.5)],
    "tau0": [P(tau=t, k_s=2.0, vhod=0.5, **BARS) for t in (1, 2)] + [P(tau=1), P(tau=3, vhod=0.5, **BARS)],
    "gap": [P(k_s=k, R=r, **BARS) for k in (1.0, 1.5) for r in ("net", 1.0)] + [P(vhod=0.5, **BARS)],
}
MUTS = dict(touch=m_touch, take_first=m_take_first, tau0=m_tau0, gap=m_gap)


def test_equiv_real_20_points(inputs):
    """20 случайных точек (по 10 на S1 и S4, seed фиксирован, в начале — обязательные классы) — расхождений нет."""
    bad = []
    for k, sid in enumerate(SIDS):
        pts = eq.sample_points(sid, 10, 20261028 + k)
        bad += eq.compare(slow_run(inputs, sid, pts), fast_run(inputs, sid, pts), pts, sid)
    assert not bad, f"расхождений {len(bad)}; первая: {bad[0]}"


wide, perturbed = eq.wide, eq.perturbed
INPUT_OF = dict(take_first=wide)                    # мутации, которым нужны раздутые бары; остальные — реальные


@pytest.mark.parametrize("name", sorted(MUTS))
@pytest.mark.parametrize("sid", SIDS)
def test_mutation_caught(name, sid, inputs, monkeypatch):
    pts, inp = POINTS[name], INPUT_OF.get(name, lambda x: x)(inputs[sid])
    clean = slow_run(inputs, sid, pts, inp)
    assert not eq.compare(clean, fast_run(inputs, sid, pts, inp), pts, sid), "быстрый слой расходится и без мутации"
    with monkeypatch.context() as mp:
        MUTS[name](mp)
        mutant = slow_run(inputs, sid, pts, inp)
    assert eq.compare(clean, mutant, pts, sid), f"мутация {name} не меняет медленный путь на точках {sid}: тест слеп"
    caught = eq.compare(mutant, fast_run(inputs, sid, pts, inp), pts, sid)
    assert caught, f"мутация {name}: сверка не заметила расхождения"


def test_equiv_wide_bars_same_bar_events(inputs):
    """Стоп, тейк, лимит и гэп в одном баре: быстрый ≡ медленный побайтно на раздутых барах."""
    pts = [p for v in POINTS.values() for p in v]
    inp = wide(inputs["S1"])
    bad = eq.compare(slow_run(inputs, "S1", pts, inp), fast_run(inputs, "S1", pts, inp), pts, "S1")
    assert not bad, f"расхождений {len(bad)}; первая: {bad[0]}"


# --- (д) Д5: кэш сигналов / σ обязан менять ключ при подмене входа ---
FAST_MODS = ("fast_etap1", "fast_exec", "fast_signals", "fast_sigma")
KEYPT = [P(sigma="rv22", vhod=0.5, k_s=2.0, R=1.5, **BARS), P(sigma="rv5", k_s=1.5, tau=3, **BARS)]


def isolate_caches(mp, tmp_path):
    """Диск и память кэшей быстрого слоя — в пустое место (мутация ключа не должна травить настоящий кэш)."""
    mp.setenv("FASTBT_FS_CACHE", str(tmp_path / "fs"))
    mp.setenv("FASTBT_CACHE", str(tmp_path / "c"))
    mp.setattr(cache, "CACHE", tmp_path / "c")
    for m in fast_mods():
        for k, v in list(vars(m).items()):
            if isinstance(v, dict) and ("mem" in k.lower() or "cache" in k.lower()):
                mp.setattr(m, k, {})


def fast_mods():
    return [importlib.import_module(f"fastbt.{n}") for n in FAST_MODS if importlib.util.find_spec(f"fastbt.{n}")]


def freeze_keys(mp):
    """Ключи быстрого слоя → константа (кэш не замечает подмены входа): свои функции `*key*` и cache.key, когда его зовёт
    модуль быстрого слоя (прочие вызовы — как есть, чтобы не портить хэш входов `vhody` у медленного пути)."""
    orig, names = cache.key, {m.__name__ for m in fast_mods()}
    for m in fast_mods():
        for k, v in list(vars(m).items()):
            if "key" in k.lower() and callable(v) and getattr(v, "__module__", None) == m.__name__:
                mp.setattr(m, k, lambda *a, **kw: "K")
    mp.setattr(cache, "key", lambda *a, **kw: "K" if sys._getframe(1).f_globals["__name__"] in names else orig(*a, **kw))


def test_cache_key_tracks_inputs(inputs, tmp_path, monkeypatch):
    """Д5: после подмены σ и баров быстрый слой (с тем же кэшем) ≡ медленный на новых входах; сами входы дают разный результат."""
    sid, a = "S1", inputs["S1"]
    b = perturbed(a)
    isolate_caches(monkeypatch, tmp_path)
    slow_a, slow_b = slow_run(inputs, sid, KEYPT, a), slow_run(inputs, sid, KEYPT, b)
    assert eq.compare(slow_a, slow_b, KEYPT, sid), "подмена входов не меняет результат: тест слеп"
    fa, fb = fast_run(inputs, sid, KEYPT, a), fast_run(inputs, sid, KEYPT, b)
    assert not eq.compare(slow_a, fa, KEYPT, sid) and not eq.compare(slow_b, fb, KEYPT, sid)


def test_mutation_cache_key_frozen_caught(inputs, tmp_path, monkeypatch):
    """Мутация (д): ключи кэша быстрого слоя — константа → после подмены входов отдаётся устаревшее, сверка ловит."""
    sid, a = "S1", inputs["S1"]
    b = perturbed(a)
    isolate_caches(monkeypatch, tmp_path)
    slow_b = slow_run(inputs, sid, KEYPT, b)
    with monkeypatch.context() as mp:
        freeze_keys(mp)
        fast_run(inputs, sid, KEYPT, a)                         # кэш заполнен входом a
        stale = fast_run(inputs, sid, KEYPT, b)                 # вход b при том же «ключе»
    assert eq.compare(slow_b, stale, KEYPT, sid), "кэш с постоянным ключом не разошёлся с медленным путём"
