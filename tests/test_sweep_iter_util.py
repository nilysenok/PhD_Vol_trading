"""Общие помощники тестов режимов Э1-в (PR #318): синтетическая сетка и запуск `run_mode` на SynthEvaluator во временном каталоге.

Сетка: a (сигнал), w, z — безусловные; mode — переключатель; cb.q — условный блок (активен при mode=on). Реальных данных нет.
"""
from fastbt import sweep_run_iter as it
from fastbt.sweep_synth import SynthEvaluator


def par(name, ish, grid):
    return {"imya": name, "ishodnoe": ish, "setka": list(grid)}


def setka(a=(3, 5, 8, 12, 20), a0=8, w=(0.5, 1.0, 1.5, 2.0), z=(1, 2, 3, 4), mode=("on", "off"), mode0="on",
          q=(1, 2, 3), period=None):
    s = {"obekt": {"sid": "S1"},
         "bloki": {"sig": {"parametry": [par("a", a0, a), par("w", 1.0, w), par("z", 2, z), par("mode", mode0, mode)]},
                   "cb": {"uslovie": {"mode": "on"}, "parametry": [par("q", 1, q)]}}}
    if period:
        s["period"] = period
    return s


def run(rezhim, out, name="S1", params="a,w,z", trials=12, paths=4, n_jobs=1, **kw):
    o = dict(params=params if rezhim != "optuna" else None, paths=paths, trials=trials, n_jobs=n_jobs)
    return it.run_mode(name, setka(**kw), rezhim, o, out, evaluator=SynthEvaluator())


def keys(res, c=0.04):
    return {r["_pk"] for r in res["done"][c]}
