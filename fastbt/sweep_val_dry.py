"""`--dry` для validation A′: синтетическая панель, журнал «да» и каталоги — во временной папке. Реальных данных не читает."""
from __future__ import annotations

from pathlib import Path

from fastbt import sweep_val as sv
from fastbt.sweep_a import SynthPanelSource, make_sweep
from fastbt.sweep_a_core import VESA
from fastbt.sweep_val_stat import Cand

NET = {"vesa": "ravnye", "limit_pozitsiy": "net", "limit_na_strategiyu": "net"}


def make_meta(drifts=(0.0004, 0.0, 0.0008)) -> list:
    return [dict(id=f"S1:{i}", kind="S", params=NET, orig_params=NET, seed=7, drift=d, neighbors=8)
            for i, d in enumerate(drifts, 1)] + [
        dict(id="A:1", kind="A", params={**NET, "vesa": VESA[1], "limit_pozitsiy": 5}, orig_params=NET, seed=7, drift=0.0006, neighbors=8)]


def make_factory(tmp: Path, on_build=None):
    def factory(meta: list, note) -> list:
        if on_build:
            on_build()                       # тесты: здесь «читаются данные»
        out = []
        for m in meta:
            d = tmp / "sw" / m["id"].replace(":", "_")
            sw = make_sweep(SynthPanelSource(3, ("S1", "S2"), m["seed"], m["drift"]), str(d / "c"), note)
            ow = make_sweep(SynthPanelSource(3, ("S1", "S2"), m["seed"], 0.0), str(d / "o"), note)
            for lim in ("net", 3, 5, 8):     # пробы setup 2018 при 0,04: ledger для DSR и PBO
                sw.evaluate({**NET, "limit_pozitsiy": lim}, "setup")
            out.append(Cand(m["id"], m["kind"], sw, m["params"], ow, m["orig_params"], m["neighbors"],
                            sw.trials_sr(), sw.excess))
        return out
    return factory


def make_spec(tmp: Path, meta: list, note=None) -> tuple:
    """(Puti, spec) с журналом: строка «да» с полным sha, позже заморозки."""
    P = sv.Puti(koren=tmp)
    P.zhurnal.parent.mkdir(parents=True, exist_ok=True)
    stroka = f"да, открыть validation A′, sha {sv.sha_kandidatov(meta)}"
    P.zhurnal.write_text(f"### Ж-099 · 28.10.2026, 09:30\n{stroka}\n", "utf-8")
    return P, dict(zamorozheno_v="2026-10-28 09:00", reopen_note=note,
                   da_vladeltsa=dict(stroka=stroka, zhurnal="Ж-099"))


def dry_run(tmp: Path) -> dict:
    meta = make_meta()
    P, spec = make_spec(tmp, meta)
    return sv.run(P, spec, meta, make_factory(tmp), reps_lw=200, reps_k2=20)
