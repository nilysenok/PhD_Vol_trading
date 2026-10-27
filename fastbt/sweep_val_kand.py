"""Фабрика кандидатов validation A′ и заморозка списка (Ж-031 Э1-д, Ж-037). Читает только таблицы итераций (train) и ledger.

Кандидаты: (а) стратегия Sk: топ-3 различных по плато × панель одной стратегии, сборка (равные, net, net), база — «A только Sk» (s9);
(б) A′: топ-2 сборки × ранг 1…3 точек всех шести стратегий, база — боевая A (s9, Ж-039 п. 1; ссылка `sweep_val_baza`). Дубли (отличаются только неактивным параметром)
сводятся, топ-3 добирается следующим различным. Заморозка — `docs/otbor/zamorozka_A_prim.json`; `sweep_val` читает её оттуда.
Реальная фабрика данные читает ТОЛЬКО под замком (вызывается из `sweep_val.run` после замка).
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import sweep_setka as sk
from fastbt import sweep_val_baza as bz
from fastbt import sweep_val as sv

SIDS = tuple(f"S{i}" for i in range(1, 7))
DIR = Path(__file__).parent / "designs" / "sweep"
TABLES = Path("docs/tables/sweep_e1v2")
NET = {"vesa": "ravnye", "limit_pozitsiy": "net", "limit_na_strategiyu": "net"}
TOP_K, N_A = 3, 36
REN = {"rang": "rank", "plato": "plateau", "sosedey": "neighbors", "tune_004": "tune_sharpe", "ctrl_004": "ctrl_sharpe",
       "storon_god": "storon_y"}                                      # S1–S3 и S4–S6 пишут шапки по-разному


def _typed(v, key, grid):
    """Значение из csv -> тип сетки yaml (int там, где сетка целая); пусто -> None."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v
    if v is None or pd.isna(v):
        return None
    nums = [g for g in grid[key] if isinstance(g, (int, float)) and not isinstance(g, bool)]
    return int(v) if nums and all(isinstance(g, int) for g in nums) else float(v)


def eff_key(sid: str, p: dict) -> str:
    """Действующие параметры точки: неактивные RM-блоки и т. п. отброшены (`sweep_post.split`)."""
    from fastbt.sweep_post import split
    from fastbt.sigprov import SIG_GRIDS
    sig = set(SIG_GRIDS[sid])
    return json.dumps([{k: p[k] for k in sorted(sig)}, *split(sid, p, sig)], sort_keys=True, default=str)


def _sig(r) -> tuple:
    """Подпись результата на train: точки с одинаковыми tune/ctrl/сторонами — одна и та же торговля (параметр, различающий
    их, на этих данных не действует, напр. tf_reentry 3 и 10 у S5): дубль."""
    return tuple(round(float(r[c]), 9) for c in ("tune_sharpe", "ctrl_sharpe", "storon_y"))


def kandidaty(sid: str, root=".", k: int = TOP_K) -> dict:
    """{base: исходная точка, cands: [{rank, point, neighbors, plateau, n_vsego}]} — топ-k различных из `S{k}_kandidaty_plato.csv`."""
    exp = sk.expand(DIR / f"{sid}_setka_e1b.yaml")
    base, grid, d = exp["base"], exp["grid"], Path(root) / TABLES
    pool = pd.read_csv(d / f"{sid}_kandidaty_plato.csv", comment="#").rename(columns=REN).sort_values("rank")
    top = pd.read_csv(d / f"{sid}_top3.csv", comment="#")
    top = top[top["mode"] == "plato"].sort_values("rank")

    def pt(r):
        extra = [c for c in r.index if c.startswith("p.") and c[2:] not in base]
        if extra:
            raise ValueError(f"{sid}: параметры вне сетки {extra}")
        return {key: _typed(r[f"p.{key}"], key, grid) if f"p.{key}" in r.index else base[key] for key in base}

    for j in range(len(top)):                        # топ-3 файла и начало пула — одни и те же точки
        if pt(top.iloc[j]) != pt(pool.iloc[j]):
            raise ValueError(f"{sid}: top3.csv и kandidaty_plato.csv расходятся на ранге {j + 1}")
    seen, out = set(), []
    for _, r in pool.iterrows():
        p = pt(r)
        ks = (eff_key(sid, p), _sig(r))
        if not seen & set(ks):
            seen |= set(ks)
            out.append(dict(rank=int(r["rank"]), point=p, neighbors=int(r["neighbors"]), plateau=float(r["plateau"]),
                            n_vsego=int(r["N_vsego"])))
        if len(out) == k:
            return dict(base=dict(base), cands=out)
    raise ValueError(f"{sid}: различных кандидатов {len(out)} < {k}")


def top2_sborki(root=".") -> list:
    t = pd.read_csv(Path(root) / TABLES / "A_sborki.csv", comment="#")
    t = t[t["top2"] > 0].sort_values("top2")
    if len(t) != 2:
        raise ValueError(f"A_sborki.csv: топ-2 сборок {len(t)} ≠ 2")
    f = lambda v: v if v == "net" else int(v)                      # noqa: E731
    return [dict(vesa=r.vesa, limit_pozitsiy=f(r.limit_pozitsiy), limit_na_strategiyu=f(r.limit_na_strategiyu),
                 plateau_neighbors=int(r.plateau_neighbors)) for r in t.itertuples()]


def build_spisok(root=".") -> tuple:
    """(spisok, bazy, N_vsego) — всё JSON-сериализуемо; порядок и состав однозначны по таблицам."""
    kd = {s: kandidaty(s, root) for s in SIDS}
    sp =[dict(id=f"{s}:{i}", kind="S", sid=s, params=NET, orig_params=NET, tochki={s: c["point"]}, baza=bz.baza_ref(s),
               neighbors=c["neighbors"], plateau=c["plateau"], rank_tablitsy=c["rank"])
          for s in SIDS for i, c in enumerate(kd[s]["cands"], 1)]
    for j, sb in enumerate(top2_sborki(root), 1):
        for i in range(TOP_K):
            sp.append(dict(id=f"A:{i + 1}:{j}", kind="A", params={k: sb[k] for k in NET}, orig_params=NET,
                           tochki={s: kd[s]["cands"][i]["point"] for s in SIDS}, baza=bz.baza_ref("A"),
                           neighbors=sb["plateau_neighbors"], plateau=None, rank_tablitsy=i + 1))
    n = {s: kd[s]["cands"][0]["n_vsego"] for s in SIDS}
    return sp, {b: bz.baza_ref(b) for b in ("A", *SIDS)}, dict(n, A=N_A)


def freeze(root=".", path=None) -> dict:
    path = Path(path or Path(root) / "docs" / "otbor" / "zamorozka_A_prim.json")
    if path.exists():
        raise FileExistsError(f"{path}: заморозка уже есть; новая — отдельное решение")
    sp, bazy, n = build_spisok(root)
    z = dict(zamorozheno_v=datetime.now().strftime("%Y-%m-%d %H:%M"), sha=sv.sha_kandidatov(sp), N_vsego=n, bazy=bazy, spisok=sp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(z, ensure_ascii=False, indent=1, sort_keys=True), "utf-8")
    return z


def _commit_ok(P) -> None:
    f = Path(P.zamorozka).resolve()
    rel, g = str(f.relative_to(Path(P.koren).resolve())), ["git", "-C", str(P.koren)]
    if subprocess.run([*g, "ls-files", "--error-unmatch", rel], capture_output=True).returncode != 0:
        raise sv.Stop("файл заморозки не закоммичен")
    if subprocess.run([*g, "diff", "--quiet", "HEAD", "--", rel], capture_output=True).returncode != 0:
        raise sv.Stop("файл заморозки правлен после коммита")


def load_zamorozka(P) -> dict:
    """Заморозка из файла: sha обязан совпасть с пересчётом по списку, база кандидата — s9-ссылка нужного вида, равная bazy; файл — как в коммите."""
    f = Path(P.zamorozka)
    if not f.exists():
        raise sv.Stop(f"нет файла заморозки {f}")
    z = json.loads(f.read_text("utf-8"))
    if sv.sha_kandidatov(z["spisok"]) != z["sha"]:
        raise sv.Stop("файл заморозки правлен без пересчёта sha")
    if z["bazy"] and any(not bz.baza_ok(m, z["bazy"]) for m in z["spisok"]):         # bazy пуст — только синтетика `--dry`
        raise sv.Stop("база не s9-A / «A только Sk» (или не равна записи в bazy)")
    if P.git_proverka:
        _commit_ok(P)
    return z


def grid_a() -> list:
    from fastbt.sweep_a import load_grid
    g = load_grid()
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


def ledger_sk(sid: str, root) -> tuple:
    """SR уникальных новых проб (ключи ledger уникальны) и их ряды (PBO). N — отдельно (`Cand.n_total`), список не дополняется
    (толкование 27.10 20:20)."""
    d = Path(root) / TABLES / "_ledger" / sid
    return list(json.loads((d / "ledger.json").read_text("utf-8")).values()), json.loads((d / "ledger_excess.json").read_text("utf-8"))


def n_a(n_vsego: dict) -> int:
    """N для сборок A′: 36 проб сборки + Σ N_vsego по S1–S6 (точки выбраны среди всех проб стратегий; толкование 27.10 20:20)."""
    return int(n_vsego["A"]) + sum(int(n_vsego[s]) for s in SIDS)


def make_factory(P, root=None):
    """factory(meta, note) -> [Cand]; каталоги проб — внутри P.out (его до взгляда нет)."""
    from fastbt.sweep_a import SkPanelSource, make_sweep
    from fastbt.sweep_val_stat import Cand
    root, srcs, bases = Path(P.koren if root is None else root), {}, {}

    def src(points):
        return srcs.setdefault(json.dumps(points, sort_keys=True, default=str), SkPanelSource(points))

    def factory(meta: list, note) -> list:
        z, out = json.loads(Path(P.zamorozka).read_text("utf-8")), []
        for m in meta:
            d = Path(P.out) / "_sw" / m["id"].replace(":", "_")
            sw = make_sweep(src(m["tochki"]), str(d / "c"), note)
            ow = bases.setdefault(m["baza"]["id"], bz.S9Baza(m["baza"]))     # база — s9, одна на A / на Sk, не центр сетки
            if m["kind"] == "S":
                trials, exc = ledger_sk(m["sid"], root)
                n_total = z["N_vsego"][m["sid"]]
            else:
                n_total = n_a(z["N_vsego"])
                for p in grid_a():                    # N = 36 проб сборки на setup 2018 при 0,04 — ledger и ряды для PBO
                    sw.evaluate(p, "setup")
                trials, exc = sw.trials_sr(), sw.excess
            out.append(Cand(m["id"], m["kind"], sw, m["params"], ow, m["orig_params"], m["neighbors"], trials, exc, n_total))
        return out
    return factory


if __name__ == "__main__":
    z = freeze(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(f"кандидатов {len(z['spisok'])}, sha {z['sha']}, заморожено {z['zamorozheno_v']}, N_vsego {z['N_vsego']}")
