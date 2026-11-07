"""Р-1, находка 1: воспроизведение нестабильности test_loyo_ne_vidit_celi_goda_Y (код проекта не меняется).

Причина: select.run_loyo -> runner.map_jobs берёт результаты через Pool.imap_unordered; порядок строк df (и порядок суммирования
в groupby.mean, и порядок строк `out` при sort=False) зависит от порядка прихода задач из двух процессов.
Тест сравнивает a["score"].tolist() == b["score"].tolist() — списки, а не словари по метке и не с допуском.

Запуск (из корня репозитория):
  python3 -W ignore docs/otchety/harj_panel_hybrid/r1/r1_loyo_poryadok.py det     # детерминированно: подмена порядка прихода
  python3 -W ignore docs/otchety/harj_panel_hybrid/r1/r1_loyo_poryadok.py nat N   # естественно: N прогонов, считает варианты порядка/значений
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, ".")
from models_v2.s4 import mkt, runner, select  # noqa: E402
from models_v2.s4.fit import HP  # noqa: E402
from models_v2.s10 import hp_k23_world as KW, hp_synth as SS  # noqa: E402

SMALL = (HP(2, 30.0, 2.0, 0.1, 0.8, 0.5, 40), HP(3, 30.0, 2.0, 0.1, 0.8, 0.5, 40))


def make_world():
    base = SS.weekdays(900)
    skip_b = [base[i] for i in range(300, 330)]
    skip_t = [d for d in base if d < pd.Timestamp("2017-10-02")]
    bars, imo, _ = SS.world(900, ("A", "B", "C", "T"), skip={"B": skip_b, "T": skip_t})
    return KW.build_world(bars["A"], imo, [mkt.paper(bars[t]) for t in ("A", "B", "C")])


def scores(w) -> pd.DataFrame:
    return KW.loyo_scores(w, (2018,), 2)


def det(w) -> None:
    orig = runner.map_jobs
    a = scores(w)
    runner.map_jobs = lambda *x, **k: list(reversed(orig(*x, **k)))   # то же множество результатов, другой порядок прихода
    try:
        b = scores(w)
    finally:
        runner.map_jobs = orig
    print("a:", a["score"].tolist(), a["label"].tolist() if "label" in a else "")
    print("b:", b["score"].tolist())
    print("списки равны (как в тесте):", a["score"].tolist() == b["score"].tolist())
    sa, sb = a.set_index("label")["score"].to_dict(), b.set_index("label")["score"].to_dict()
    print("словари по метке равны:", sa == sb, "| макс. |Δ|:", max(abs(sa[k] - sb[k]) for k in sa))


def nat(w, n: int) -> None:
    seen_order, seen_val = set(), set()
    for _ in range(n):
        s = scores(w)
        seen_order.add(tuple(s["label"]))
        seen_val.add(tuple(s["score"]))
    print(f"прогонов {n}: вариантов порядка меток {len(seen_order)}, вариантов списка score {len(seen_val)}")


if __name__ == "__main__":
    select.GRID132 = SMALL
    world = make_world()
    if len(sys.argv) > 1 and sys.argv[1] == "nat":
        nat(world, int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        det(world)
