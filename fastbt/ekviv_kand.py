"""Перезаморозка A′ (вариант (а), Ж-039/Ж-040): новый топ-3 по плато среди точек, которые воспроизводит боевой путь s9.

`python -m fastbt.ekviv_kand <scratch>` — по каждой S1–S6 `ekviv_filtr.plato_vosp` (ledger копируется в scratch), `izbrat` (соседей ≥ 3,
активность ≥ 0,5 базы — внутри `plato_vosp`, дубли по подписи сведены), запись `kand_a.csv`, затем дописывание реестра
(`reestr_otbora_pz`). Данные train 2018/2019; validation и ≥ 2022 не читаются. Расширенные ledger остаются в scratch.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from fastbt import ekviv_filtr as ef

SIDS = tuple(f"S{i}" for i in range(1, 7))
KAND = Path("docs/tables/sweep_e1v2/kand_a.csv")
REN = {"tune_sharpe_c0.035": "tune_sharpe_c0035", "ctrl_sharpe_c0.035": "ctrl_sharpe_c0035", "sdelok_y": "storon_y",
       "tune_sharpe": "tune_004", "ctrl_sharpe": "ctrl_004"}
HEAD = ("# статус: разработка, train 2018 (tune) / 2019 (ctrl), до взгляда на validation; перезаморозка A′, вариант (а): только точки с боевым "
        "эталоном s9 (ekviv_filtr); {t}\n# пул: топ-10 воспроизводимых допустимых проб ledger по цели (+ достроенные соседи ±1, они в N), по плато; top3=1 — выбраны; "
        "соседей ≥ 3, активность ≥ 0,5 базы tune 2018; storon_y = сторон/год; N_vsego = проб ledger после достройки + N_starye; "
        "dsr_vsego — var_sr по ledger репозитория (392…451 проб), N = N_vsego\n")


def tablitsa(res: dict, k: int = 3) -> pd.DataFrame:
    """Пул (без краёв, по плато) всех стратегий одной таблицей: rank, top3 (1 — выбрана `izbrat`), плато, метрики, N, DSR, p.*."""
    out = []
    for sid, r in res.items():
        t = r["table"]
        pool = t[~t["edge"]].sort_values("plateau", ascending=False)
        pool = pool.rename(columns=REN).drop(columns=["mode", "edge", "objective", "comm_pct", "N", "dsr"])
        pool.insert(0, "top3", pool.index.isin(ef.izbrat(t, k).index).astype(int))
        pool.insert(0, "rank", range(1, len(pool) + 1))
        pool.insert(0, "strategiya", sid)
        out.append(pool)
    return pd.concat(out, ignore_index=True)


YAML = Path(__file__).parent / "designs" / "sweep" / "ekviv_kand.yaml"
YAML_HEAD = """# Кандидаты перезаморозки A′ (вариант (а)) — точки из kand_a.csv (`sweep_val_kand.kandidaty`), класс «кандидаты заморозки»: сверка с боевым путём s9
# (`ekviv_s9.run_sk`) тем же допуском, что `ekviv_20.yaml`: позиции побайтно, дневной PnL abs ≤ 1e-12; окна 2018 и 2019, комиссии 0,04 и 0,035 %.
v: 1
format: ekviv-kand
dopusk: {pozitsii: pobaytno, pnl_abs: 1.0e-12}
komissii_pct: [0.04, 0.035]
gody: [2018, 2019]
tochki:
"""


def zapisat_yaml(root=".") -> int:
    """Пишет `ekviv_kand.yaml` по `kandidaty()` (в порядке S1…S6, ранг 1…3); возвращает число точек."""
    import json
    from fastbt import sweep_val_kand as kd
    rows = [f"  - {{id: K_{s}_{i}, sid: {s}, klass: kandidat_zamorozki, etalon: da, p: {json.dumps(c['point'], ensure_ascii=False)}}}"
            for s in SIDS for i, c in enumerate(kd.kandidaty(s, root)["cands"], 1)]
    YAML.write_text(YAML_HEAD + "\n".join(rows) + "\n", "utf-8")
    return len(rows)


def main(scratch, root="."):
    from fastbt import reestr_otbora_pz as pz
    res = {s: ef.plato_vosp(s, scratch, root) for s in SIDS}
    t = tablitsa(res)
    (Path(root) / KAND).write_text(HEAD.format(t=datetime.now().strftime("%d.%m.%Y %H:%M")) + t.to_csv(index=False), "utf-8")
    pz.dopisat(res, t[t["top3"] == 1], root)
    for s, r in res.items():
        print(s, {x: r[x] for x in ("n_before", "n_after", "n_dopustimyh", "n_vosp")})
    print(t[t["top3"] == 1][["strategiya", "rank", "plateau", "neighbors", "tune_004", "ctrl_004", "storon_y", "N_vsego", "dsr_vsego"]].round(4).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "yaml":
        print("точек:", zapisat_yaml())
    else:
        main(sys.argv[1])
