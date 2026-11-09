"""V6 лист FIN: сводные таблицы FIN_ALL, FIN_YEARS, FIN_TESTS, FIN_ENCOMPASS, FIN_STRESS, FIN_MCS, FIN_INFO_ALL, FIN_LEVELS — из СОХРАНЁННЫХ прогнозов 28 рядов (10 контролей и 18 конфигураций). Определения прежние (функции листов 1a–6 без правок):
метрики `v6_1_eval.metrics_table`, Δ в % уровня K1 с 90 % интервалом (блок 20, 2000, seed 0), DM (НУ 5), Холм N = 722 / 3122, охват, стресс-окна. MCS 90 % — реализация V5: `stats_d.mcs` и `hp_stats_tst.block_idx` (импорт)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_tst as TST
from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_io as IO, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV, v6_1_stage as S1, v6_1b_tab as TB, v6_fin_data as D

MCS_WINS = ("2017_2021", "2022_2025", "all")
SENS_COLS = ["qlike_all", "d_pct_K1_all", "qlike_2017_2021", "d_pct_K1_2017_2021"]
YEAR_WINS = [str(y) for y in range(2017, 2027)]


def metrics(d: dict) -> pd.DataFrame:
    return EV.metrics_table(d["w"]["rv"].to_numpy(), d["fs"], d["w"]["t_plus"].to_numpy())


def years(allm: pd.DataFrame) -> pd.DataFrame:
    y = allm[allm["window"].isin(YEAR_WINS)].pivot(index="window", columns="series", values="qlike")[D.SER]
    y.index = y.index.astype(int)
    return y.rename_axis("year").reset_index()


def tests(d: dict) -> pd.DataFrame:
    return pd.concat([S1.tests_all(d["w"], d["fs"][c], d["cp0"]).assign(config=c) for c in D.CFG], ignore_index=True)


def encompass(tst: pd.DataFrame) -> pd.DataFrame:
    return tst[tst["section"] == "coverage"][["config", "window", "base", "w", "lo90", "hi90"]].reset_index(drop=True)


def read_sens(c: str) -> pd.DataFrame:
    s = IO.read_csv("%s_SENS.csv" % c)
    return s[["variant"] + SENS_COLS].rename(columns={"variant": "scenario"}).assign(config=c, kind="sens")


def stress(d: dict) -> pd.DataFrame:
    """Стресс-окна (пересчёт из прогнозов) и соседние точки (чувствительность: λ, число деревьев, k, κ — читаются с диска, обучение не повторяется)."""
    w, fs, out = d["w"], d["fs"], []
    refs = {k: fs[k] for k in ("K1", "K2", "K3", "K8")}
    for c in D.CFG:
        out.append(CV.stress_table(w["rv"].to_numpy(), fs[c], refs, w["t_plus"].to_numpy(), w["K1"].to_numpy()).assign(config=c, kind="stress"))
        out.append(read_sens(c))
    cols = ["config", "kind", "scenario", "n", "qlike", "qlike_K1", "d_pct_K1", "d_pct_K2", "d_pct_K3", "d_pct_K8"] + SENS_COLS
    return pd.concat(out, ignore_index=True).reindex(columns=cols)


def mcs(d: dict) -> pd.DataFrame:
    rv, tp, rows = d["w"]["rv"].to_numpy(), d["w"]["t_plus"].to_numpy(), []
    for win in MCS_WINS:
        m = Q.wmask(tp, win)
        L = np.column_stack([Q.ql(rv[m], d["fs"][s][m]) for s in D.SER])
        r = SD.mcs(L, TST.block_idx(len(L)))
        rows += [dict(window=win, model=s, mean_qlike=float(L[:, i].mean()), mcs_p=float(r["p"][i]), in_mcs90=bool(r["in"][i]), n_dates=len(L)) for i, s in enumerate(D.SER)]
    return pd.DataFrame(rows)


def info_pairs(info: pd.DataFrame) -> list:
    """Пересчёт показателей пар (без Холма, он зависит от состава пар листа) — для сверки с `*_INFO.csv`."""
    return list(dict.fromkeys(zip(info["x"], info["y"])))


def recompute_info(d: dict, info: pd.DataFrame) -> pd.DataFrame:
    return TB.info_table(d["fs"], info_pairs(info), d["w"], d["cp0"])


def tables(d: dict) -> dict:
    from models_v2.s13 import v6_fin_abl as AB
    allm, tst = metrics(d), tests(d)
    return {"FIN_ALL": allm, "FIN_YEARS": years(allm), "FIN_TESTS": tst, "FIN_ENCOMPASS": encompass(tst), "FIN_STRESS": stress(d), "FIN_MCS": mcs(d), "FIN_INFO_ALL": D.read_info(),
            "FIN_LEVELS": D.read_levels(), "FIN_ABLATION": AB.ablation(d)}
