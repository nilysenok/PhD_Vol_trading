"""V7-1b-B: запуск конфигураций 1–2 и выбор энкодера для 3 (стадия `run_v7 --stage 1b`). Пишет `data/regen/harj_hybrid_v7/1b/cfg1.parquet`, `cfg2.parquet` (d, t_plus, rv, H, g, F),
CHECKPOINT `1b.cfg1`, `1b.cfg2` (sha256 содержимого и файла), `1b.choice3` — ДО любого счёта конфигурации 3 (она здесь не считается), строки результата в ALL_TRIALS.csv и `1b_B_QLIKE.csv`.
Только QLIKE 2018–2021 (выбор); окна 2022+, p-значения и сравнения с K2/K3/NTP2/A2 — лист 1b-C. `check()` ничего не пишет."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_cfg as CFG, v7_1b_sta as STA, v7_1m_reg as RG

STATUS_1B = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-232, лист 1b-B (конфигурации 1–2; окна 2022+ не оценивались; конфигурация 3 не считалась)"
CHECK_FROM = pd.Timestamp("2019-07-01")                 # переоценка для побитной сверки: первая, не раньше этой даты


def path(kind: str):
    return IO.OUT / "1b" / ("cfg%s.parquet" % CFG.CFG_ID[kind])


def content_hash(df: pd.DataFrame) -> str:
    return STA.content_hash({c: (df[c].to_numpy("datetime64[ns]").astype(np.int64) if str(df[c].dtype).startswith("datetime") else df[c].to_numpy(float)) for c in df.columns})


def read_cp() -> dict:
    import json
    p = IO.REP / "CHECKPOINT.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def update_1b(sub: str, value: dict) -> None:
    cur = dict(read_cp().get("1b", {}))
    cur[sub] = value
    IO.merge_checkpoint("1b", cur)


def save(kind: str, df: pd.DataFrame, seconds: float) -> dict:
    p = path(kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    q, n = CFG.qlike_choice(df)
    return dict(file=str(p.relative_to(IO.ROOT)), sha256_content=content_hash(df), sha256_file=IO.sha_file(p), n_rows=len(df), first=str(df["d"].iloc[0].date()), last=str(df["d"].iloc[-1].date()),
                qlike_2018_2021=q, n_dates_2018_2021=n, params=5, grid_lam_g=[g[0] for g in CFG.GRID], lam_lb_lam_s="не применяются (нет маршрутизатора)", seeds=list(CFG.FT.SEEDS), seconds=round(seconds, 1))


def write_cfgs() -> int:
    import shutil
    if shutil.disk_usage(IO.ROOT).free / 1e9 < 40:
        raise SystemExit("свободно < 40 ГБ: запись отменена")
    from models_v2.s14 import v7_1m_data as DATA
    tab, dfs = DATA.load_tab(), {}
    for k in CFG.KINDS:
        cp = read_cp().get("1b", {}).get("cfg" + CFG.CFG_ID[k])
        if cp and path(k).exists() and IO.sha_file(path(k)) == cp["sha256_file"]:
            dfs[k] = pd.read_parquet(path(k))
            print("конфигурация %s: уже посчитана, пропуск" % CFG.CFG_ID[k])
            continue
        t0 = time.time()
        dfs[k] = CFG.run_kind(k, tab)
        update_1b("cfg" + CFG.CFG_ID[k], save(k, dfs[k], time.time() - t0))
        print("конфигурация %s (%s): %d строк, %.0f с" % (CFG.CFG_ID[k], k, len(dfs[k]), time.time() - t0))
    ch = CFG.choose(dfs)
    old = read_cp().get("1b", {}).get("choice3")
    if old and old["choice"] != ch["choice"]:
        raise SystemExit("choice3 уже записан (%s) и не совпал с пересчётом (%s): не переписывается" % (old["choice"], ch["choice"]))
    if not old:
        update_1b("choice3", dict(ch, rule=RG.ENC_RULE, written_before_cfg3=True, cfg3_computed=False))
    for k in CFG.KINDS:
        RG.add_result(CFG.CFG_ID[k], dict(stage="1b-B", qlike_2018_2021=ch["qlike"][k], sha256_content=read_cp()["1b"]["cfg" + CFG.CFG_ID[k]]["sha256_content"], n_rows=len(dfs[k])))
    tbl = pd.DataFrame([dict(config=CFG.CFG_ID[k], encoder=k, qlike_2018_2021=ch["qlike"][k], n_dates=ch["n_dates"], chosen_for_3=(k == ch["choice"])) for k in CFG.KINDS])
    IO.write_csv(tbl, "1b_B_QLIKE.csv", "1b-B: средняя QLIKE на датах целей 2018–2021 (общие даты), выбор энкодера для конфигурации 3", status=STATUS_1B)
    print("QLIKE 2018–2021: %s; выбор для 3: %s" % ({k: round(v, 6) for k, v in ch["qlike"].items()}, ch["choice"]))
    return 0


def check(cp: dict) -> list:
    """-> замечания (пусто = ок). Читает файлы, ничего не пишет; одна переоценка пересчитывается побитно."""
    bad, dfs = [], {}
    if not all(("cfg" + i) in cp for i in CFG.CFG_ID.values()) or "choice3" not in cp:
        return ["CHECKPOINT 1b: нет cfg1/cfg2/choice3"]
    for k in CFG.KINDS:
        rec, p = cp["cfg" + CFG.CFG_ID[k]], path(k)
        if not p.exists():
            bad.append("%s: нет файла" % p.name)
            continue
        df = pd.read_parquet(p)
        dfs[k] = df
        if IO.sha_file(p) != rec["sha256_file"] or content_hash(df) != rec["sha256_content"]:
            bad.append("%s: sha256 ≠ CHECKPOINT" % p.name)
        if not ((df["F"] > 0).all() and np.array_equal(df["F"][df["g"] == 0], df["H"][df["g"] == 0])):
            bad.append("%s: F ≤ 0 или g = 0, но F ≠ H" % p.name)
    if bad:
        return bad
    if CFG.choose(dfs)["choice"] != cp["choice3"]["choice"]:
        bad.append("choice3 ≠ пересчёту по сохранённым прогнозам")
    from models_v2.s13 import v6_1_wf as W6
    from models_v2.s14 import v7_1m_data as DATA, v7_1m_wf as WF
    tab = DATA.load_tab()
    a = next(i for i in W6.refit_positions(tab.index, WF.FIRST_REFIT) if tab.index[i] >= CHECK_FROM)
    got, df = CFG.refit_block("tcn", tab, a), dfs["tcn"]
    want = df.set_index("d")["g"].reindex(tab.index[a:a + len(got)]).to_numpy(float)
    if not np.array_equal(got, want):
        bad.append("переоценка %s (TCN): пересчёт не совпал побитно с cfg1.parquet" % tab.index[a].date())
    return bad
