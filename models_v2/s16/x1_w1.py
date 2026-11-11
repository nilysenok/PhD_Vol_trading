"""W1 (Ж-250 пп. 1–2): сопоставимая таблица SBER, h = 1, по СОХРАНЁННЫМ прогнозам: склейка архив + продолжение 2026, QLIKE окон M, H, 2017, F (среднее ДНЕВНЫХ потерь на общих датах),
Δ к K2, K3, X1 в % QLIKE базы; бутстрап, DM, Холм с N программы, MCS; вклад лет; PIT-состояния S1…S8; смещение. Обучения и новых прогнозов нет; испытаний не регистрируется.
Запуск: `python3 -W ignore -m models_v2.s16.x1_w1 [--write | --check]`; --check ничего не пишет. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; диагностика после просмотра; Ж-250."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_t0 as T0, x1_w1_join as J

REP, CKPT = T0.REP, T0.REP / "CHECKPOINT.json"
OUTS = ("W1_JOIN.csv", "W1_TABLE.csv", "W1_STATS.csv", "W1_YEARS.csv", "W1_CONC.csv", "W1_STATES.csv", "W1_BIAS.csv", "W1_REPORT.md")
MAIN = ("K1", "K2", "K3", "X1", "ZB", "ZB0", "cfg8")
T0S = ("K1", "K2", "K3", "A2", "NTP2", "cfg6", "cfg8", "X1")
SPEC = (("M", ("K1", "K2", "K3", "X1"), ("ZB", "ZB0", "cfg8"), None, None), ("H", T0S, ("ZB", "ZB0"), None, J.ARCH_END),
        ("2017", ("K1", "K2", "K3", "A2", "NTP2"), (), pd.Timestamp("2017-01-01"), pd.Timestamp("2017-12-31")), ("F", MAIN, (), pd.Timestamp("2026-03-02"), None))
HEAD = "# DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; диагностика после просмотра; Ж-250 (лист W1; сохранённые прогнозы, обучения нет); l = RV/F − ln(RV/F) − 1, среднее дневных потерь на общих датах окна; код — ветка harj-x1, 311b9ea3\n"
ORIGIN = {"K1": "T0 (Ж-242) ← harj_panel_hybrid/har_SBER.parquet; продолжение — Л1 (`s17/fwd_k1`)", "K2": "T0 ← harj_panel_hybrid/k23_SBER.parquet; продолжение — Л2 (`s17/fwd_k23`)", "K3": "T0 ← harj_panel_hybrid/k23_SBER.parquet; продолжение — Л2",
          "A2": "V6 (Ж-227) ← harj_hybrid_v6/A2.parquet; только архив", "NTP2": "V4 (Ж-225) ← harj_hybrid_v4/NTP2.parquet; только архив", "cfg6": "V7 лист 2 ← harj_hybrid_v7/2/cfg6.parquet; только архив",
          "cfg8": "V7 лист 3 ← harj_hybrid_v7/3/cfg8.parquet; продолжение — Л4 (справочно)", "X1": "Deep E1 (Ж-240) ← harj_deep/E1/x1.parquet; продолжение — Л4", "ZB": "лист Z (Ж-246): ZB ← data/fwd/models/zb.parquet; продолжение — Л4", "ZB0": "лист Z (Ж-246): ZB0 ← data/fwd/models/zb0.parquet; продолжение — Л4"}


def sl(df: pd.DataFrame, lo, hi) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    m &= (df["t_plus"] >= lo).to_numpy() if lo is not None else True
    m &= (df["t_plus"] <= hi).to_numpy() if hi is not None else True
    return df[m].reset_index(drop=True)


def windows(fr: dict) -> dict:
    """Окно -> {ряд: кадр на ОБЩИХ датах}. Ряды, не покрывающие общий период ядра, получают отдельное окно «окно|ряд» (не обрезают главное)."""
    out = {}
    for name, core, extra, lo, hi in SPEC:
        cut = {k: sl(fr[k], lo, hi) for k in core + extra}
        idx = T0.common({k: cut[k] for k in core})["K1"]["d"]
        names = list(core) + [e for e in extra if idx.isin(cut[e]["d"]).all()]
        out[name] = T0.common({k: cut[k] for k in names})
        for e in extra:
            if e not in names:
                out["%s|%s" % (name, e)] = T0.common({k: cut[k] for k in core + (e,)})
    return out


def window_qlike(rv, f) -> float:
    """QLIKE окна = среднее дневных потерь (не среднее показателей периодов; мутация № 160)."""
    return float(Q.ql(np.asarray(rv, float), np.asarray(f, float)).mean())


def pct(q: float, qb: float) -> float:
    return 100.0 * (q - qb) / qb


def mcs90(rv, F: dict) -> pd.DataFrame:
    """MCS 90 % (T_max; блок 20, B 2000, seed 0) на дневных потерях всего окна. Как `v7_1m_stat.mcs90`, но без его окна «all» (2018–2026): окно 2017 в нём пустое."""
    names = list(F)
    L = np.column_stack([Q.ql(rv, F[k]) for k in names])
    r = T0.ST.SD.mcs(L, T0.ST.SD.block_idx(len(L), B_=T0.ST.B, block=T0.ST.BLOCK, seed=T0.ST.SEED), alpha=0.10)
    return pd.DataFrame({"series": names, "mean_qlike": L.mean(axis=0), "p": r["p"], "in_mcs": r["in"]}).set_index("series")


def table(wins: dict, sha: tuple) -> pd.DataFrame:
    rows = []
    for w, al in wins.items():
        ref = al["K1"]
        if not all(pd.DatetimeIndex(v["d"]).equals(pd.DatetimeIndex(ref["d"])) for v in al.values()):
            raise SystemExit("окно %s: ряды заданы не на одних и тех же общих датах" % w)
        rv, tp = ref["rv"].to_numpy(float), pd.DatetimeIndex(ref["t_plus"])
        mc = mcs90(rv, {k: v["F"].to_numpy(float) for k, v in al.items()})
        qs = {k: window_qlike(rv, v["F"]) for k, v in al.items()}
        for k, v in al.items():
            b = {x: (pct(qs[k], qs[x]) if x in qs else np.nan) for x in ("K2", "K3", "X1")}
            rows.append(dict(window=w, series=k, origin=ORIGIN[k], archive_sha256=sha[0][k], new_sha256=sha[1].get(k, ""), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), n=len(tp),
                             qlike=qs[k], d_k2_pct=b["K2"], d_k3_pct=b["K3"], d_x1_pct=b["X1"], mcs90=bool(mc.at[k, "in_mcs"]), mcs_p=float(mc.at[k, "p"])))
    return pd.DataFrame(rows)


def input_shas() -> dict:
    ps = sorted((J.MODELS).glob("*.parquet")) + sorted(J.NEW.glob("*.parquet"))
    return {str(p.relative_to(T0.ROOT)): T0.sha_file(p) for p in ps}


def compute() -> dict:
    """Все таблицы и текст отчёта из сохранённых прогнозов (ничего не пишет)."""
    from models_v2.s16 import x1_w1_doc as DOC, x1_w1_st as S, x1_w1_stat as SS
    fr, jn, sha = J.load_joined()
    wins = windows(fr)
    t = dict(join=jn, table=table(wins, sha), stats=SS.stats_table(wins))
    t["years"], t["conc"] = SS.years_conc(wins["M"])
    t["states"], t["bias"] = S.states_bias(fr, wins["M"])
    t["md"] = DOC.build(t)
    return t


def csv_text(df: pd.DataFrame, extra: str = "") -> str:
    return HEAD + extra + df.to_csv(index=False, lineterminator="\n")


def join_header() -> str:
    return "".join("# input %s sha256 %s\n" % kv for kv in input_shas().items())


def files(t: dict) -> dict:
    n = dict(join="W1_JOIN.csv", table="W1_TABLE.csv", stats="W1_STATS.csv", years="W1_YEARS.csv", conc="W1_CONC.csv", states="W1_STATES.csv", bias="W1_BIAS.csv")
    out = {v: csv_text(t[k], join_header() if k == "join" else "") for k, v in n.items()}
    out["W1_REPORT.md"] = t["md"]
    return out


def merge_checkpoint(ckpt: Path, t: dict, texts: dict) -> None:
    """Только ключ W1, слиянием: прочие ключи сохраняются побайтно (формат файла: indent 1, sorted keys)."""
    old = ckpt.read_text(encoding="utf-8")
    cp = json.loads(old)
    cp["W1"] = dict(inputs=input_shas(), outputs={k: T0.hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in texts.items()}, n_holm=int(t["stats"]["n_holm"].iloc[0]), configurations=0, forecasts_written=0)
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write(out: Path = REP, ckpt: Path = CKPT) -> None:
    t = compute()
    texts = files(t)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in texts.items():
        (out / k).write_text(v, encoding="utf-8")
    merge_checkpoint(ckpt, t, texts)


def check(out: Path = REP) -> list:
    """Пересчёт и сверка с файлами; ничего не пишет."""
    t = compute()
    bad = []
    for k, v in files(t).items():
        p = out / k
        bad += [] if p.exists() and p.read_text(encoding="utf-8") == v else ["%s: текст ≠ пересчёту" % k]
    return bad


def tree_hash(out: Path = REP, ckpt: Path = CKPT) -> dict:
    return {p.name: T0.sha_file(p) for p in sorted(out.iterdir()) if p.is_file()} | {"CHECKPOINT.json": T0.sha_file(ckpt)}


def run_check(out: Path = REP, ckpt: Path = CKPT) -> list:
    """`--check`: пересчёт и сверка; хэш всех выходов и CHECKPOINT до и после обязан совпасть."""
    h0 = tree_hash(out, ckpt)
    bad = check(out)
    return bad + (["--check изменил выходы"] if tree_hash(out, ckpt) != h0 else [])


if __name__ == "__main__":
    if "--check" in sys.argv:
        bad = run_check()
        print("W1 --check: %s" % ("расхождений нет" if not bad else "; ".join(bad)))
        sys.exit(1 if bad else 0)
    write()
    print("W1: записано в", REP)
