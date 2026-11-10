"""Y-B, стадия: compute() (всё в памяти), write() (ПИШЕТ y3/y4.parquet, CHECKPOINT `YB` → `YB.eval` (БЕЗ p) → таблицы YB_*.csv с p → отчёт → `YB.rules` → `YB.condition`), check() (ТОЛЬКО ЧИТАЕТ: пересчёт, сверка, хэш выходов до и после).
Вход: `python3 -W ignore -m models_v2.s16.x1_yb_stage [--check]` (`run_x1.py` не правится: его текст входит в хэш P0). Расходуется 2 конфигурации (Y3, Y4); Y5/Y6 — отказ `x1_reg.guard` всегда.
Входы (X1, K1–K3, NTP2, A2, cfg6, cfg8, склад M10 V7, прогнозы Y1/Y2 листа Y-A) только читаются; sha256 Y1/Y2 сверяются с CHECKPOINT `YA`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым, до models_v2.s14)
from models_v2.s16 import x1_reg as R, x1_ya_stage as SA, x1_yb_doc as DOC, x1_yb_eval as EV, x1_yb_feat as FE, x1_yb_walk as W

ROOT, REP = R.ROOT, R.REP
OUT, REPORT = ROOT / "data/regen/harj_x1/YB", REP / "YB_REPORT.md"
TABLES = {"YB_%s" % k: k for k in ("METRICS", "PAIRS", "RULE62", "YEARS", "CONDITION", "GO", "MCS", "STRESS", "CELLS", "PARAMS")}
HEAD = "# диагностика/оценка, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-243, лист Y-B (Y3, Y4; SBER, h = 1); код yb %s\n"
YA_OUT = SA.OUT


def code_hash() -> str:
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("x1_yb_*.py")) + [Path(R.__file__)]:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def load() -> dict:
    """Входы листа Y-A (SA.load) + Y1/Y2 из файлов Y-A (sha256 = CHECKPOINT YA) + YA_RULE62.csv."""
    inp = SA.load()
    cp = SA.read_cp().get("YA")
    if not cp:
        raise SystemExit("нет ключа YA в CHECKPOINT: лист Y-A не выполнен")
    for y in ("Y1", "Y2"):
        p = YA_OUT / (y.lower() + ".parquet")
        if SA.sha_file(p) != cp["sha256"][y]:
            raise SystemExit("%s.parquet листа Y-A ≠ sha256 в CHECKPOINT YA" % y.lower())
        f = pd.read_parquet(p)
        if not pd.DatetimeIndex(f["d"]).equals(inp["d"]):
            raise SystemExit("%s: даты ≠ общим" % y)
        inp["ser"][y] = f["F"].to_numpy(float)
        inp["shas"][y.lower() + "_parquet"] = SA.sha_file(p)
    inp["rule_a"] = pd.read_csv(REP / "YA_RULE62.csv", comment="#")
    inp["shas"]["YA_RULE62"] = SA.sha_file(REP / "YA_RULE62.csv")
    return inp


def pair_inputs(inp: dict, st=None) -> tuple:
    st = st or FE.load_store()
    D, diag = FE.pair_data(st, inp["d"], inp["x1"]["rv"].to_numpy(float), inp["x1"]["F"].to_numpy(float), phi=FE.phi_all(st), pbar=FE.profile_all(st))
    pr, v = st["present"], np.exp(st["lvol"][st["present"]])
    diag.update(vol_min=int(round(v.min())), vol_max=int(round(v.max())), w4_dev=float(abs(diag.pop("mean_w4") - 1.0)))
    return D, diag, st


def compute(inp: dict | None = None) -> tuple:
    """-> (прогнозы {Y3, Y4}, таблицы, числа)."""
    inp = inp or load()
    D, diag, st = pair_inputs(inp)
    ys = {y: W.walk(inp["x1"], D, y) for y in EV.YB}
    s = {**inp["ser"], **{y: f["F"].to_numpy(float) for y, f in ys.items()}}
    t = EV.main_tables(inp["rv"], s, inp["tp"], inp["d"], inp["x1"]["s"].to_numpy(int), ys, inp["rule_a"])
    num = dict(n_dates=int(len(inp["d"])), first=str(inp["d"][0].date()), last=str(inp["d"][-1].date()), pairs_at_activation=W.check_activation(inp["d"], inp["tp"]), min_pairs=W.MIN_PAIRS,
               n_inactive=int((ys["Y3"]["active"] == 0).sum()), seeds=list(W.FT.SEEDS), input_sha256=inp["shas"], configs_spent=2, n_holm=SA.EV.N_HOLM, n_wide=SA.EV.N_WIDE, lams=list(W.FT.FT.LAMS), **diag)
    return ys, t, num


def condition(t: dict) -> dict:
    c = t["CONDITION"].set_index("item")["rule62"]
    return dict(condition=str(c["условие Y5/Y6"]), rule62_vs_X1={"Y1": str(c["Y1"]), "Y3": str(c["Y3"])}, rule="правило 6.2 против X1 = да у Y1 или у Y3",
                note="Y5/Y6 не запускаются без отдельного разрешения владельца (Ж-243)")


def outputs_hash() -> str:
    return SA.digest(SA.WATCH + [REP, OUT, YA_OUT])


def write() -> None:
    R.guard("Y3"), R.guard("Y4")
    at = R.now_text()
    ys, t, num = compute()
    OUT.mkdir(parents=True, exist_ok=True)
    for y, f in ys.items():
        f.to_parquet(OUT / (y.lower() + ".parquet"), index=False)
    SA.merge_cp("YB", dict(sha256={y: SA.sha_file(OUT / (y.lower() + ".parquet")) for y in ys}, computed_at=at, n=int(len(ys["Y3"])), code=code_hash(), columns=W.OUT_COLS, configs=list(ys)))
    R.guard("Y3", computed_at=at), R.guard("Y4", computed_at=at)
    SA.merge_cp("YB.eval", dict(numbers=SA._json(SA.eval_numbers(t)), run=SA._json(num), written_at=R.now_text()))
    for k, v in TABLES.items():
        (REP / (k + ".csv")).write_text(csv_text(t[v]), encoding="utf-8")
    REPORT.write_text(DOC.build(t, num), encoding="utf-8")
    SA.merge_cp("YB.rules", dict(rules=SA._json(SA.rules(t)), sha256={k: SA.sha_file(REP / (k + ".csv")) for k in TABLES}, sha256_report=SA.sha_file(REPORT), written_at=R.now_text(), code=code_hash()))
    SA.merge_cp("YB.condition", dict(**condition(t), written_at=R.now_text()))


def _check_files(t, num, ys, ck, ev, ru, cd, bad) -> None:
    for y, f in ys.items():
        p = OUT / (y.lower() + ".parquet")
        if not p.exists() or SA.sha_file(p) != ck["sha256"][y] or not SA._same(pd.read_parquet(p), f):
            bad.append("%s.parquet ≠ пересчёту / sha256 ≠ CHECKPOINT YB" % y.lower())
    if not SA._close(SA._json(SA.eval_numbers(t)), ev["numbers"]) or not SA._close(SA._json(num), ev["run"]):
        bad.append("YB.eval ≠ пересчёту")
    if not SA._close(SA._json(SA.rules(t)), ru["rules"]):
        bad.append("YB.rules ≠ пересчёту")
    if {k: v for k, v in cd.items() if k != "written_at"} != condition(t):
        bad.append("YB.condition ≠ пересчёту")
    for k, v in TABLES.items():
        p = REP / (k + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(t[v]):
            bad.append("%s.csv ≠ пересчёту" % k)
        elif ru["sha256"].get(k) != SA.sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % k)
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != DOC.build(t, num):
        bad.append("YB_REPORT.md ≠ генератору")


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0)."""
    before, bad, cp = outputs_hash(), [], SA.read_cp()
    ck, ev, ru, cd = (cp.get(k) for k in ("YB", "YB.eval", "YB.rules", "YB.condition"))
    if not (ck and ev and ru and cd):
        return ["нет ключей YB / YB.eval / YB.rules / YB.condition в CHECKPOINT"]
    for y in ("Y3", "Y4"):
        try:
            R.guard(y, computed_at=ck["computed_at"])
        except SystemExit as e:
            bad.append(str(e))
    ts = [R._parse(x) for x in (ck["computed_at"], ev["written_at"], ru["written_at"], cd["written_at"])]
    if ts != sorted(ts) or "p_holm" in json.dumps(ev["numbers"]) or "p_one" in json.dumps(ev["numbers"]):
        bad.append("порядок/состав ключей: YB.eval должен быть записан раньше YB.rules и без p")
    ys, t, num = compute()
    _check_files(t, num, ys, ck, ev, ru, cd, bad)
    if outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad


def run(do_check: bool) -> int:
    if not do_check:
        write()
        print("YB: y3/y4.parquet, YB_*.csv и YB_REPORT.md записаны; CHECKPOINT ключи YB, YB.eval, YB.rules, YB.condition")
        return 0
    before, bad = outputs_hash(), check()
    print("\n".join(bad) if bad else "OK: Y3/Y4 пересчитаны и равны файлам, YB_*.csv и отчёт = генератор, реестр до прогноза, ключ YB.eval раньше таблиц с p; выходы V6/V7/DE/P0/T0/YA и YB не изменены (хэш %s)" % before[:16])
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Лист Y-B: Y3, Y4 (Y5/Y6 не запускаются)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stage", default="YB", choices=["YB", "Y5", "Y6"])
    a = ap.parse_args(argv)
    if a.stage in ("Y5", "Y6"):
        R.guard(a.stage)
    return run(a.check)


if __name__ == "__main__":
    raise SystemExit(main())
