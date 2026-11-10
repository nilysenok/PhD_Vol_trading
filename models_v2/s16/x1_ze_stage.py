"""Лист ZE (Ж-247 п. 4), стадия: `--precondition` (ключ `ZE.precondition` ДО реестра; при max |corr| > 0,8 — «условие ложно», ZE не считается), write() (ПИШЕТ ze/zep.parquet, CHECKPOINT `ZE` → `ZE.eval` (БЕЗ p) → таблицы ZE_*.csv с p → отчёт → `ZE.rules`),
check() (ТОЛЬКО ЧИТАЕТ: пересчёт, сверка с файлами, хэш чужих выходов до и после). Вход: `python3 -W ignore -m models_v2.s16.x1_ze_stage [--precondition|--check]`. Реестр — `x1_ze_reg`, ОТДЕЛЬНОЙ командой между предусловием и прогнозом.
Счёт — `x1_ze_inp`, числа и хэши — `x1_ze_num`. Расходуются 2 конфигурации (ZE, ZEp); Y5, Y6 и любые сверх — отказ (`x1_ze_reg.guard`)."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from models_v2.s16 import x1_reg as R, x1_ya_stage as SA, x1_ze_doc as DOC, x1_ze_reg as ZR, x1_ze_walk as W
from models_v2.s16.x1_ze_inp import CP_FILE, OUT, REP, REPORT, code_hash, compute, csv_text, load, merge_cp, pre_numbers, read_cp, sha_file
from models_v2.s16.x1_ze_num import eval_numbers, outputs_hash, rules

TABLES = {"ZE_%s" % k: k for k in ("METRICS", "PAIRS", "RULE62", "YEARS", "STATES", "MCS", "STRESS", "CRIT", "DIAG", "PARAMS")}


def precondition() -> int:
    """Считает G_d и корреляции, пишет `ZE.precondition` (слиянием). -> 0 — условие выполнено, 2 — условие ложно (стоп)."""
    _, bad, pre = pre_numbers(load())
    merge_cp("ZE.precondition", dict(numbers=SA._json(pre), n_degenerate_days=int(bad), ok=bool(pre["ok"]), written_at=R.now_text(), code=code_hash()))
    print("ZE.precondition: n = %d, corr(G, v_d) = %.4f, corr(G, s_d) = %.4f, corr(G, x_d) = %.4f, max |corr| = %.4f → %s" % (
        pre["n"], pre["corr_G_v"], pre["corr_G_s"], pre["corr_G_x_info"], pre["max_abs"], "условие выполнено" if pre["ok"] else "условие ЛОЖНО: ZE дубль, не регистрировать и не считать"))
    return 0 if pre["ok"] else 2


def write() -> None:
    if not ZR.precondition_ok(read_cp()):
        raise SystemExit("ZE не считается: нет ключа ZE.precondition (ok) — условие дубля не проверено или ложно")
    for c in ZR.IDS:
        ZR.guard(c)
    at = R.now_text()
    zs, t, num = compute()
    OUT.mkdir(parents=True, exist_ok=True)
    for k, f in zs.items():
        f.to_parquet(OUT / (k.lower() + ".parquet"), index=False)
    merge_cp("ZE", dict(sha256={k: sha_file(OUT / (k.lower() + ".parquet")) for k in zs}, computed_at=at, n=int(len(zs["ZE"])), code=code_hash(), columns=W.OUT_COLS, configs=list(zs)))
    for c in ZR.IDS:
        ZR.guard(c, computed_at=at)
    merge_cp("ZE.eval", dict(numbers=SA._json(eval_numbers(t)), run=SA._json(num), written_at=R.now_text()))
    for k, v in TABLES.items():
        (REP / (k + ".csv")).write_text(csv_text(t[v]), encoding="utf-8")
    REPORT.write_text(DOC.build(t, num), encoding="utf-8")
    merge_cp("ZE.rules", dict(rules=SA._json(rules(t)), sha256={k: sha_file(REP / (k + ".csv")) for k in TABLES}, sha256_report=sha_file(REPORT), written_at=R.now_text(), code=code_hash()))


def _check_files(t, num, zs, ck, ev, ru, bad) -> None:
    for k, f in zs.items():
        p = OUT / (k.lower() + ".parquet")
        if not p.exists() or sha_file(p) != ck["sha256"][k] or not SA._same(pd.read_parquet(p), f):
            bad.append("%s.parquet ≠ пересчёту / sha256 ≠ CHECKPOINT ZE" % k.lower())
    if not SA._close(SA._json(eval_numbers(t)), ev["numbers"]) or not SA._close(SA._json(num), ev["run"]):
        bad.append("ZE.eval ≠ пересчёту")
    if not SA._close(SA._json(rules(t)), ru["rules"]):
        bad.append("ZE.rules ≠ пересчёту")
    for k, v in TABLES.items():
        p = REP / (k + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(t[v]):
            bad.append("%s.csv ≠ пересчёту" % k)
        elif ru["sha256"].get(k) != sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % k)
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != DOC.build(t, num):
        bad.append("ZE_REPORT.md ≠ генератору")


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0)."""
    before, bad, cp = outputs_hash(), [], read_cp()
    pre, ck, ev, ru = cp.get("ZE.precondition"), cp.get("ZE"), cp.get("ZE.eval"), cp.get("ZE.rules")
    if not (pre and ck and ev and ru):
        return ["нет ключей ZE.precondition / ZE / ZE.eval / ZE.rules в CHECKPOINT"]
    _, _, now_pre = pre_numbers(load())
    if not SA._close(SA._json(now_pre), pre["numbers"]) or not pre["ok"]:
        bad.append("ZE.precondition ≠ пересчёту или условие ложно")
    for c in ZR.IDS:
        try:
            ZR.guard(c, computed_at=ck["computed_at"])
            ZR.guard(c, computed_at=ev["written_at"])
        except SystemExit as e:
            bad.append(str(e))
    reg = R.read().set_index("id")["registered_at"]
    ts = [R._parse(x) for x in (pre["written_at"], min(reg[c] for c in ZR.IDS), ck["computed_at"], ev["written_at"], ru["written_at"])]
    if ts != sorted(ts) or "p_holm" in json.dumps(ev["numbers"]) or "p_one" in json.dumps(ev["numbers"]):
        bad.append("порядок ключей: ZE.precondition ≤ реестр ≤ ZE ≤ ZE.eval (без p) ≤ ZE.rules")
    zs, t, num = compute()
    _check_files(t, num, zs, ck, ev, ru, bad)
    if outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad


def run(do_check: bool) -> int:
    if not do_check:
        write()
        print("ZE: ze/zep.parquet, ZE_*.csv и ZE_REPORT.md записаны; CHECKPOINT ключи ZE, ZE.eval, ZE.rules")
        return 0
    before, bad = outputs_hash(), check()
    print("\n".join(bad) if bad else "OK: ZE/ZEp пересчитаны и равны файлам, ZE_*.csv и отчёт = генератор, предусловие и реестр до прогноза, ключ ZE.eval раньше таблиц с p; выходы V6/V7/DE/P0/T0/YA/YB/Z/ZD/Z2 не изменены (хэш %s)" % before[:16])
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Лист ZE: ZE, ZEp (Y5, Y6 и любые сверх двух — отказ)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--precondition", action="store_true")
    ap.add_argument("--stage", default="ZE", choices=["ZE", "Y5", "Y6"])
    a = ap.parse_args(argv)
    if a.stage != "ZE":
        ZR.guard(a.stage)
    return precondition() if a.precondition else run(a.check)


if __name__ == "__main__":
    raise SystemExit(main())
