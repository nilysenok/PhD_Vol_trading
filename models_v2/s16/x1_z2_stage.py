"""Этап Z2, стадия (Ж-246 п. 6). Порядок СТРОГИЙ: реестр H1, Hc, H2 (`x1_z2_reg`, ОТДЕЛЬНОЙ командой до счёта) → H0 → HD → условие L1–L3 → CHECKPOINT `Z2.condition` → ТОЛЬКО при истинном условии H1 и Hc → `Z2` → `Z2.eval` (БЕЗ p) → таблицы с p → `Z2.rules`.
H2 не считается никогда (`ZR.guard` — отказ). write() пишет файлы, check() ТОЛЬКО ЧИТАЕТ (пересчёт = файлы; хэш чужих выходов, включая Z и ZD, до и после). Вход: `python3 -W ignore -m models_v2.s16.x1_z2_stage [--check]`."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым)
from models_v2.s16 import x1_reg as R, x1_ya_stage as SA, x1_z2_cond as CD, x1_z2_doc as DOC, x1_z2_ev as EV, x1_z2_h0 as H0, x1_z2_hd as HD, x1_z2_io as IO, x1_z2_reg as ZR, x1_z2_run as RUN, x1_z_walk as W

BASE_T = {"Z2_H0_EVENTS": "H0_EVENTS", "Z2_H0_LR": "H0_LR", "Z2_H0_DUP": "H0_DUP", "Z2_HD": "HD", "Z2_CONDITION": "CONDITION"}
REP, OUT, ZOUT, REPORT = IO.REP, IO.OUT, IO.ZOUT, IO.REPORT
read_cp, merge_cp, code_hash, csv_text, foreign_hash, read_table = IO.read_cp, IO.merge_cp, IO.code_hash, IO.csv_text, IO.foreign_hash, IO.read_table


def compute_h0() -> tuple:
    """-> (таблицы H0/HD/условие, числа, словарь условия). Цель RV не читается; ZB — сохранённый ряд (sha256 = CHECKPOINT Z)."""
    from models_v2.s15 import de0_data as D, de_e1_state as S
    ctx, x1 = H0.real_ctx(), pd.read_parquet(SA.X1_FILE)
    d, days = pd.DatetimeIndex(x1["d"]), ctx["days"]
    ok = np.flatnonzero(ctx["base_inlik"])
    ident = bool(np.array_equal((ctx["N"][ok] > 0.5).any(1).astype(int), S.states(D.load_store(), days[ok])))
    if not ident:
        raise SystemExit("e_d ≠ 1[ΣN ≥ 1]: тождество нарушено")
    Dpos = [int(days.get_loc(d[p])) for p in W.month_starts(d) if d[p] >= W.ACTIVATION]
    mo, fits = H0.monthly(ctx, Dpos)
    full = H0.lr_full(ctx)
    lr = pd.concat([mo, pd.DataFrame([H0.row_of("2015_2021", "full", full)])], ignore_index=True)
    dup = H0.dup_table(ctx, *fits[Dpos[0]])
    zb, ck = pd.read_parquet(ZOUT / "zb.parquet"), read_cp().get("Z")
    if not ck or SA.sha_file(ZOUT / "zb.parquet") != ck["sha256"]["ZB"]:
        raise SystemExit("zb.parquet ≠ sha256 в CHECKPOINT Z")
    pos, Nb = days.get_indexer(pd.DatetimeIndex(zb["d"])), ctx["N"] > 0.5
    hd = HD.hd_table(zb, x1["s"].to_numpy(int), EV.last_tau(Nb)[pos], Nb.sum(1)[pos])
    c = CD.condition(read_table("Z_CRIT"), read_table("Z_PAIRS"), read_table("Z_STATES"), HD.hd_negative(hd), mo, {**full, **H0.HK.lr_test(full["ll"], full["ll0"])}, dup)
    reg = ZR.R.read().set_index("id")["registered_at"]
    t = {"H0_EVENTS": H0.events_table(ctx), "H0_LR": lr, "H0_DUP": dup, "HD": hd, "CONDITION": CD.table(c)}
    num = dict(identity=ident, identity_days=int(len(ok)), configs_spent=0, registered_at="; ".join("%s %s" % (i, reg[i]) for i in ZR.IDS), n_holm_z=749,
               input_sha256={"x1": SA.sha_file(SA.X1_FILE), "zb": SA.sha_file(ZOUT / "zb.parquet"), **{n: SA.sha_file(REP / (n + ".csv")) for n in ("Z_CRIT", "Z_PAIRS", "Z_STATES")}})
    return t, num, c


def table_diffs(tabs: dict, t: dict, rep, shas: dict) -> list:
    """Файлы таблиц = пересчёту (текст) и sha256 = CHECKPOINT; подменённое число в любой таблице (в т. ч. `Z2_CRIT.csv`) ловится."""
    bad = []
    for k, v in tabs.items():
        p = rep / (k + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(t[v]):
            bad.append("%s.csv ≠ пересчёту" % k)
        elif shas.get(k) != SA.sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % k)
    return bad


def order_ok(cp: dict) -> bool:
    """Z2.condition раньше Z2 раньше Z2.eval раньше Z2.rules; в Z2.eval нет p-значений."""
    ts = [R._parse(x) for x in (cp["Z2.condition"]["written_at"], cp["Z2"]["computed_at"], cp["Z2.eval"]["written_at"], cp["Z2.rules"]["written_at"])]
    return ts == sorted(ts) and "p_holm" not in json.dumps(cp["Z2.eval"]["numbers"]) and "p_one" not in json.dumps(cp["Z2.eval"]["numbers"])


def cond_keys(c: dict) -> dict:
    return dict(L1=c["L1"], L2a=c["L2a"], L2b=c["L2b"], L3=c["L3"], L3_detail=c["L3_detail"])


def h1_inputs() -> tuple:
    """Входы ветки H1/Hc: Y-B/Z загрузчик, ряды ZB, ZB0, ZC (sha256 = CHECKPOINT Z), события H1 и Hc."""
    from models_v2.s16 import x1_z_stage as ZS
    inp, ck, zser = ZS.load(), read_cp().get("Z"), {}
    for k in ("ZB", "ZB0", "ZC"):
        p = ZOUT / (k.lower() + ".parquet")
        if SA.sha_file(p) != ck["sha256"][k]:
            raise SystemExit("%s.parquet ≠ sha256 в CHECKPOINT Z" % k.lower())
        zser[k] = pd.read_parquet(p)["F"].to_numpy(float)
    return inp, zser, H0.real_ctx(), H0.real_ctx(True)


def run_h1(c: dict, t: dict, num: dict) -> None:
    """Бегун H1/Hc: ложное условие — отказ (SystemExit, ничего не считается); H2 — отказ всегда. Истинное — расчёт и запись `Z2`, `Z2.eval`, таблиц с p, `Z2.rules`; t, num дополняются."""
    if not c["value"]:
        raise SystemExit("H1, Hc: не запускалось, условие ложно (%s)" % ", ".join(c["reasons"] + c["l3_reasons"]))
    RUN.check_register()
    at = R.now_text()
    zs, t1, n1 = RUN.compute_h1(*h1_inputs())
    OUT.mkdir(parents=True, exist_ok=True)
    for k, f in zs.items():
        f.to_parquet(OUT / (k.lower() + ".parquet"), index=False)
    merge_cp("Z2", dict(sha256={k: SA.sha_file(OUT / (k.lower() + ".parquet")) for k in zs}, computed_at=at, n=int(len(zs["H1"])), code=code_hash(), columns=list(zs["H1"].columns)))
    for cfg in ZR.COUNTED:
        ZR.guard(cfg, computed_at=at)
    merge_cp("Z2.eval", dict(numbers=SA._json(RUN.eval_numbers(t1)), run=SA._json(n1), written_at=R.now_text()))
    for k, v in RUN.TABLES.items():
        (REP / (k + ".csv")).write_text(csv_text(t1[v]), encoding="utf-8")
    t.update(t1)
    num.update(n1)
    merge_cp("Z2.rules", dict(rules=SA._json(RUN.rules(t1)), sha256={k: SA.sha_file(REP / (k + ".csv")) for k in RUN.TABLES}, written_at=R.now_text(), code=code_hash()))


def write() -> None:
    for cfg in ZR.COUNTED:
        ZR.guard(cfg)
    t, num, c = compute_h0()
    for k, v in BASE_T.items():
        (REP / (k + ".csv")).write_text(csv_text(t[v]), encoding="utf-8")
    merge_cp("Z2.condition", dict(value=c["value"], reasons=c["reasons"] + c["l3_reasons"], numbers=SA._json(cond_keys(c)), run=SA._json(num), sha256={k: SA.sha_file(REP / (k + ".csv")) for k in BASE_T}, written_at=R.now_text(), code=code_hash()))
    try:
        run_h1(c, t, num)
    except SystemExit as e:
        print(e)
    REPORT.write_text(DOC.build(t, num), encoding="utf-8")
    merge_cp("Z2.report", dict(sha256_report=SA.sha_file(REPORT), written_at=R.now_text()))


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0). Ветка H1/Hc проверяется пересчётом, только если условие истинно."""
    before, bad, cp = foreign_hash(), [], read_cp()
    ck, rp = cp.get("Z2.condition"), cp.get("Z2.report")
    if not ck or not rp:
        return ["нет ключей Z2.condition / Z2.report в CHECKPOINT"]
    for cfg in ZR.COUNTED:
        try:
            ZR.guard(cfg)
        except SystemExit as e:
            bad.append(str(e))
    try:
        ZR.guard("H2")
        bad.append("H2 не отказан")
    except SystemExit:
        pass
    t, num, c = compute_h0()
    if not SA._close(SA._json(cond_keys(c)), ck["numbers"]) or not SA._close(SA._json(num), ck["run"]) or ck["value"] != c["value"]:
        bad.append("Z2.condition ≠ пересчёту")
    bad += table_diffs(BASE_T, t, REP, ck["sha256"])
    if c["value"]:
        bad += check_h1(cp, t, num)
    elif (OUT.exists() and any(OUT.iterdir())) or any(k in cp for k in ("Z2", "Z2.eval", "Z2.rules")):
        bad.append("условие ложно, но есть прогнозы или ключи оценки H1/Hc")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != DOC.build(t, num) or rp["sha256_report"] != SA.sha_file(REPORT):
        bad.append("Z2_REPORT.md ≠ генератору")
    if foreign_hash() != before:
        bad.append("--check изменил чужие выходы")
    return bad


def check_h1(cp: dict, t: dict, num: dict) -> list:
    bad, (z, ev, ru) = [], (cp.get("Z2"), cp.get("Z2.eval"), cp.get("Z2.rules"))
    if not (z and ev and ru):
        return ["условие истинно, но нет ключей Z2 / Z2.eval / Z2.rules"]
    if not order_ok(cp):
        bad.append("порядок ключей: Z2.condition → Z2 → Z2.eval (без p) → Z2.rules")
    zs, t1, n1 = RUN.compute_h1(*h1_inputs())
    for k, f in zs.items():
        p = OUT / (k.lower() + ".parquet")
        if not p.exists() or SA.sha_file(p) != z["sha256"][k] or not SA._same(pd.read_parquet(p), f):
            bad.append("%s.parquet ≠ пересчёту" % k.lower())
    if not SA._close(SA._json(RUN.eval_numbers(t1)), ev["numbers"]) or not SA._close(SA._json(RUN.rules(t1)), ru["rules"]):
        bad.append("Z2.eval / Z2.rules ≠ пересчёту")
    bad += table_diffs(RUN.TABLES, t1, REP, ru["sha256"])
    t.update(t1)
    num.update(n1)
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Этап Z2: H0, HD, условие L1–L3 (H1, Hc — только при истинном условии; H2 — никогда)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stage", default="Z2", choices=["Z2", "H2"])
    a = ap.parse_args(argv)
    if a.stage == "H2":
        ZR.guard("H2")
    if not a.check:
        write()
        print("Z2: Z2_H0_*.csv, Z2_HD.csv, Z2_CONDITION.csv, Z2_REPORT.md записаны; CHECKPOINT ключи Z2.condition, Z2.report")
        return 0
    before, bad = foreign_hash(), check()
    print("\n".join(bad) if bad else "OK: H0/HD/условие пересчитаны и равны файлам, отчёт = генератор, H2 отказан, чужие выходы не изменены (хэш %s)" % before[:16])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
