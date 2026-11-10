"""Этап Z, стадия: compute() (всё в памяти), write() (ПИШЕТ za/zb/zb0/zc/zcp.parquet, CHECKPOINT `Z` → `Z.eval` (БЕЗ p) → таблицы Z_*.csv с p → отчёт → `Z.rules`), check() (ТОЛЬКО ЧИТАЕТ: пересчёт, сверка, хэш выходов до и после).
Вход: `python3 -W ignore -m models_v2.s16.x1_z_stage [--check]`. Расходуется 5 конфигураций (ZA, ZB, ZB0, ZC, ZCp; реестр — `x1_z_reg`, запись ОТДЕЛЬНОЙ командой до прогноза). Входы (X1, K1–K3, NTP2, A2, cfg6, cfg8, склад M10 V7, Y1/Y2/Y3) только читаются;
sha256 Y1/Y2/Y3 сверяются с CHECKPOINT `YA`/`YB`. Хэш выходов чужих листов не включает файлы `ZD_*` и ключи `ZD*` CHECKPOINT (параллельный лист ZD пишет свои)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым, до models_v2.s14)
from models_v2.s16 import x1_reg as R, x1_ya_stage as SA, x1_yb_stage as SB, x1_z_crit as CR, x1_z_doc as DOC, x1_z_eval as EZ, x1_z_feat as FE, x1_z_fit as FT, x1_z_reg as ZR, x1_z_walk as W

ROOT, REP = R.ROOT, R.REP
OUT, CP_FILE, REPORT = ROOT / "data/regen/harj_x1/Z", REP / "CHECKPOINT.json", REP / "Z_REPORT.md"
TABLES = {"Z_%s" % k: k for k in ("METRICS", "PAIRS", "RULE62", "YEARS", "STATES", "CELLS", "GO", "MCS", "STRESS", "CRIT", "PARAMS")}
HEAD = "# диагностика/оценка, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; ZB, ZC — после просмотра Y1–Y4; Ж-246, этап Z (ZA, ZB, ZB0, ZC, ZCp; SBER, h = 1); код z %s\n"
WATCH, YA_OUT, YB_OUT = SA.WATCH, SA.OUT, SB.OUT


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("x1_z_*.py")) + [Path(R.__file__)]:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def merge_cp(key: str, val: dict) -> None:
    """Слияние: чужие ключи сохраняются."""
    cp = json.loads(CP_FILE.read_text(encoding="utf-8")) if CP_FILE.exists() else {}
    cp[key] = val
    CP_FILE.write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def read_cp() -> dict:
    return json.loads(CP_FILE.read_text(encoding="utf-8")) if CP_FILE.exists() else {}


def load() -> dict:
    """Входы Y-B (SB.load: ряды T0, Y1/Y2 по sha) + Y3 из файла Y-B (sha256 = CHECKPOINT YB)."""
    inp = SB.load()
    ck = SA.read_cp().get("YB")
    p = YB_OUT / "y3.parquet"
    if not ck or SA.sha_file(p) != ck["sha256"]["Y3"]:
        raise SystemExit("y3.parquet листа Y-B ≠ sha256 в CHECKPOINT YB")
    y3 = pd.read_parquet(p)
    if not pd.DatetimeIndex(y3["d"]).equals(inp["d"]):
        raise SystemExit("Y3: даты ≠ общим")
    inp["ser"]["Y3"], inp["shas"]["y3_parquet"] = y3["F"].to_numpy(float), SA.sha_file(p)
    return inp


def notes() -> list:
    return ["Граница γ0 плана не задаёт: взята |γ0| ≤ %g, как у γ (старт γ0 = %g)." % (FT.BOUND_G, FT.START_G0),
            "Штраф L2 λ: ZA, ZB — на (b, a); ZB0 — на b; ZC, ZCp — на γ, b, a без γ0. Стандартизация z̃ — среднее и σ (ddof = 0) по парам пула; σ ≈ 0 → делитель 1.",
            "ZA: (b, a) обучаются при ω = 1 на пуле без блока валидации, ω выбирается по потере блока (ничья: больший λ, затем меньший ω), затем (b, a) переобучаются на всём пуле при выбранном λ; g = ω·(b + a·s).",
            "ZCp: z_{d−%d} циклически (np.roll) — первые %d строк получают z конца ряда; контроль, критериев не получает (план: поправка b + a·s — с настоящим s_d)." % (FE.SHIFT, FE.SHIFT),
            "K3 читается буквально «Δ к X1 ≤ 0»: год с F = X1 побитно засчитывается; число лет со строгим Δ < 0 — колонка strict_years_neg в Z_CRIT.csv. K2: состояние N за весь период включает дни до включения (Δ = 0 там).",
            "Холм: N = %d, N_wide = %d, семейство — все пары окна (как YA/YB). Реестр — в ALL_TRIALS.csv (хэш этого файла в сверку P0/YA/YB не входит), не в отдельном Z_TRIALS.csv." % (EZ.N_HOLM, EZ.N_WIDE)]


def compute(inp: dict | None = None) -> tuple:
    """-> (прогнозы {ZA…ZCp}, таблицы, числа)."""
    inp = inp or load()
    x1, d, tp, e = inp["x1"], inp["d"], inp["tp"], inp["x1"]["s"].to_numpy(int)
    x, v, bad = FE.x_v(FE.load_store(), d)
    s_d = np.asarray(inp["ft"]["s"], float)
    D = dict(u=inp["rv"] / x1["F"].to_numpy(float), s=s_d, e=e, Z=FE.raw_z(e, x, s_d, v))
    zs = {k: W.walk(x1, D, k) for k in W.KINDS}
    s = {**inp["ser"], **{k: f["F"].to_numpy(float) for k, f in zs.items()}}
    t = CR.main_tables(inp["rv"], s, tp, d, e, zs)
    reg = ZR.R.read().set_index("id")["registered_at"]
    pool = W.closed_pool(tp, d, W.ACTIVATION)
    num = dict(n_dates=int(len(d)), first=str(d[0].date()), last=str(d[-1].date()), pairs_at_activation=W.check_activation(d, tp), min_pairs=W.MIN_PAIRS, n_ext_at_activation=int(FT.ext_pool(pool, e).size), min_ext=FT.MIN_EXT,
               n_inactive={k: int((f["active"] == 0).sum()) for k, f in zs.items()}, n_extreme_days=int(e.sum()), n_degenerate_days=int(bad), lams=list(FT.LAMS), omegas=list(FT.OMEGAS), v_full=FT.V_FULL, v_ext=FT.V_EXT,
               n_holm=EZ.N_HOLM, n_wide=EZ.N_WIDE, input_sha256=inp["shas"], configs_spent=len(ZR.IDS), registered_at="; ".join("%s %s" % (c, reg[c]) for c in ZR.IDS), notes=notes())
    return zs, t, num


def eval_numbers(t: dict) -> dict:
    """`Z.eval`: QLIKE по окнам, Δ с интервалами, годы, состояния, клетки, K1–K5 — БЕЗ p-значений."""
    m, pr, c, st = t["METRICS"], t["PAIRS"], t["CELLS"], t["STATES"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    pair = {"%s|%s|%s" % (r.y, r.base, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg), int(r.n_years)] for r in pr.itertuples()}
    states = {"%s|%s|%s" % (r.y, r.state, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), int(r.n), float(r.mean_g), float(r.mean_gate)] for r in st.itertuples()}
    cells = {"%s|%s|%s" % (r.cell, r.y, r.base): [float(r.delta_q), float(r.delta_pct), float(r.lo90), float(r.hi90), int(r.n)] for r in c.itertuples()}
    crit = {"%s|%s" % (r.y, r.crit): [r.name, float(r.value), bool(r.passed)] for r in t["CRIT"].itertuples()}
    yrs = {"%s|%s|%d" % (r.y, r.base, r.year): float(r.delta_pct) for r in t["YEARS"].itertuples()}
    return dict(qlike=ql, pairs=pair, states=states, cells=cells, crit=crit, years=yrs, rule62={r.y: r.rule62 for r in t["RULE62"].itertuples()}, p_values="не входят: записаны в Z.rules после этого ключа")


def rules(t: dict) -> dict:
    pr = t["PAIRS"]
    return dict(p={"%s|%s|%s" % (r.y, r.base, r.window): dict(p_one=float(r.p_one), p_holm_N=float(r.p_holm_N), p_holm_Nwide=float(r.p_holm_Nwide)) for r in pr.itertuples()},
                rule62={r.y: dict(rule62=r.rule62, p_dm_one=float(r.p_dm_one), p_holm_N=float(r.p_holm_N), confirmed=bool(r.rule62 == "да" and r.p_holm_N < 0.05)) for r in t["RULE62"].itertuples()})


def rep_digest() -> str:
    """Хэш docs/otchety/harj_x1 без файлов ZD_* и ключей ZD* CHECKPOINT (их пишет параллельный лист ZD)."""
    h = hashlib.sha256()
    for f in sorted(p for p in REP.rglob("*") if p.is_file() and not p.name.startswith("ZD_")):
        if f.name == "CHECKPOINT.json":
            cp = {k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items() if not k.startswith("ZD")}
            h.update(str(f).encode() + json.dumps(cp, sort_keys=True).encode())
        else:
            h.update(str(f).encode() + SA.sha_file(f).encode())
    return h.hexdigest()


def outputs_hash() -> str:
    return SA.digest(WATCH + [OUT, YA_OUT, YB_OUT]) + rep_digest()


def write() -> None:
    for c in ZR.IDS:
        ZR.guard(c)
    at = R.now_text()
    zs, t, num = compute()
    OUT.mkdir(parents=True, exist_ok=True)
    for k, f in zs.items():
        f.to_parquet(OUT / (k.lower() + ".parquet"), index=False)
    merge_cp("Z", dict(sha256={k: SA.sha_file(OUT / (k.lower() + ".parquet")) for k in zs}, computed_at=at, n=int(len(zs["ZA"])), code=code_hash(), columns=W.OUT_COLS, configs=list(zs)))
    for c in ZR.IDS:
        ZR.guard(c, computed_at=at)
    merge_cp("Z.eval", dict(numbers=SA._json(eval_numbers(t)), run=SA._json(num), written_at=R.now_text()))
    for k, v in TABLES.items():
        (REP / (k + ".csv")).write_text(csv_text(t[v]), encoding="utf-8")
    REPORT.write_text(DOC.build(t, num), encoding="utf-8")
    merge_cp("Z.rules", dict(rules=SA._json(rules(t)), sha256={k: SA.sha_file(REP / (k + ".csv")) for k in TABLES}, sha256_report=SA.sha_file(REPORT), written_at=R.now_text(), code=code_hash()))


def _check_files(t, num, zs, ck, ev, ru, bad) -> None:
    for k, f in zs.items():
        p = OUT / (k.lower() + ".parquet")
        if not p.exists() or SA.sha_file(p) != ck["sha256"][k] or not SA._same(pd.read_parquet(p), f):
            bad.append("%s.parquet ≠ пересчёту / sha256 ≠ CHECKPOINT Z" % k.lower())
    if not SA._close(SA._json(eval_numbers(t)), ev["numbers"]) or not SA._close(SA._json(num), ev["run"]):
        bad.append("Z.eval ≠ пересчёту")
    if not SA._close(SA._json(rules(t)), ru["rules"]):
        bad.append("Z.rules ≠ пересчёту")
    for k, v in TABLES.items():
        p = REP / (k + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(t[v]):
            bad.append("%s.csv ≠ пересчёту" % k)
        elif ru["sha256"].get(k) != SA.sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % k)
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != DOC.build(t, num):
        bad.append("Z_REPORT.md ≠ генератору")


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0)."""
    before, bad, cp = outputs_hash(), [], read_cp()
    ck, ev, ru = cp.get("Z"), cp.get("Z.eval"), cp.get("Z.rules")
    if not (ck and ev and ru):
        return ["нет ключей Z / Z.eval / Z.rules в CHECKPOINT"]
    for c in ZR.IDS:
        try:
            ZR.guard(c, computed_at=ck["computed_at"])
        except SystemExit as e:
            bad.append(str(e))
    ts = [R._parse(x) for x in (ck["computed_at"], ev["written_at"], ru["written_at"])]
    if ts != sorted(ts) or "p_holm" in json.dumps(ev["numbers"]) or "p_one" in json.dumps(ev["numbers"]):
        bad.append("порядок/состав ключей: Z.eval должен быть записан раньше Z.rules и без p")
    zs, t, num = compute()
    _check_files(t, num, zs, ck, ev, ru, bad)
    if outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad


def run(do_check: bool) -> int:
    if not do_check:
        write()
        print("Z: za/zb/zb0/zc/zcp.parquet, Z_*.csv и Z_REPORT.md записаны; CHECKPOINT ключи Z, Z.eval, Z.rules")
        return 0
    before, bad = outputs_hash(), check()
    print("\n".join(bad) if bad else "OK: ZA/ZB/ZB0/ZC/ZCp пересчитаны и равны файлам, Z_*.csv и отчёт = генератор, реестр до прогноза, ключ Z.eval раньше таблиц с p; выходы V6/V7/DE/P0/T0/YA/YB не изменены (хэш %s)" % before[:16])
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Этап Z: ZA, ZB, ZB0, ZC, ZCp (Y5/Y6 и любые сверх пяти — отказ)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stage", default="Z", choices=["Z", "Y5", "Y6"])
    a = ap.parse_args(argv)
    if a.stage != "Z":
        ZR.guard(a.stage)
    return run(a.check)


if __name__ == "__main__":
    raise SystemExit(main())
