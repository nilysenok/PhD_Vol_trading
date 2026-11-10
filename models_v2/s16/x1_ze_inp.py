"""Лист ZE, входы и счёт в памяти (Ж-247 п. 4): load() — ряды T0, Y1–Y3 (через `x1_z_stage.load`, sha256 по CHECKPOINT YA/YB) и ZA, ZB, ZB0 из файлов Z (sha256 = CHECKPOINT Z); pre_numbers() — G_d и предусловие;
compute() — прогнозы ZE, ZEp, таблицы, числа запуска. Файлов не пишет. Входы только читаются, не пересчитываются. Данные после 2026-02-27 (`data/fwd/`) не читаются."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым, до models_v2.s14)
from models_v2.s16 import x1_reg as R, x1_ya_stage as SA, x1_z_feat as ZF, x1_z_stage as ZS, x1_ze_crit as CR, x1_ze_feat as FE, x1_ze_reg as ZR, x1_ze_walk as W

ROOT, REP = R.ROOT, R.REP
OUT, CP_FILE, REPORT = ROOT / "data/regen/harj_x1/ZE", REP / "CHECKPOINT.json", REP / "ZE_REPORT.md"
HEAD = "# диагностика/оценка, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; ZE — после просмотра Y1–Y4, Z, ZD; Ж-247, лист ZE (ZE, ZEp; SBER, h = 1); код ze %s\n"
read_cp, sha_file = SA.read_cp, SA.sha_file


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("x1_ze_*.py")) + [Path(R.__file__)]:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def merge_cp(key: str, val: dict) -> None:
    """Слияние: чужие ключи сохраняются."""
    cp = read_cp()
    cp[key] = val
    CP_FILE.write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def load() -> dict:
    inp, cp = ZS.load(), read_cp().get("Z")
    if not cp:
        raise SystemExit("нет ключа Z в CHECKPOINT: лист Z не выполнен")
    for k in ("ZA", "ZB", "ZB0"):
        p = ZS.OUT / (k.lower() + ".parquet")
        if sha_file(p) != cp["sha256"][k]:
            raise SystemExit("%s.parquet листа Z ≠ sha256 в CHECKPOINT Z" % k.lower())
        z = pd.read_parquet(p)
        if not pd.DatetimeIndex(z["d"]).equals(inp["d"]):
            raise SystemExit("%s: даты ≠ общим" % k)
        inp["ser"][k], inp["shas"][k.lower() + "_parquet"] = z["F"].to_numpy(float), sha_file(p)
    return inp


def pre_numbers(inp: dict, st: dict | None = None) -> tuple:
    """-> (G, число вырожденных дней, предусловие) по входам: цели t⁺ в предусловие не входят."""
    st = st or ZF.load_store()
    x1, d = inp["x1"], inp["d"]
    G, bad = FE.gap(st, d, x1["F"].to_numpy(float))
    x, v, _ = ZF.x_v(st, d)
    return G, bad, FE.precondition(G, v, inp["ft"]["s"], x, x1["s"].to_numpy(int), inp["tp"])


def notes() -> list:
    return ["Трактовка строже плана: L2-штраф λ(b² + a² + c²) на все три параметра (план: «L2 на (b, a, c)»); c ∈ [0, 1] — граница L-BFGS-B; |b|, |a| ≤ 5.",
            "«Без 2022» = все t⁺, кроме 2022 года (включая 2026); 2023–2025 = годы t⁺ 2023, 2024, 2025. Диагностика, не критерий.",
            "K3 читается буквально «Δ к X1 ≤ 0»: год с F = X1 побитно засчитывается; число лет со строгим Δ < 0 — колонка strict_years_neg в ZE_CRIT.csv. K2: состояние N за весь период включает дни до включения (Δ = 0 там).",
            "K6: Δ(ZE − ZB) в состоянии E за 2023–2025 — разность QLIKE на днях e_d = 1 с t⁺ 2023–2025, в % QLIKE ZB этих дней; K6 и 6.2 считаются независимо от K1–K5.",
            "Холм: N = %d, N_wide = %d, семейство — все пары окна (как YA/YB/Z); базы ZE: X1, K2, K3, NTP2, cfg8, Y2, Y3, ZB, ZB0, ZA и контраст ZE против ZEp. Реестр — в ALL_TRIALS.csv (не в отдельном файле)." % (CR.N_HOLM, CR.N_WIDE),
            "Ключ `ZE.rules` (p-значения, запись после таблиц) добавлен сверх списка задания: порядок записи `ZE` → `ZE.eval` (без p) → таблицы → `ZE.rules`, как у Z."]


def compute(inp: dict | None = None) -> tuple:
    """-> (прогнозы {ZE, ZEp}, таблицы, числа)."""
    inp = inp or load()
    x1, d, tp, e = inp["x1"], inp["d"], inp["tp"], inp["x1"]["s"].to_numpy(int)
    G, bad, _ = pre_numbers(inp)
    D = dict(u=inp["rv"] / x1["F"].to_numpy(float), s=np.asarray(inp["ft"]["s"], float), e=e, G=G)
    zs = {k: W.walk(x1, D, k) for k in W.KINDS}
    s = {**inp["ser"], **{k: f["F"].to_numpy(float) for k, f in zs.items()}}
    t = CR.main_tables(inp["rv"], s, tp, d, e, zs)
    pre, reg = read_cp()["ZE.precondition"], R.read().set_index("id")["registered_at"]
    pool = W.closed_pool(tp, d, W.ACTIVATION)
    num = dict(n_dates=int(len(d)), first=str(d[0].date()), last=str(d[-1].date()), pairs_at_activation=W.ZW.check_activation(d, tp), min_pairs=W.MIN_PAIRS, n_ext_at_activation=int(W.FT.ext_pool(pool, e).size), min_ext=W.FT.MIN_EXT,
               n_inactive={k: int((f["active"] == 0).sum()) for k, f in zs.items()}, n_extreme_days=int(e.sum()), n_degenerate_days=int(bad), lams=list(W.FT.LAMS), v_ext=W.FT.V_EXT, n_holm=CR.N_HOLM, n_wide=CR.N_WIDE,
               input_sha256=inp["shas"], configs_spent=len(ZR.IDS), registered_at="; ".join("%s %s" % (c, reg[c]) for c in ZR.IDS), precondition=pre["numbers"], precondition_at=pre["written_at"], notes=notes())
    return zs, t, num
