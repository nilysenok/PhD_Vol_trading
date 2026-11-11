"""S-Z-EVAL (Ж-258; `S_Z_SPEC.md`, разд. 7, 8): оценка S-RCKZ, S-RZ, S-RCKZs на D_M (t⁺ 2018-01-04…2026-11-10, n = 2198). Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, исследовательский результат на раскрытой истории, не OOS; механизм выбран после просмотра истории.
Читает СОХРАНЁННЫЕ прогнозы `data/regen/harj_struct/{Z,R,D,L}/*.parquet`, ряды X1, W5-G, K1–K3, B_mix и `S_Z_NULL.csv`; прогнозов и реестр не трогает. Обвязка — `a_eval_*`; здесь — `z_eval_cfg` (константы), `z_eval_tab`, `z_eval_gd`, `z_eval_par`.
Публичное API:
- `KEYS` — 12 таблиц `docs/otchety/harj_struct/S_Z_{TABLE,STATS,MECH,GUARD,CONC,YEARS,SPA,MCS,POWER,PARAMS,SEL,DEC}.csv` (`OUTS`); отчёт `S_Z_REPORT.md` — `z_doc`; `KEY` = "S.Z.eval".
- `load_inputs(ckpt)`, `compute(inp) -> dict` таблиц (H-Z1, H-Z2, условия разд. 7.3 (два набора) и Z-DEC выводятся из таблиц), `texts(t, tp)`, `merge_checkpoint`.
- `write()`: ключ `S.Z.eval` (written_at, хэши входов, выходов, кода) пишется в `docs/otchety/harj_x1/CHECKPOINT.json` ПЕРВЫМ (слиянием), затем таблицы с p. `check()` — только чтение: `z_eval_chk.problems`; sha256 выходов до и после.
Запуск: `python3 -W ignore -m models_v2.s18.z_eval [--write | --check]` (перед `--write` — `z_eval_null --run`)."""
from __future__ import annotations

import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import r_walk_ctx as RC
from models_v2.s18 import z_eval_cfg as C
from models_v2.s18 import z_eval_gd as GD
from models_v2.s18 import z_eval_null as ZN
from models_v2.s18 import z_eval_par as PAR
from models_v2.s18 import z_eval_tab as T
from models_v2.s18 import z_walk as ZW

ROOT, REP, CKPT, KEY = C.ROOT, C.REP, C.CKPT, C.KEY
KEYS = ("table", "stats", "mech", "guard", "conc", "years", "spa", "mcs", "power", "params", "sel", "dec")
OUTS = tuple("S_Z_%s.csv" % k.upper() for k in KEYS)
REPORT = "S_Z_REPORT.md"
MARK = "исследовательский результат на раскрытой истории, не OOS; механизм выбран после просмотра истории"
MARKS = (MARK, "критерии разд. 7 S_Z_SPEC.md объявлены и зарегистрированы до счёта; «подтверждено» только при p Холма (N программы) < 0,05 и p SPA < 0,05 по своей строке; замена кандидата — только решение владельца")


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def head(tp) -> str:
    tp = pd.DatetimeIndex(tp)
    return "\n".join("# " + m for m in MARKS) + "\n# D_M: t⁺ %s…%s, n = %d; l = RV/F − ln(RV/F) − 1, QLIKE = среднее дневных потерь (без обрезки); Δ в %% QLIKE базы, минус — лучше; DEVELOPMENT / HISTORICAL WALK-FORWARD\n" % (tp.min().date(), tp.max().date(), len(tp))


def load_inputs(ckpt: Path = CKPT) -> dict:
    inp = C.load_inputs(ckpt)
    if not ZN.OUT.exists():
        raise SystemExit("нет S_Z_NULL.csv: сначала `z_eval_null --run` (диагностика объявлена до QLIKE)")
    inp["null"] = ZN.read()
    return inp


def compute(inp: dict) -> dict:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = T.losses(rv, F)
    spa = T.spa_blocks(ls)
    tb, st = T.table(rv, tp, F, ls), T.stats(rv, tp, ls, spa, inp["N"], inp["N_wide"])
    yrs, cnc = GD.years(tp, ls), GD.conc(tp, ls)
    g = GD.guard(yrs, cnc)
    ctx, (zm, _) = RC.make_ctx("CK"), ZW.load_z()
    par = pd.concat([PAR.params(inp["frames"]), PAR.ztilde(inp["frames"], ctx, zm)], ignore_index=True)
    return dict(table=tb, stats=st, mech=PAR.mech(inp, ls, inp["null"]), guard=g, conc=cnc, years=yrs, spa=T.spa_table(spa), mcs=T.mcs(rv, F), power=T.power_table(tp, ls),
                params=par, sel=PAR.sel({k: inp["frames"][k] for k in C.REAL_Z}), dec=GD.dec_rows(tb, g, st), identity=GD.identity_gap(yrs))


def texts(t: dict, tp) -> dict:
    h = head(tp)
    return {n: h + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g") for k, n in zip(KEYS, OUTS)}


def inputs_sha() -> dict:
    ps = [str(p.relative_to(ROOT)) for p in C.FILES.values()] + ["data/regen/harj_struct/Z/zmaps.npz", "data/regen/harj_x1/W5/w5g.parquet", "docs/otchety/harj_struct/S_Z_SPEC.md", "docs/otchety/harj_struct/S_Z_NULL.csv"]
    return {p: _sha(ROOT / p) for p in ps}


def merge_checkpoint(ckpt: Path, t: dict, tx: dict, inp: dict, now: str | None = None) -> None:
    from models_v2.s18 import r_reg as RR
    text = ckpt.read_text(encoding="utf-8")
    cp = json.loads(text)
    if json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False) != text:
        raise SystemExit("CHECKPOINT.json: формат не indent 1 / sorted keys — слияние ключа сломало бы прочие ключи")
    code, d = Path(__file__).resolve().parent, t["dec"].set_index("id")
    cp[KEY] = dict(written_at=now or RR.now_text(), inputs=inputs_sha(), outputs={k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()},
                   code={p.name: _sha(p) for p in sorted(code.glob("z_eval*.py")) + sorted(code.glob("z_doc*.py"))}, n=C.N_D, n_holm=inp["N"], n_holm_wide=inp["N_wide"], m_family=C.M_FAMILY,
                   boot=S.B, boot_block=S.MEAN_BLOCK, boot_seed=S.SEED, lag_main=S.nw_lag(C.N_D), ktrim=C.KTRIM, thr=C.THR, hz1=bool(d.loc["H-Z1", "passed"]), hz2=bool(d.loc["H-Z2", "passed"]), decision=str(d.loc["Z-DEC", "value"]))
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    inp = load_inputs(ckpt)
    t = compute(inp)
    tx = texts(t, inp["tp"])
    merge_checkpoint(ckpt, t, tx, inp)                  # ключ S.Z.eval — до таблиц с p-значениями
    for k, v in tx.items():
        (rep / k).write_text(v, encoding="utf-8")
    return t


def tree_hash(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    ps = [rep / n for n in OUTS + (REPORT, "S_Z_NULL.csv") if (rep / n).exists()] + [ckpt] + list(C.FILES.values()) + [ROOT / "data/regen/harj_struct/Z/zmaps.npz"]
    return {str(p): _sha(p) for p in ps}


def check(rep: Path = REP, ckpt: Path = CKPT) -> list:
    from models_v2.s18 import z_eval_chk as K
    h0, inp = tree_hash(rep, ckpt), load_inputs(ckpt)
    t = compute(inp)
    tx = texts(t, inp["tp"])
    bad = [] if all((rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v for n, v in tx.items()) else ["таблицы ≠ пересчёту"]
    bad += K.problems(t, inp, rep, ckpt, tx)
    d = t["dec"].set_index("id")
    print("N Холма %d / %d, m = %d; H-Z1 %s, H-Z2 %s; Z-DEC %s: %s" % (inp["N"], inp["N_wide"], C.M_FAMILY, d.loc["H-Z1", "passed"], d.loc["H-Z2", "passed"], d.loc["Z-DEC", "value"], d.loc["Z-DEC", "text"]))
    return bad + ([] if tree_hash(rep, ckpt) == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        t = write()
        print("Z-DEC:", t["dec"].iloc[-1]["value"], "|", t["dec"].iloc[-1]["text"])
    elif "--check" in sys.argv:
        b = check()
        print("S-Z eval --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
