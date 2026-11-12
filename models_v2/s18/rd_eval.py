"""S-RCD-EVAL (Ж-259; `S_RCD_SPEC.md`, разд. 6, 7, 8.4): оценка S-RCD, S-CDF, S-CKF на D_M (t⁺ 2018-01-04…2026-11-10, n = 2198). Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, исследовательский результат на раскрытой истории, не OOS; гипотеза выбрана после просмотра результатов S-R, S-D, S-L, S-Z.
Читает СОХРАНЁННЫЕ прогнозы `data/regen/harj_struct/{RD,CM,R,D,Z,L}/*.parquet`, ряды X1, W5-G, K1–K3, B_mix и `S_RCD_NULL.csv`; прогнозов и реестр не трогает. Обвязка — `a_eval_*`; здесь — `rd_eval_cfg` (константы), `_tab`, `_gd`, `_dc`, `_par`.
Публичное API:
- `KEYS` — 12 таблиц `docs/otchety/harj_struct/S_RCD_{TABLE,STATS,DECOMP,GUARD,CONC,YEARS,SPA,MCS,POWER,PARAMS,SEL,DEC}.csv` (`OUTS`); отчёт `S_RCD_REPORT.md` — `rd_doc`; `KEY` = "S.RCD.eval".
- `load_inputs(ckpt)`, `compute(inp) -> dict` таблиц (H-RCD1, наборы А и Б, s и RCD-DEC выводятся из таблиц), `texts(t, tp)`, `merge_checkpoint`.
- `write()`: ключ `S.RCD.eval` (written_at, хэши входов, выходов, кода) пишется в `docs/otchety/harj_x1/CHECKPOINT.json` ПЕРВЫМ (слиянием), затем таблицы с p. `check()` — только чтение: `rd_eval_chk.problems`; sha256 выходов до и после.
Запуск: `python3 -W ignore -m models_v2.s18.rd_eval [--write | --check]` (перед `--write` — `rd_eval_null --run`)."""
from __future__ import annotations

import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import rd_eval_cfg as C
from models_v2.s18 import rd_eval_dc as DC
from models_v2.s18 import rd_eval_gd as GD
from models_v2.s18 import rd_eval_null as RN
from models_v2.s18 import rd_eval_par as PAR
from models_v2.s18 import rd_eval_tab as T

ROOT, REP, CKPT, KEY = C.ROOT, C.REP, C.CKPT, C.KEY
KEYS = ("table", "stats", "decomp", "guard", "conc", "years", "spa", "mcs", "power", "params", "sel", "dec")
OUTS = tuple("S_RCD_%s.csv" % k.upper() for k in KEYS)
REPORT = "S_RCD_REPORT.md"
MARK = "исследовательский результат на раскрытой истории, не OOS; гипотеза выбрана после просмотра результатов S-R, S-D, S-L, S-Z"
MARKS = (MARK, "критерии разд. 7 S_RCD_SPEC.md объявлены и зарегистрированы до счёта; «подтверждено» только при p Холма (N программы) < 0,05 и p SPA < 0,05 по своей строке; S-RCK не заменяется без решения владельца")
EXTRA = ("docs/otchety/harj_struct/S_RCD_SPEC.md", "docs/otchety/harj_struct/S_RCD_NULL.csv", "docs/otchety/harj_struct/GATES_SRCD_IMPL.md", "data/regen/harj_x1/W5/w5g.parquet")


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def head(tp) -> str:
    tp = pd.DatetimeIndex(tp)
    return "\n".join("# " + m for m in MARKS) + "\n# D_M: t⁺ %s…%s, n = %d; l = RV/F − ln(RV/F) − 1, QLIKE = среднее дневных потерь (без обрезки); Δ в %% QLIKE базы, минус — лучше; DEVELOPMENT / HISTORICAL WALK-FORWARD\n" % (tp.min().date(), tp.max().date(), len(tp))


def load_inputs(ckpt: Path = CKPT) -> dict:
    inp = C.load_inputs(ckpt)
    if not RN.OUT.exists():
        raise SystemExit("нет S_RCD_NULL.csv: сначала `rd_eval_null --run` (диагностика объявлена до QLIKE)")
    inp["null"] = RN.read()
    return inp


def compute(inp: dict) -> dict:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = T.losses(rv, F)
    spa, act = T.spa_blocks(ls), DC.act_mask(inp)
    tb, st = T.table(rv, tp, F, ls), T.stats(rv, tp, ls, spa, inp["N"], inp["N_wide"], act)
    yrs, cnc = GD.years(tp, ls), GD.conc(tp, ls)
    g = GD.guard(yrs, cnc)
    return dict(table=tb, stats=st, decomp=DC.decomp(inp, ls, inp["null"]), guard=g, conc=cnc, years=yrs, spa=T.spa_table(spa), mcs=T.mcs(rv, F), power=T.power_table(tp, ls),
                params=PAR.params(inp["frames"]), sel=PAR.sel(inp["frames"]), dec=GD.dec_rows(tb, g, st), identity=GD.identity_gap(yrs))


def texts(t: dict, tp) -> dict:
    h = head(tp)
    return {n: h + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g") for k, n in zip(KEYS, OUTS)}


def inputs_sha() -> dict:
    ps = [str(p.relative_to(ROOT)) for p, *_ in C.SERIES.values()] + list(EXTRA)
    return {p: _sha(ROOT / p) for p in ps}


def code_sha() -> dict:
    code = Path(__file__).resolve().parent
    return {p.name: _sha(p) for p in sorted(code.glob("rd_eval*.py")) + sorted(code.glob("rd_doc*.py"))}


def merge_checkpoint(ckpt: Path, t: dict, tx: dict, inp: dict, now: str | None = None) -> None:
    from models_v2.s18 import r_reg as RR
    text = ckpt.read_text(encoding="utf-8")
    cp = json.loads(text)
    if json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False) != text:
        raise SystemExit("CHECKPOINT.json: формат не indent 1 / sorted keys — слияние ключа сломало бы прочие ключи")
    d = t["dec"].set_index("id")
    cp[KEY] = dict(written_at=now or RR.now_text(), inputs=inputs_sha(), outputs={k: _sha_text(v) for k, v in tx.items()}, code=code_sha(), n=C.N_D, n_holm=inp["N"], n_holm_wide=inp["N_wide"], m_family=C.M_FAMILY,
                   boot=S.B, boot_block=S.MEAN_BLOCK, boot_seed=S.SEED, lag_main=S.nw_lag(C.N_D), ktrim=C.KTRIM, thr=C.THR, h_rcd1=bool(d.loc["H-RCD1", "passed"]), decision=str(d.loc["RCD-DEC", "value"]))
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    inp = load_inputs(ckpt)
    t = compute(inp)
    tx = texts(t, inp["tp"])
    merge_checkpoint(ckpt, t, tx, inp)                  # ключ S.RCD.eval — до таблиц с p-значениями
    for k, v in tx.items():
        (rep / k).write_text(v, encoding="utf-8")
    return t


def amend(reason: str, only=("S_RCD_DEC.csv",), rep: Path = REP, ckpt: Path = CKPT) -> list:
    """Поправка перечисленных таблиц (только текст DEC): пересчёт, запись изменившихся, слияние их хэшей и кода в ключ `S.RCD.eval` + поле `amended` (время, причина, таблицы); прочие таблицы и ключи не трогаются."""
    from models_v2.s18 import r_reg as RR
    inp = load_inputs(ckpt)
    tx = texts(compute(inp), inp["tp"])
    changed = [n for n in tx if tx[n] != (rep / n).read_text(encoding="utf-8")]
    if set(changed) - set(only):
        raise SystemExit("поправка касается не только %s: %s" % (only, changed))
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    for n in changed:
        (rep / n).write_text(tx[n], encoding="utf-8")
        cp[KEY]["outputs"][n] = _sha_text(tx[n])
    cp[KEY]["code"] = code_sha()
    cp[KEY]["amended"] = dict(at=RR.now_text(), reason=reason, tables=changed)
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return changed


def tree_hash(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    ps = [rep / n for n in OUTS + (REPORT, "S_RCD_NULL.csv") if (rep / n).exists()] + [ckpt] + [p for p, *_ in C.SERIES.values()]
    return {str(p): _sha(p) for p in ps}


def check(rep: Path = REP, ckpt: Path = CKPT) -> list:
    from models_v2.s18 import rd_eval_chk as K
    h0, inp = tree_hash(rep, ckpt), load_inputs(ckpt)
    t = compute(inp)
    tx = texts(t, inp["tp"])
    bad = [] if all((rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v for n, v in tx.items()) else ["таблицы ≠ пересчёту"]
    bad += K.problems(t, inp, rep, ckpt, tx)
    d = t["dec"].set_index("id")
    print("N Холма %d / %d, m = %d; H-RCD1 %s; s = %s; RCD-DEC %s: %s" % (inp["N"], inp["N_wide"], C.M_FAMILY, d.loc["H-RCD1", "passed"], d.loc["доля s", "value"], d.loc["RCD-DEC", "value"], d.loc["RCD-DEC", "text"]))
    return bad + ([] if tree_hash(rep, ckpt) == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        t = write()
        print("RCD-DEC:", t["dec"].iloc[-1]["value"], "|", t["dec"].iloc[-1]["text"])
    elif "--amend" in sys.argv:
        print("поправлены:", amend("текст RCD-DEC-2 («численно улучшает … не подтверждено»), проверка строки «S-RCD против S-RCK» в p_dec, замечания проверяющего; числа и код решения не менялись"))
    elif "--check" in sys.argv:
        b = check()
        print("S-RCD eval --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
