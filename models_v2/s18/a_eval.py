"""S-A-EVAL (Ж-254 пп. 11, 12, 16; `S_A_SPEC.md`, разд. 6, 8): оценка пилота S-A0 / S-A1 / S-A2 / S-A2p на D_M (t⁺ 2018-01-04…2026-11-10, n = 2198). Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD,
исследовательский результат на раскрытой истории, не независимый OOS; архитектура выбрана после изучения истории. Читает СОХРАНЁННЫЕ прогнозы `data/regen/harj_struct/A/*.parquet`, ряды X1, W5-G, W8-G, K1–K3, ZB; прогнозов и реестр не трогает.
Публичное API:
- `OUTS` — 9 CSV `docs/otchety/harj_struct/S_A_{TABLE,STATS,YEARS,CONC,GUARD,SPA,MCS,POWER,PARAMS}.csv`; `KEY` = "S.A.eval"; `SPEC_SHA` — первые 16 знаков sha256 входов по таблице спецификации разд. 8.
- `load_inputs() -> dict`: ряды F (K1, K2, K3, X1, W5-G, W8-G, B_mix, ZB, S-A0…S-A2p и слои X1_A(S-A*)) на одних датах; сверка d, rv, K1 с окном M — расхождение SystemExit; n = 2198, t⁺ 2018-01-04…2026-11-10.
- `compute(inp) -> dict` таблиц (`table`, `stats`, `years`, `conc`, `guard`, `spa`, `mcs`, `power`, `params`), `decision`, `identity`; `texts(t) -> {имя: текст}` (шапка с пометками, границами дат и n).
- `write()`: ключ `S.A.eval` (written_at, хэши входов, выходов и кода) в `docs/otchety/harj_x1/CHECKPOINT.json` пишется ПЕРВЫМ (слиянием, прочие ключи побайтно; формат indent 1 / sorted keys), затем таблицы с p.
- `check()` — только чтение: пересчёт = файлам; Σ вкладов = Δ (1e-12); лаг NW основной колонки = 8, B = 10 000; N = 831 / 3367 и m = 12; пороги = 0,415 / 0,50 / 1,00; решение выведено из строк GUARD, порядок A2 → A1; sha входов = спецификации; ключ = хэшам; хэш выходов до и после равен.
- `check_spa_cli()`: синтетика с известным ответом (`a_eval_spa.check_spa`) с σ_LR разностей из данных.
Запуск: `python3 -W ignore -m models_v2.s18.a_eval [--write | --check | --check-spa]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w5_eval as E5, x1_w8_diag_core as K
from models_v2.s18 import a_eval_par as PAR, a_eval_spa as SP, a_eval_stat as S, a_eval_tab as T, a_eval_yr as Y, a_reg as R

ROOT, A_DIR = R.ROOT, R.A_DIR
REP = ROOT / "docs/otchety/harj_struct"
CKPT, W8 = R.CKPT, ROOT / "data/regen/harj_x1/W8/w8g.parquet"
KEY = "S.A.eval"
KEYS = ("table", "stats", "years", "conc", "guard", "spa", "mcs", "power", "params")
OUTS = tuple("S_A_%s.csv" % k.upper() for k in KEYS)
VFILES = {"S-A0": "a0.parquet", "S-A1": "a1.parquet", "S-A2": "a2.parquet", "S-A2p": "a2p.parquet"}
SPEC_SHA = {"data/fwd/models/k1.parquet": "4092d6080c98de9d", "data/fwd/models/new/k1.parquet": "c1e4b23d67eebf72", "data/fwd/models/cfg8.parquet": "9b7062774189ad89",
            "data/fwd/models/new/cfg8.parquet": "c3a658bb8fdc7ad2", "data/regen/harj_deep/E1/x1.parquet": "56d9351607780959", "data/fwd/models/new/x1.parquet": "12c2b3614fd2a63b",
            "data/fwd/models/k23.parquet": "61f0548e385ebeb2", "data/fwd/models/new/k23.parquet": "0ff9ca293639f538", "data/regen/harj_x1/W5/w5g.parquet": "0ae40e1bcc7691e1",
            "docs/otchety/harj_x1/W5_GAP_PIT.csv": "dfd3092330ccda09"}
EXTRA = ("data/regen/harj_x1/W8/w8g.parquet", "data/fwd/models/zb.parquet", "data/fwd/models/new/zb.parquet")
FIRST, LAST, N_D = "2018-01-04", "2026-11-10", 2198
EPS = 1e-12
MARKS = ("исследовательский результат на раскрытой истории, не независимый OOS (Ж-254 п. 11); архитектура выбрана после изучения истории",
         "структура A0/A1/A2/A2p и критерии разд. 6 S_A_SPEC.md объявлены и зарегистрированы до счёта; «подтверждено» только при p Холма (N программы) < 0,05 и p SPA < 0,05")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def head(tp) -> str:
    tp = pd.DatetimeIndex(tp)
    return "\n".join("# " + m for m in MARKS) + "\n# D_M: t⁺ %s…%s, n = %d; l = RV/F − ln(RV/F) − 1, QLIKE = среднее дневных потерь; Δ в %% QLIKE базы, минус — лучше; DEVELOPMENT / HISTORICAL WALK-FORWARD\n" % (
        tp.min().date(), tp.max().date(), len(tp))


def load_inputs() -> dict:
    x, w8 = K.load(), pd.read_parquet(W8)
    if not (len(x) == N_D and str(x["t_plus"].min().date()) == FIRST and str(x["t_plus"].max().date()) == LAST):
        raise SystemExit("окно M ≠ D_M (2198 пар, t⁺ %s…%s)" % (FIRST, LAST))
    F = {"K1": x["K1"], "K2": x["K2"], "K3": x["K3"], "X1": x["F"], "W5-G": x["W5G"], "W8-G": w8["F"], "B_mix": x["M"], "ZB": E5._zb(x["d"])}
    frames = {}
    for v, f in VFILES.items():
        a = pd.read_parquet(A_DIR / f)
        ok = len(a) == len(x) and (a["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(a["rv"].to_numpy(), x["rv"].to_numpy()) and np.allclose(a["K1"], x["K1"], rtol=1e-12, atol=0)
        if not ok or not (np.isfinite(a["F"]).all() and np.isfinite(a["X1_A"]).all()):
            raise SystemExit("%s: даты, rv, K1 расходятся с окном M или есть нечисловые прогнозы" % v)
        frames[v], F[v], F["X1_A(%s)" % v] = a, a["F"], a["X1_A"]
    if not ((w8["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(w8["rv"].to_numpy(), x["rv"].to_numpy())):
        raise SystemExit("W8-G: даты или rv расходятся с окном M")
    return dict(x=x, rv=x["rv"].to_numpy(float), tp=pd.DatetimeIndex(x["t_plus"]), F={k: np.asarray(v, float) for k, v in F.items()}, frames=frames)


def compute(inp: dict) -> dict:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = T.losses(rv, F)
    spa = T.spa_blocks(ls)
    t = dict(table=T.table(rv, tp, F), stats=T.stats(rv, tp, F, ls, spa), years=Y.years(tp, ls), conc=Y.conc(tp, ls), spa=T.spa_table(spa), power=Y.power_table(tp, ls), params=PAR.params(inp["frames"]))
    t["guard"] = Y.guard(t["table"], t["years"], t["conc"])
    t["mcs"] = S.mcs(rv, {k: F[k] for k in ("S-A0", "S-A1", "S-A2", "W5-G", "X1", "W8-G", "K2", "K3", "B_mix")})
    t["decision"], t["identity"] = Y.decide(t["guard"]), Y.identity_gap(t["years"])
    return t


def texts(t: dict, tp) -> dict:
    h = head(tp)
    return {n: h + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g") for k, n in zip(KEYS, OUTS)}


def inputs_sha() -> dict:
    ps = list(SPEC_SHA) + list(EXTRA) + ["data/regen/harj_struct/A/" + f for f in VFILES.values()]
    return {p: _sha(ROOT / p) for p in ps}


def merge_checkpoint(ckpt: Path, t: dict, tx: dict, now: str | None = None) -> None:
    text = ckpt.read_text(encoding="utf-8")
    cp = json.loads(text)
    if json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False) != text:
        raise SystemExit("CHECKPOINT.json: формат не indent 1 / sorted keys — слияние ключа сломало бы прочие ключи")
    code = Path(__file__).resolve().parent
    cp[KEY] = dict(written_at=now or R.now_text(), inputs=inputs_sha(), outputs={k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()},
                   code={p.name: _sha(p) for p in sorted(code.glob("a_eval*.py"))}, n_holm=T.N_HOLM, n_holm_wide=T.N_WIDE, m_family=T.M_FAMILY, boot=S.B, lag_main=S.nw_lag(N_D), decision=t["decision"], n=N_D)
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    inp = load_inputs()
    t = compute(inp)
    tx = texts(t, inp["tp"])
    merge_checkpoint(ckpt, t, tx)                  # ключ S.A.eval — до таблиц с p-значениями
    for k, v in tx.items():
        (rep / k).write_text(v, encoding="utf-8")
    return t


def tree_hash(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    ps = [rep / n for n in OUTS if (rep / n).exists()] + [ckpt] + sorted(A_DIR.glob("*.parquet")) + [rep / "S_A_REPORT.md"] * (rep / "S_A_REPORT.md").exists()
    return {str(p): _sha(p) for p in ps}


def check(rep: Path = REP, ckpt: Path = CKPT) -> list:
    from models_v2.s18 import a_eval_chk as C
    h0, inp = tree_hash(rep, ckpt), load_inputs()
    t = compute(inp)
    tx = texts(t, inp["tp"])
    bad = [] if all((rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v for n, v in tx.items()) else ["таблицы ≠ пересчёту"]
    bad += C.problems(t, rep, ckpt, tx)
    print("решение:", t["decision"])
    return bad + ([] if tree_hash(rep, ckpt) == h0 else ["--check изменил выходы"])


def check_spa_cli() -> list:
    inp = load_inputs()
    rv, F = inp["rv"], inp["F"]
    d = S.Q.ql(rv, F["W5-G"]) - S.Q.ql(rv, F["S-A2"])
    return SP.check_spa(S.sigma_lr(d, S.nw_lag(len(d))), float(S.Q.ql(rv, F["W5-G"]).mean()), len(d))


if __name__ == "__main__":
    if "--write" in sys.argv:
        print("решение:", write()["decision"])
    elif "--check" in sys.argv:
        b = check()
        print("S-A eval --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    elif "--check-spa" in sys.argv:
        b = check_spa_cli()
        print("S-A eval --check-spa: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write, --check или --check-spa")
