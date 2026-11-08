"""V5, лист 1b: стадия DX1 = пересчёт в памяти (`compute`), запись (`write`, не в `--check`) и сверка с диском (`check`, только чтение). Моделей не обучает, конфигурации не расходуются.
Не пишет и не читает `CHECKPOINT.json`, `ALL_TRIALS.csv`, `state/` (их ведёт лист 1a). Стадия вне путей прогноза."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_fin_check as FC
from models_v2.s12 import v5_0_bars as BA, v5_0_io as IO, v5_dx1_bns as B, v5_dx1_jump as JM, v5_dx1_mom as MO, v5_dx1_phi as PH, v5_dx1_report as RP

TABLES = {"DX1_MOMENTS.csv": "moments", "DX1_JUMPS.csv": "jumps"}
STATUS = "DEVELOPMENT / HISTORICAL, диагностика задним числом, не независимый OOS; Ж-226, лист 1b (моделей не обучали, конфигураций 0)"
CODE = ["v5_dx1_phi.py", "v5_dx1_mom.py", "v5_dx1_bns.py", "v5_dx1_jump.py", "v5_dx1_stage.py", "v5_dx1_report.py", "run_v5_dx.py"]
PIT_CUT = pd.Timestamp("2019-12-31")


def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE:
        h.update(f.encode() + (Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def header(note: str) -> str:
    return "# %s; код v5.1b %s; входы %s; %s\n" % (STATUS, code_hash(), IO.inputs_hash(), note)


def own_files() -> list:
    return sorted(f for d, pat in ((IO.REP, "DX1_*"), (IO.OUT, "dx1_*")) if d.exists() for f in d.glob(pat) if f.is_file())


def digest() -> str:
    """Хэш только выходов листа 1b (CHECKPOINT/ALL_TRIALS/state ведёт лист 1a и сюда не входят)."""
    h = hashlib.sha256()
    for f in own_files():
        h.update(str(f).encode() + IO.sha_file(f).encode())
    return h.hexdigest()


def pit_phi(b: pd.DataFrame, c: pd.DataFrame, phi: pd.DataFrame) -> float:
    """Макс. |φ(полная выборка) − φ(данные только до PIT_CUT)| на днях ≤ PIT_CUT: ноль, если φ зависят лишь от прошлого."""
    cut = PH.phi_table(b[b["date"] <= PIT_CUT], c[c["date"] <= PIT_CUT])
    a = cut.merge(phi, on=["ticker", "date"], suffixes=("_c", "_f"))
    return float(max(np.nanmax(np.abs(a[p + "_c"] - a[p + "_f"])) for p in PH.PHI))


def compute(write_cache: bool = False) -> dict:
    k = IO.frame()
    c = S.common(k, ["k1", "k2", "k3", "d2", "M", "K8"])
    c = c[c["group"] == "M16"].reset_index(drop=True)
    b = BA.cached_bands(write=False)
    phi = PH.phi_table(b, c)
    mom = MO.table(c, phi)
    days = B.cached_days(write=write_cache)
    jumps, jdec = JM.jump_table(k, days, b)
    return dict(moments=mom, jumps=jumps, mdec=MO.decision(mom), jdec=jdec, phi=phi, pit=pit_phi(b, c, phi), days=days, n_rows=len(c))


def write(r: dict) -> list:
    notes = {"moments": "G16: m_k = mean(s·φ_k), s = RV/F − 1, база K8/M, M16; t по НУ (лаг 5, панель по датам), Вальд по (m₀…m₆); правило п. 6.3",
             "jumps": "G17: BNS (BV проекта, TQ по M10-барам), уровень 0,1 %; частота, события, доля QLIKE; правило п. 6.5"}
    out = []
    for f, v in TABLES.items():
        p = IO.REP / f
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(header(notes[v]))
            r[v].to_csv(fh, index=False, lineterminator="\n")
        out.append(p)
    (IO.REP / "DX1_REPORT.md").write_text(RP.report(r), encoding="utf-8")
    return out


def check(r: dict) -> list:
    """Сверка пересчёта с диском (только чтение) + содержательные проверки G16, G17."""
    bad = []
    for f, v in TABLES.items():
        if not (IO.REP / f).exists():
            bad.append("%s: нет файла" % f)
        elif not FC.frames_equal(IO.read_csv(f), r[v], 1e-9):
            bad.append("%s: диск ≠ пересчёт в памяти" % f)
    m = r["moments"]
    if len(m) != 2 * 11 * 8 or not m["p_wald"].dropna().between(0, 1).all():
        bad.append("G16: строк моментов %d (ждём %d) или p вне [0;1]" % (len(m), 2 * 11 * 8))
    if r["pit"] > 1e-9:
        bad.append("G16: φ зависят от будущего: max|Δ| = %.2e" % r["pit"])
    if r["jumps"][r["jumps"]["section"] == "conc"].shape[0] != 6:
        bad.append("G17: DX1_JUMPS conc: ждём 6 строк (3 базы × 2 окна)")
    rep = IO.REP / "DX1_REPORT.md"
    if not rep.exists() or rep.read_text(encoding="utf-8") != RP.report(r):
        bad.append("DX1_REPORT.md: нет или ≠ пересчёту")
    return bad
