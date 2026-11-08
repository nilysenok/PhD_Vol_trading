"""V4-B, G9: порог «go» и N_wide — в CHECKPOINT.json РАНЬШЕ любого файла результатов B. go_threshold_B = min(crit_panel K1, K2, K3, LGBM-D2) − 0,0005 (`hp_stats_sel.crit_panel`, M16 × цели 2017–2021).
N_wide = N_declared_before_stats (694) + уникальные пробы Optuna h = 1 (`data/regen/*h1*trials*.csv`)."""
from __future__ import annotations

import datetime as dt
import glob
import json

import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_b_data as D, v4_io as IO

CP = IO.REP / "CHECKPOINT.json"
CTRL = ["k1", "k2", "k3", "d2"]
N_WIDE_RULE = ("N_wide = N_declared_before_stats + число уникальных строк-проб в файлах data/regen/*h1*trials*.csv (ключ: файл, model, step, trial или point). В файлах нет колонки бумаги, поэтому "
               "считаются все пробы (верхняя оценка: «по этим бумагам» не отделить); пробы с одинаковыми гиперпараметрами в разных файлах — разные эксперименты и считаются отдельно")


def controls(k: pd.DataFrame) -> dict:
    """crit_panel четырёх контролей на общих датах (M16 × цели 2017–2021)."""
    t = S.crit_panel(k, CTRL).set_index("series")["crit_panel"]
    return {c: float(t[c]) for c in CTRL}


def go_threshold(ctrl: dict) -> float:
    return min(ctrl.values()) - S.MARGIN


def optuna_trials() -> tuple:
    """-> (число уникальных проб, {файл: проб})."""
    parts = {}
    for p in sorted(glob.glob(str(IO.P.ROOT / "data/regen/*h1*trials*.csv"))):
        d = pd.read_csv(p)
        key = ["model", "h", "step"] + (["trial"] if "trial" in d else ["point"])
        parts[p.split("/")[-1]] = int(len(d.drop_duplicates(key)))
    return sum(parts.values()), parts


def declare(k: pd.DataFrame) -> dict:
    """Записать (один раз) порог, контроли, N_wide и время в CHECKPOINT.json; прежние ключи не меняются. Повторный вызов только сверяет значения."""
    cp = json.loads(CP.read_text())
    ctrl = controls(k)
    n_opt, parts = optuna_trials()
    new = {"go_threshold_B": go_threshold(ctrl), "B_controls_crit_panel": ctrl, "B_margin": S.MARGIN, "N_wide": cp["N_declared_before_stats"] + n_opt, "N_wide_optuna_trials": n_opt,
           "N_wide_optuna_parts": parts, "N_wide_rule": N_WIDE_RULE,
           "B_rule": "«go» — лучшая из B1–B6 по crit_panel ≤ go_threshold_B; окна 2022+ в выборе не участвуют; B5 считается только по правилу (B5_RULE.csv)"}
    if "go_threshold_B" in cp:
        bad = [x for x in ("go_threshold_B", "N_wide") if abs(cp[x] - new[x]) > 1e-12]
        if bad:
            raise SystemExit("объявленные порог/N_wide не совпадают с пересчётом: %s" % bad)
        return cp
    cp.update(new)
    cp["B_declared_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    CP.write_text(json.dumps(cp, ensure_ascii=False, indent=1))
    return cp


def update(**kw) -> None:
    """Дописать ключи (время счёта и т. п.) после результатов; объявленное не меняется."""
    cp = json.loads(CP.read_text())
    for key in kw:
        if key in ("go_threshold_B", "N_wide", "B_declared_at"):
            raise ValueError("объявленное не меняется: " + key)
    cp.update(kw)
    CP.write_text(json.dumps(cp, ensure_ascii=False, indent=1))


def declared_before(files) -> list:
    """Нарушения: файлы результатов B, созданные не позже времени объявления."""
    t = dt.datetime.strptime(json.loads(CP.read_text())["B_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    return [str(f) for f in files if f.stat().st_mtime <= t]
