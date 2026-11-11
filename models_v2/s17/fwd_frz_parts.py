"""Л5: разделы `FREEZE.json` v1 (протокол HARJ_HYBRID_SBER_FWD, раздел 6): замыкание импортов + git, входы архива и новых, параметры, снимок состояния на 2026-02-27, календарь.
Всё — чтение: файлы данных новых периодов не открываются (только MANIFEST.json); цен новых файлов нет (cut <= 2026-02-27).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # noqa: F401  (первым: общий OpenMP, класс № 152)

from models_v2.s6.xgb import cfgx
from models_v2.s10 import hp_k23_calc as KC, hp_k23_world as KW, hp_paths as HP
from models_v2.s14 import v7_1b_sta as STA, v7_1m_fit as FT, v7_1m_wf as WF, v7_3_cfg as C8
from models_v2.s13 import v6_1_wf as W6
from models_v2.s15 import de_e1_state as E1
from models_v2.s16 import x1_z_fit as ZF, x1_z_walk as ZW
from models_v2.s17 import fwd_cal as FC, fwd_cfg8 as F8, fwd_k23 as K, fwd_store as FS, fwd_x1 as X1, fwd_zb as ZB
from models_v2.s17.fwd_src import ARCHIVE_END, LAST_SESSION, ROOT, Src

COMMIT = "0cc2442e"
BRANCH = "harj-x1"
OPENER = ROOT / "scripts/fwd/otkrytie_harj_fwd.py"
V7, PAN = ROOT / "data/regen/harj_hybrid_v7", ROOT / "data/regen/harj_panel_hybrid"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def import_chain() -> None:
    """Импорт всех `models_v2.s17.fwd_*` и скрипта открытия (если он есть): то, что бегун загружает в `sys.modules`."""
    for p in sorted((ROOT / "models_v2/s17").glob("fwd_*.py")):
        importlib.import_module("models_v2.s17." + p.stem)
    if OPENER.exists():
        spec = importlib.util.spec_from_file_location("otkrytie_harj_fwd", OPENER)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def loaded_files() -> dict:
    """{относительный путь: sha256} всех файлов `models_v2/**` в `sys.modules` (+ скрипт открытия, если есть)."""
    out, base = {}, (ROOT / "models_v2").resolve()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).resolve().is_relative_to(base):
            out[str(Path(f).resolve().relative_to(ROOT.resolve()))] = sha(f)
    for p in sorted((ROOT / "scripts/fwd").glob("*.py")):
        out[str(p.relative_to(ROOT))] = sha(p)
    return dict(sorted(out.items()))


def closure_problems(frozen: dict, loaded: dict) -> dict:
    """Что загружено бегуном, но не замораживается (`not_in_freeze`), и у чего sha другой (`sha_differs`)."""
    return {"not_in_freeze": sorted(set(loaded) - set(frozen)), "sha_differs": sorted(k for k in loaded if k in frozen and frozen[k] != loaded[k])}


def _git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_section(files: list, commit: str | None = None) -> dict:
    """Коммит ветки `harj-x1` (по умолчанию `0cc2442e`; код Л5/Л6 — коммит, названный дирижёром); blob каждого файла замыкания в коммите против рабочего файла (`git hash-object`)."""
    full = _git("rev-parse", (commit or COMMIT) + "^{commit}")
    on_branch = subprocess.run(["git", "merge-base", "--is-ancestor", full, BRANCH], cwd=ROOT).returncode == 0
    absent, differs = [], []
    for f in files:
        t = _git("ls-tree", full, "--", f)
        if not t:
            absent.append(f)
        elif t.split()[2] != _git("hash-object", f):
            differs.append(f)
    dirty = [ln[3:] for ln in _git("status", "--porcelain", "--", *files).splitlines()]
    return dict(commit=full, branch=BRANCH, commit_on_branch=on_branch, head_workdir=_git("rev-parse", "HEAD"), workdir_branch=_git("branch", "--show-current"), not_in_commit=absent, differs_from_commit=differs, status_dirty_paths=dirty)


def packages() -> dict:
    import lightgbm, scipy, xgboost  # noqa: E401
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__}


def inputs_archive() -> dict:
    """sha256 всех входов раздела 1 + мета/решения K2/K3 + бары 17 бумаг и IMOEX; .npz энкодеров — content_hash."""
    paths = [PAN / n for n in ("har_SBER.parquet", "k23_SBER.parquet", "daily_SBER.parquet", "k23_meta_SBER.json", "panel_base.parquet")]
    paths += [V7 / "store_raw.npz", V7 / "X_w5.npz", V7 / "3/cfg8.parquet", ROOT / "data/regen/harj_deep/E1/x1.parquet", ROOT / "data/regen/harj_x1/Z/zb.parquet", ROOT / "data/regen/harj_x1/Z/zb0.parquet"]
    paths += [KW.ACC, cfgx.XGB_REP / "X2_DECISION.json", ROOT / "docs/otchety/harj_hybrid_v7/CHECKPOINT.json"]
    paths += [Src().archive_path(t) for t in KW.POOL + ["IMOEX"]]
    out = {str(Path(p).resolve().relative_to(ROOT.resolve())): sha(p) for p in paths}
    for s in F8.SEEDS:
        nm = STA.name("tcn", 2026, s)
        out[f"data/regen/harj_hybrid_v7/stageA/{nm}"] = STA.content_hash(STA.load(STA.directory() / nm))
    return out


def inputs_new() -> dict:
    """Из MANIFEST.json (файлы данных не читаются): sha256 манифеста и его содержимое."""
    out = {}
    for d in ("sber_m10", "mkt_m10"):
        p = ROOT / "data/fwd" / d / "MANIFEST.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        out[f"data/fwd/{d}/MANIFEST.json"] = sha(p)
        files = m.get("out_sha256") or {}
        for v in (m.get("instruments") or {}).values():
            files = {**files, **v.get("out_sha256", {})}
        out.update({f"data/fwd/{d}/{k}": v for k, v in files.items()})
        if "rv_sealed_sha256" in m:
            out[f"data/fwd/{d}/rv_fwd_SEALED.parquet"] = m["rv_sealed_sha256"]
    return out


def derived_sha() -> dict:
    return {f"data/fwd/models/{p.name}": sha(p) for p in sorted((ROOT / "data/fwd/models").glob("*")) if p.is_file() and not p.name.startswith(("tmp_", "OPENED", "rehearsal_part"))}


def params() -> dict:
    meta = json.loads((PAN / "k23_meta_SBER.json").read_text(encoding="utf-8"))
    return dict(E1=dict(GRID=list(E1.GRID), VAL=E1.VAL, A_DEF=E1.A_DEF, Q_EXTREME=E1.Q_EXTREME, EPS=E1.EPS), Z=dict(LAMS=list(ZF.LAMS), V_EXT=ZF.V_EXT, MIN_EXT=ZF.MIN_EXT, V_FULL=ZF.V_FULL, ACTIVATION=str(ZW.ACTIVATION.date()), MIN_PAIRS=ZW.MIN_PAIRS),
                cfg8=dict(GRID8=[list(g) if hasattr(g, "__iter__") else g for g in C8.GRID8], SEEDS=list(FT.SEEDS), HOLD=WF.HOLD, MIN_TRAIN=WF.MIN_TRAIN, FIRST_REFIT=WF.FIRST_REFIT, SPEC8=C8.SPEC8),
                K2K3=dict(SEED=42, i2_by_year=json.loads(meta["i2"]), i2_source="k23_meta_SBER.json; новый выбор И2 не делается (период < 01.01.2027)"), stageA=dict(kind="tcn", WINDOW=5, year=2026, seeds=list(F8.SEEDS)),
                K1=dict(MIN_PAIRS=HP.MIN_PAIRS, FIRST_TARGET=HP.FIRST_TARGET))


def calendar_section(src: Src) -> tuple:
    """Календарь по правилу IMOEX (только begin): дни, пары (d, t⁺), n_new, даты переоценок. Сверка с calendar.parquet Л1."""
    cal = FC.build_cal(LAST_SESSION, src)
    pd.testing.assert_frame_equal(pd.read_parquet(FC.OUT), cal)
    pairs, days = FC.new_pairs(cal), cal[(cal["date"] > ARCHIVE_END) & (cal["date"] <= LAST_SESSION)]["date"]
    first = days.groupby(days.dt.to_period("M")).min()
    iso = lambda s: [str(pd.Timestamp(x).date()) for x in s]  # noqa: E731
    return dict(rule="IMOEX: daily.build_daily, index_days = сессии IMOEX на склейке архив + новые; по begin, без цен", days=iso(days), pairs=[[str(a.date()), str(b.date())] for a, b in zip(pairs["date"], pairs["t_plus"])],
                n_new=len(pairs), refit_cfg8=iso(first), refit_alpha_x1=iso(first), refit_zb=iso(first), note="правило rv > 0 по ценам проверяет скрипт открытия после замка; расхождение — ↺"), pairs


def snapshot(src: Src) -> dict:
    """Начальное состояние на d = 2026-02-27 (только архив): π₀, π₁, η, α₀, α₁ X1; λ, b, a ZB (переоценка 2026-02-02); λ cfg8 февраля по seeds; И2 2026; прогнозы строки d = 02-27."""
    c = ARCHIVE_END
    w = K.build_world_at(c, src)
    tp = pd.Timestamp(K.next_day(pd.Timestamp(w["F"].dates[-1]), src))
    wp, i2 = K.pad_world(w, tp), K.load_i2()
    row = KC.forecast_frame(wp, i2, KC.select_rows(wp, min(i2), pd.DatetimeIndex([c])), 2)
    k1, daily = F8.k1_at(c, src)
    tab = F8.make_tab(k1, pd.concat([pd.read_parquet(F8.K23), row], ignore_index=True), daily)
    store = FS.build_store(c, src)
    st, zm = FS.build_states(c, src, store), F8.z_maps(tab.index, c, src, F8.SEEDS, store, FS.build_xw5(store))
    blk = F8.build_cfg8(c, src, dict(tab=tab, zmap=zm))
    x1, r = X1.run_x1(pd.concat([pd.read_parquet(F8.OUT), blk.iloc[[-1]]], ignore_index=True), tab, st)
    zb = ZB.build_zb(c, src, x1, st, store, "ZB")
    f = lambda v: float(v)  # noqa: E731
    return dict(d=str(c.date()), t_plus=str(tp.date()), X1={k: f(v) for k, v in X1.state_of(r).items()}, ZB={k: f(v) for k, v in ZB.state_of(zb).items() if k in ("lam", "b", "a", "n_ext")}, ZB_refit="2026-02-02",
                cfg8_refit=str(tab.index[W6.refit_positions(tab.index, WF.FIRST_REFIT)[-1]].date()), cfg8_lam_by_seed=blk.attrs["lam"], I2_2026=dict(hp_year=int(row["hp_year"].iloc[0]), hp_k2=str(row["hp_k2"].iloc[0]), hp_k3=str(row["hp_k3"].iloc[0])),
                forecast_d0227=dict(K2=f(row["sigma2_k2"].iloc[0]), K3=f(row["sigma2_k3"].iloc[0]), cfg8=f(blk["F"].iloc[-1]), X1=f(x1["F"].iloc[-1]), ZB=f(zb["F"].iloc[-1])))
