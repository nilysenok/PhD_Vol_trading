"""Л6: песочница сквозного пути открытия на поддельном «новом периоде» из АРХИВА (`otkrytie_harj_fwd.py --pesochnica <tmp>`). Реальное открытие не затрагивает.
Архив обрезан по 2025-12-31; «новые» файлы — архивные бары 2026-01-01…2026-02-27 в форматах `data/fwd/*/` (тот же `to_parquet(index=False)`); запечатанная RV — кодом
запечатывания `fwd_sber_m10_check` (`build_daily(.., proxy_days)` → `to_parquet`); FREEZE — настоящий `fwd_freeze.build` (тяжёлые разделы — календарь, входы, снимок — подменены на
песочничные); замок, выходы, FREEZE, репетиция, мощность, гейты — в tmp. Весь путь `run` после замка выполняется под сторожем (`otkrytie_guard`): реальные новые файлы,
реальный запечатанный файл и реальный замок не открываются. Итог сверяется с архивом 2026: K1, K2, K3, cfg8, X1, ZB, ZB0.
Подмена констант ARCHIVE_END/LAST_SESSION — только на время песочницы (setattr в fwd_cal, fwd_k23, fwd_cfg8); код Л1–Л5 не правится.
"""
from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pandas as pd

import otkrytie_guard as GD
import otkrytie_harj_fwd as OP
import otkrytie_proverki as PV
import otkrytie_raschet as RS

ROOT = PV.ROOT
sys.path.insert(0, str(ROOT / "scripts/dannye"))
AE, LAST = pd.Timestamp("2025-12-31"), pd.Timestamp("2026-02-27")
MODELS = (("k1", "target_date"), ("k23", "t_plus"), ("cfg8", "t_plus"), ("x1", "t_plus"), ("zb", "t_plus"), ("zb0", "t_plus"))
GATES = "# GATES песочницы\n\n" + "".join(f"## {g}. лист\n\n| Гейт | EVIDENCE |\n|---|---|\n| {g}.1 | passed |\n\n" for g in ("Л1", "Л2", "Л3", "Л4", "Л5"))


def cut_bars(src_path: Path, new_path: Path, arch_path: Path) -> None:
    df = pd.read_parquet(src_path).reset_index(drop=True)
    lim, end = AE + pd.Timedelta(hours=23, minutes=59), LAST + pd.Timedelta(hours=23, minutes=59)
    for p, part in ((arch_path, df[df["begin"] <= lim]), (new_path, df[(df["begin"] > lim) & (df["begin"] <= end)])):
        p.parent.mkdir(parents=True, exist_ok=True)
        part.reset_index(drop=True).to_parquet(p, index=False)


def prepare_data(tmp: Path):
    """Файлы данных песочницы в tmp: обрезанный архив, «новые» бары, запечатанная RV, обрезанные архивные выходы Л1–Л4. -> (Paths, каталог моделей)."""
    from models_v2.s10 import hp_k23_world as KW
    from models_v2.s17 import fwd_src as S
    import fwd_sber_m10_check as CHK
    from models_v2 import daily as D
    real, r = S.Paths(), tmp / "repo"
    p = S.Paths(bars_dir=r / "data/iss_check/bars_10m_full", imoex_archive=r / "data/iss_check/index_10m/IMOEX.parquet", new_sber=r / "data/fwd/sber_m10/SBER_m10_fwd.parquet",
                new_mkt_dir=r / "data/fwd/mkt_m10", sealed_rv=r / "data/fwd/sber_m10/rv_fwd_SEALED.parquet", lock=r / "data/fwd/models/OPENED_HARJ_FWD.json")
    srcp = {t: S.Src(real).archive_path(t) for t in KW.POOL + ["IMOEX"]}
    for t, sp in srcp.items():
        cut_bars(sp, S.Src(p).new_path(t), S.Src(p).archive_path(t))
    new = pd.read_parquet(p.new_sber)
    D.build_daily(new, CHK.proxy_days(new))[0].to_parquet(p.sealed_rv)  # запечатывание как `fwd_sber_m10_check.rv_rows`
    models = r / "data/fwd/models_arch"
    models.mkdir(parents=True, exist_ok=True)
    for name, col in MODELS:
        df = pd.read_parquet(ROOT / "data/fwd/models" / f"{name}.parquet")
        df[df[col] <= AE].reset_index(drop=True).to_parquet(models / f"{name}.parquet")
    return p, models


def paths_ok(p) -> list:
    """Все пути песочницы лежат вне реальных запрещённых путей (проверка путей, а не только сторож)."""
    return [str(x) for x in (p.bars_dir, p.imoex_archive, p.new_sber, p.new_mkt_dir, p.sealed_rv, p.lock) if GD.is_forbidden(x, ROOT) or ROOT.resolve() in Path(x).resolve().parents]


def freeze_patches(p, tmp: Path, rep: Path):
    """Подмены разделов FREEZE (остальное — настоящий `fwd_freeze.build`): календарь и входы песочницы, снимок и git пустые, репетиция и мощность — tmp."""
    from models_v2.s10 import hp_k23_world as KW
    from models_v2.s14 import v7_1b_sta as STA
    from models_v2.s17 import fwd_cal as FC, fwd_freeze as FZ, fwd_src as S
    src = S.Src(p)
    cal = FC.build_cal(LAST, src)
    pairs, days = FC.new_pairs(cal), cal[(cal["date"] > AE) & (cal["date"] <= LAST)]["date"]
    calsec = dict(days=[str(d.date()) for d in days], pairs=[[str(a.date()), str(b.date())] for a, b in zip(pairs["date"], pairs["t_plus"])], n_new=len(pairs))
    arch = {str(src.archive_path(t)): PV.sha256_file(src.archive_path(t)) for t in KW.POOL + ["IMOEX"]}
    arch.update({str(STA.directory() / STA.name("tcn", 2026, s)): PV.content_hash(STA.directory() / STA.name("tcn", 2026, s)) for s in range(5)})
    new = {str(src.new_path(t)): PV.sha256_file(src.new_path(t)) for t in KW.POOL + ["IMOEX"]} | {str(p.sealed_rv): PV.sha256_file(p.sealed_rv)}
    (rep / "REHEARSAL.csv").write_text("d,node,ok\n", encoding="utf-8")
    (rep / "POWER.csv").write_text("# песочница\nkind,window,hyp,n_new,block,power_holm7\n", encoding="utf-8")
    patches = [mock.patch.object(FZ.P, "calendar_section", lambda s: (calsec, pairs)), mock.patch.object(FZ.P, "inputs_archive", lambda: arch), mock.patch.object(FZ.P, "inputs_new", lambda: new),
               mock.patch.object(FZ.P, "derived_sha", lambda: {}), mock.patch.object(FZ.P, "snapshot", lambda s: {}), mock.patch.object(FZ.P, "git_section", lambda f, c=None: {}),
               mock.patch.object(FZ.RH, "OUT", rep / "REHEARSAL.csv"), mock.patch.object(FZ.PW, "OUT", rep / "POWER.csv")]
    return patches, FZ, arch, new


def main(tmp_arg: str) -> int:
    from models_v2.s17 import fwd_cal, fwd_cfg8, fwd_k23, fwd_src as S
    tmp = Path(tmp_arg).resolve()
    lock_state = lambda: (S.Paths().lock.exists(), S.Paths().lock.stat().st_mtime_ns if S.Paths().lock.exists() else None)  # noqa: E731
    real_lock_before, log = lock_state(), []
    p, models = prepare_data(tmp)
    assert not paths_ok(p), paths_ok(p)
    rep = tmp / "repo/docs/otchety/harj_fwd"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "GATES_FWD.md").write_text(GATES, encoding="utf-8")
    consts = [mock.patch.object(mod, name, AE if name == "ARCHIVE_END" else LAST) for mod, name in ((fwd_cal, "ARCHIVE_END"), (fwd_cal, "LAST_SESSION"), (fwd_k23, "LAST_SESSION"), (fwd_cfg8, "LAST_SESSION"))]
    with ExitStack() as st:
        for m in consts:
            st.enter_context(m)
        patches, FZ, arch, new = freeze_patches(p, tmp, rep)
        for m in patches:
            st.enter_context(m)
        frz = FZ.build(None)
    (rep / "FREEZE.json").write_text(json.dumps(frz, ensure_ascii=False), encoding="utf-8")
    ctx = PV.Ctx(root=ROOT, rep_dir=rep, freeze=rep / "FREEZE.json", gates=rep / "GATES_FWD.md", lock=p.lock, rehearsal=rep / "REHEARSAL.csv", power=rep / "POWER.csv", new_dir=tmp / "repo/data/fwd/models/new",
                 required=lambda: (list(arch), list(new)), rh_problems=lambda f: [])
    nodes = RS.real_nodes(ctx)
    nodes.Src, nodes.AE, nodes.LAST, nodes.archive = (lambda: S.Src(p)), AE, LAST, (lambda n: pd.read_parquet(models / f"{n}.parquet"))
    with ExitStack() as st:
        for m in consts:
            st.enter_context(m)
        st.enter_context(GD.guard(ROOT, log))
        rc = OP.run(ctx, dry=False, nodes=nodes)
    out = dict(rc=rc, lock_status=json.loads(p.lock.read_text(encoding="utf-8")).get("status"), real_lock_unchanged=lock_state() == real_lock_before, data_paths_read=len(log))
    for name, col in MODELS:
        got = pd.read_parquet(ctx.new_dir / f"{name}.parquet")
        ref = pd.read_parquet(ROOT / "data/fwd/models" / f"{name}.parquet")
        out[name] = RS.compare(name, got, ref[ref[col] > AE])["detail"]
    out["report_first_line"] = (rep / "FWD_REPORT.md").read_text(encoding="utf-8").splitlines()[0]
    out["family_rows"] = len(pd.read_csv(rep / "FWD_FAMILY.csv"))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if rc == 0 and out["real_lock_unchanged"] else 1
