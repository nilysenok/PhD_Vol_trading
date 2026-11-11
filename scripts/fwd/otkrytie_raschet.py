"""Л6: счёт под замком — один прогон N1…N11 на склейке «архив + новые» (протокол HARJ_HYBRID_SBER_FWD, раздел 6, шаг 6).

Вызывается ТОЛЬКО после замка (замок создаёт `otkrytie_harj_fwd.run`); `Src` открывает цены новых файлов по замку. Узлы — замороженные `build_*` Л1–Л4 (импортом,
новой логики нет). Порядок: календарь (полный `build_daily`) против замороженного → запечатанная RV → единицы объёма → K1 → K2/K3 (архивные строки из `k23.parquet` Л2 +
сверка трёх последних архивных дней + новые даты) → склад, e_d, s_d → cfg8 (полный ход) → X1 → ZB, ZB0. Архивные строки каждого узла сверяются с замороженными файлами:
расхождение сверх допуска — стоп (`ArchiveMismatch`, причина идёт в замок). Расхождение календаря, RV или n — не стоп: строка `↺` в таблице сверок.
Возврат: {"new": новые строки узлов, "tables": таблицы раздела 5, "checks": сверки}. Узлы подаются объектом `nodes` (в тестах — синтетика).
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd

from models_v2.s17 import fwd_eval as E

TOL = 1e-12
RV_TOL = 1e-12
UNITS = (0.9, 1.1)


class ArchiveMismatch(RuntimeError):
    """Архивные строки узла на склейке не равны замороженным файлам."""


def real_nodes(ctx) -> types.SimpleNamespace:
    from models_v2.s17 import fwd_cfg8, fwd_k1, fwd_k23, fwd_src, fwd_store, fwd_x1, fwd_zb
    return types.SimpleNamespace(
        Src=lambda: fwd_src.Src(fwd_src.Paths(lock=ctx.lock)), AE=fwd_src.ARCHIVE_END, LAST=fwd_src.LAST_SESSION, daily=fwd_k1.build_daily_sber, k1=fwd_k1.build_k1, k23=fwd_k23.build_k23,
        store=fwd_store.build_store, xw=fwd_store.build_xw5, states=fwd_store.build_states, inputs=fwd_cfg8.inputs_at,
        cfg8=lambda cut, src, inp: fwd_cfg8.build_cfg8(cut, src, inp=inp, full=True), x1=fwd_x1.build_x1, zb=fwd_zb.build_zb,
        archive=lambda name: pd.read_parquet(ctx.root / "data/fwd/models" / f"{name}.parquet"))


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    return (a.to_numpy() == b.to_numpy()) | (a.isna().to_numpy() & b.isna().to_numpy())


def compare(name: str, got: pd.DataFrame, ref: pd.DataFrame, tol: float = TOL) -> dict:
    """Строки got = ref: нефлоты точно, флоты |Δ| ≤ tol·(1 + |ref|) (torch/XGBoost/LightGBM допуск протокола, раздел 4); иначе ArchiveMismatch."""
    got, ref = got.reset_index(drop=True), ref.reset_index(drop=True)
    if list(got.columns) != list(ref.columns) or len(got) != len(ref):
        raise ArchiveMismatch(f"{name}: форма {got.shape} ≠ архиву {ref.shape}")
    worst, bit = 0.0, True
    for c in ref.columns:
        a, b = got[c], ref[c]
        same = _same(a, b)
        if same.all():
            continue
        bit = False
        if a.dtype.kind != "f" or b.dtype.kind != "f":
            raise ArchiveMismatch(f"{name}.{c}: {int((~same).sum())} строк не равны архиву")
        x, y = a.to_numpy(float), b.to_numpy(float)
        if (np.isnan(x) != np.isnan(y)).any():
            raise ArchiveMismatch(f"{name}.{c}: NaN не по месту")
        worst = max(worst, float((np.abs(x - y)[~same] / (1 + np.abs(y[~same]))).max()))
    if worst > tol:
        raise ArchiveMismatch(f"{name}: расхождение {worst:.3g} > {tol:.0e}")
    return dict(check=f"архив {name}", status="ok", detail=f"строк {len(ref)}; побитно {bit}; макс. отн. расхождение {worst:.3g}")


def _row(check: str, ok: bool, detail: str) -> dict:
    return dict(check=check, status="ok" if ok else "↺", detail=detail)


def check_calendar(daily: pd.DataFrame, fz: dict, ae) -> dict:
    """Полный `build_daily` (с правилом rv > 0) против замороженного календаря; расхождение — ↺, дальше применяется полное правило."""
    cal = fz["calendar"]
    ours = [d.date().isoformat() for d in daily.index if d > ae]
    frozen = sorted(str(pd.Timestamp(d).date()) for d in cal["days"] if pd.Timestamp(d) > ae)
    extra, lost = sorted(set(ours) - set(frozen)), sorted(set(frozen) - set(ours))
    return _row("календарь: полный build_daily против замороженного", not extra and not lost, f"дней {len(ours)}; лишних {extra}; потерянных {lost}")


def check_sealed(daily: pd.DataFrame, sealed: pd.DataFrame) -> dict:
    """RV новых дней против запечатанной по общим датам (значения не печатаются, только максимум отн. расхождения и числа дат)."""
    if "date" in sealed.columns:  # контракт писателя (`fwd_sber_m10_check.rv_rows`): date — индекс; колонка тоже допустима
        sealed = sealed.set_index("date")
    s = pd.Series(sealed["rv"].to_numpy(float), index=pd.to_datetime(sealed.index))
    com = daily.index.intersection(s.index)
    err = float(np.max(np.abs(daily.loc[com, "rv"].to_numpy() - s.loc[com].to_numpy()) / s.loc[com].to_numpy())) if len(com) else float("nan")
    only_o, only_s = len(daily.index[daily.index > s.index.min()].difference(s.index)), len(s.index.difference(daily.index))
    return _row("RV новых дней против rv_fwd_SEALED", bool(len(com)) and err <= RV_TOL, f"общих дат {len(com)}; макс. отн. расхождение {err:.3g}; только у бегуна {only_o}; только в запечатанной {only_s}")


def check_units(bars: pd.DataFrame, ae) -> dict:
    """Единицы объёма новых файлов: дней вне 0,9…1,1 по медиане value/volume÷close (лоты дали бы ≈ 10 и более)."""
    new = bars[bars["begin"] > ae + pd.Timedelta(days=1)]
    r = (new["value"] / new["volume"] / new["close"]).groupby(new["begin"].dt.normalize()).median()
    out = int(((r < UNITS[0]) | (r > UNITS[1])).sum())
    return _row("единицы объёма новых файлов", out == 0 and len(r) > 0, f"дней {len(r)}; вне допуска {out}")


# xtask: длинная — один линейный прогон 11 узлов; разбивка по узлам разорвала бы порядок замороженного протокола
def run_nodes(ctx, fz: dict, nodes=None) -> dict:
    N = nodes or real_nodes(ctx)
    src, cut, ae = N.Src(), N.LAST, N.AE
    daily = N.daily(cut, src)
    chk = [check_calendar(daily, fz, ae), check_sealed(daily, src.sealed_rv()), check_units(src.bars("SBER", cut), ae)]
    store = N.store(cut, src)
    xw, states = N.xw(store), N.states(cut, src, store)
    k1 = N.k1(cut, src)
    chk.append(compare("k1", k1[k1["target_date"] <= ae], N.archive("k1")))
    a23 = N.archive("k23")
    f23 = N.k23(cut, src, dates="all")  # дневная переобучка по всей склейке (как `fwd_k23.main` на архиве)
    arch = (f23["t_plus"] <= ae).to_numpy()
    chk.append(compare("k23 (все архивные строки)", f23[arch], a23))
    k23_new = f23[~arch].reset_index(drop=True)
    k23 = pd.concat([a23, k23_new], ignore_index=True).sort_values("date").reset_index(drop=True)
    inp = N.inputs(cut, src, k23=k23, store=store, xw=xw)
    cfg8 = N.cfg8(cut, src, inp)
    x1 = N.x1(cut, src, cfg8, inp["tab"], states)
    zb, zb0 = N.zb(cut, src, x1, states, store, "ZB"), N.zb(cut, src, x1, states, store, "ZB0")
    full = dict(cfg8=cfg8, x1=x1, zb=zb, zb0=zb0)
    for k, v in full.items():
        chk.append(compare(k, v[v["t_plus"] <= ae], N.archive(k)))
    tab = inp["tab"]
    fr = E.common_frame(x1[["d", "t_plus", "rv"]], dict(K1=k1.set_index("d")["sigma2"], K2=tab["K2"], K3=tab["K3"], X1=x1.set_index("d")["F"], ZB=zb.set_index("d")["F"]),
                        dict(e=x1.set_index("d")["s"], ZB0=zb0.set_index("d")["F"], M=x1.set_index("d")["M"], cfg8=x1.set_index("d")["C"]), start=ae, end=cut)
    n_fz = fz["calendar"].get("n_new")
    chk.append(_row("число пар n против замороженного", n_fz is None or len(fr) == n_fz, f"n общих пар {len(fr)}; заморожено {n_fz}"))
    new = dict(k1=k1[k1["target_date"] > ae], k23=k23_new, **{k: v[v["t_plus"] > ae] for k, v in full.items()})
    t = E.all_tables(fr)
    t["checks"] = pd.DataFrame(chk)
    return dict(new={k: v.reset_index(drop=True) for k, v in new.items()}, tables=t)
