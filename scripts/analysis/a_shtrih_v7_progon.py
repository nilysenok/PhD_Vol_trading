"""Прогон A′-расширенной (Ж-146 п. 3, Ж-149, Ж-152) на центрах S1–S12, train4 и v2 (< 2020-01-01). Разработка, потолок, отбора нет.
Запуск: python3 scripts/analysis/a_shtrih_v7_progon.py --period train4|v2 [--out папка]. Данные — теневой корень FASTBT_MAIN (только чтение).
Центры: S1–S4 — docs/tables/golaya_otnositelno_v7/centry.csv, S5/S6 — средний элемент сетки v5 (golaya_tf_v5.tsentr), S7–S12 — golaya_v7.CENTR.
Ряды по тикерам — fastbt.a_shtrih_v7_vhody; книга и таблицы — a_shtrih_v7 (драйвер). Метрики считаются на окне периода (прогрев — только
история для β/σ/весов). Выход: svod_<period>.csv (основная, контроли, A′-базовая, A_strogaya, центры), ablaciya_<period>.csv."""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import a_shtrih_v7 as drv  # noqa: E402
import golaya_otnositelno_v7 as gotn  # noqa: E402
import golaya_v7 as gv  # noqa: E402
from fastbt import a_shtrih_v7_vhody as V, daily_1850 as d18, metriki_e as me, mix_skleyka as mx, progon_urovni as pu  # noqa: E402
from fastbt.signals import v5_s56  # noqa: E402
from fastbt.sravneniya_wf import sravnit  # noqa: E402

OUT = ROOT / "docs" / "tables" / "a_shtrih_v7"
SIDS = tuple(f"S{i}" for i in range(1, 13))
STROGAYA = ROOT / "docs" / "tables" / "a_strogaya" / "ryad_A.parquet"
PARY = (("A'-расширенная", "A'-базовая"), ("A'-расширенная", "контроль_равный_риск"), ("A'-расширенная", "контроль_без_хеджа"))


def _tsentr(osi: dict) -> dict:
    return {k: (v[len(v) // 2].item() if hasattr(v[len(v) // 2], "item") else v[len(v) // 2]) for k, v in sorted(osi.items())}


def tsentry_s56() -> None:
    """Центры S5/S6 (средний элемент оси) — в словарь golaya_otnositelno_v7.CENTR, который читает его signaly."""
    gotn.CENTR.update(S5=_tsentr(v5_s56.G5), S6=_tsentr(v5_s56.G6))


def ryady_centrov(D: dict) -> tuple:
    """(X[S,N,T], poz[S,N,T], {sid: lag->corr Д1}) по центрам S1–S12; позиция знаковая."""
    Xs, Ps, d1 = [], [], {}
    for sid in SIDS:
        if int(sid[1:]) <= 6:
            S, EL, ES = gotn.signaly(D, sid, "net")
            X, P = V.pary(D, S, EL, ES)
        else:
            kw = gv.shag(sid)
            X, P = V.pary(D, gv.signaly(D, sid), vhod=kw["vhod"], tau=kw["tau"], sost=kw["sost"])
        d1[sid] = V.proverka_sdviga(X, P, D["cr"].to_numpy())
        Xs.append(X)
        Ps.append(P)
    return np.stack(Xs), np.stack(Ps), d1


def sektora(D: dict) -> np.ndarray:
    m = D["metki"][D["win"]][-1]
    return np.array([x if x else f"_{t}" for x, t in zip(m, D["tk"])])      # без сектора — свой собственный (не склеиваются)


def vhod_knigi(D: dict, X, P, d1: dict) -> dict:
    mix = mx.skleit(mx.zagruzit_kontrakty(Path(d18.default_main()) / "data" / "iss_mix_2015_2019"), D["daty"])["ret_oo"].to_numpy()
    return dict(X=X, poz=P, sektora=sektora(D), imoex=np.asarray(D["rm"], float), mix=mix, ks=np.asarray(D["stavka"], float),
                daty=np.asarray(D["daty"].values).astype("datetime64[D]"), znak="znakovyj", istochnik="ryady_centrov_S1_S12_v5_v7", d1=d1)


def okno(res, d: dict, w: np.ndarray):
    r = dataclasses.replace(res, **{k: getattr(res, k)[w] for k in ("ryad", "m", "f", "h", "kom_hedzh", "mix_nan", "kom_tik")},
                            poz_itog=res.poz_itog[:, w])
    return r, dict(d, daty=d["daty"][w], imoex=d["imoex"][w])


def _kf(x) -> bool:
    """z_cf ≥ 0 — вне области Корниша–Фишера (mSR не определён)."""
    x = np.asarray(x, float)
    s, k, z = me.asimmetriya(x), me.ekscess(x), me.Z95
    return bool(z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36 >= 0)


def dop(x, L: int, oborot=None) -> dict:
    x = np.asarray(x, float)[np.isfinite(x)]
    return dict(dohodnost_pct_god=float(x.sum() / L * 100), msr_vne_oblasti_KF=_kf(x),
                oborot_pct_god=np.nan if oborot is None else float(oborot / L * 100))


def stroka_knigi(res, d: dict, L: int) -> dict:
    w_ = res.poz_itog * res.m * res.f
    obor = np.abs(np.diff(w_, axis=1, prepend=0.0)).sum() + np.abs(np.diff(res.h, prepend=0.0)).sum()
    return {**drv.stroka(res, d), **dop(res.ryad, L, obor)}


def stroka_ryada(x, daty, imoex, L: int) -> dict:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    ned = me.metriki_ned(x[ok], daty[ok])
    return dict(sharpe=me.sharpe(x[ok]), msr=me.msr(x[ok]), sortino=me.sortino_e(x[ok]), mdd=me.mdd(x[ok]), kalmar=me.kalmar(x[ok]),
                sharpe_ned=ned["sharpe_ned"], msr_ned=ned["msr_ned"], sortino_ned=ned["sortino_ned"], n_nedel=ned["n_nedel"],
                dolya_rynka=drv.dolya_rynka(x, imoex), n_dnej=int(ok.sum()), **dop(x, L))


def strogaya(daty: pd.DatetimeIndex) -> np.ndarray | None:
    """Ряд A_strogaya (docs/tables/a_strogaya, центры якоря s9, исполнение open 10:00) на днях окна; нет дня — NaN."""
    if not STROGAYA.exists():
        return None
    s = pd.read_parquet(STROGAYA)                     # колонки date, excess, excess_zaem5; берём с платой за шорт (как наши ряды)
    return s.set_index(pd.to_datetime(s["date"]))["excess_zaem5"].reindex(daty).to_numpy(float)


def schitat_period(period: str, out: Path) -> dict:
    tsentry_s56()
    D = gotn.zagruzit(period)
    X, P, d1 = ryady_centrov(D)
    d = vhod_knigi(D, X, P, d1)
    drv.proverit_vhod(d)
    w, L = D["win"], len(D["gody"])
    rez = drv.schitat(d)
    ok_ = {k: okno(r, d, w) for k, r in rez.items()}
    ch = rez["osnovnaya"]
    assert (ch.kom_tik[w].sum() > 0), "G4: комиссия оборота тикеров на окне = 0"
    bh = rez["kontrol_bez_hedzha"]
    assert not bh.h.any() and not bh.kom_hedzh.any(), "G5: в «без хеджа» есть хедж или его комиссия"
    dw = ok_["osnovnaya"][1]
    komp = {s: X[i] for i, s in enumerate(SIDS)}
    poz = {s: P[i] for i, s in enumerate(SIDS)}
    baz = pu.a_shtrih(komp, pu.osnovnaya_konf(), D["daty"], poz)
    bw = baz[w]
    svod = {"A'-расширенная": stroka_knigi(*ok_["osnovnaya"], L), "контроль_равный_риск": stroka_knigi(*ok_["kontrol_ravnyj_risk"], L),
            "контроль_без_хеджа": stroka_knigi(*ok_["kontrol_bez_hedzha"], L), "A'-базовая": stroka_ryada(bw, dw["daty"], dw["imoex"], L)}
    sg = strogaya(D["daty"][w])
    svod["A_strogaya"] = (stroka_ryada(sg, dw["daty"], dw["imoex"], L) if sg is not None and np.isfinite(sg).sum() > 100 else
                          {"примечание": "не сопоставимо: ряд ryad_A (центры якоря s9, вход open 10:00) на этом периоде не найден"})
    for i, s in enumerate(SIDS):
        svod[f"центр_{s}"] = stroka_ryada(np.nanmean(X[i], axis=0)[w], dw["daty"], dw["imoex"], L)
    ryd = {n: pd.Series(ok_[k][0].ryad, dw["daty"]) for n, k in (("A'-расширенная", "osnovnaya"), ("контроль_равный_риск", "kontrol_ravnyj_risk"),
                                                                ("контроль_без_хеджа", "kontrol_bez_hedzha"))}
    ryd["A'-базовая"] = pd.Series(bw, dw["daty"])
    if sg is not None and np.isfinite(sg).sum() > 100:
        ryd["A_strogaya"] = pd.Series(sg, dw["daty"])
    par = PARY + ((("A'-базовая", "A_strogaya"),) if "A_strogaya" in ryd else ())
    cmp_ = pd.DataFrame(sravnit(ryd, pary=par)["rows"])
    abl = {k: stroka_knigi(*ok_[k], L) for k in ("ablaciya_1", "ablaciya_2", "ablaciya_3", "ablaciya_4")}
    sh = {**drv.shapka(d), "okno": f"{dw['daty'].min()} .. {dw['daty'].max()}, {len(dw['daty'])} дней (прогрев — только история)",
          "status": "разработка, период " + period + ", до аудита, потолок, отбора нет"}
    out.mkdir(parents=True, exist_ok=True)
    for nm, tab in (("svod", pd.DataFrame(svod).T), ("ablaciya", pd.DataFrame(abl).T)):
        p = out / f"{nm}_{period}.csv"
        h = {**sh, "stroki": ", ".join(tab.index)}
        p.write_text("".join(f"# {k}: {h[k]}\n" for k in drv.POLYA_SHAPKI) + tab.to_csv(index_label="stroka"), encoding="utf-8")
    pd.DataFrame(d1).T.rename(columns=lambda k: f"lag_{k}").to_csv(out / f"d1_{period}.csv", index_label="centr")
    cmp_.to_csv(out / f"sravnenie_{period}.csv", index=False)
    return dict(d1=d1, svod=svod, abl=abl, cmp=cmp_)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=gv.PERIODY_OK)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    r = schitat_period(a.period, Path(a.out))
    print({s: {k: round(v, 3) for k, v in c.items()} for s, c in r["d1"].items()})
    print(pd.DataFrame(r["svod"]).T[["sharpe", "msr", "dohodnost_pct_god", "mdd", "dolya_rynka"]].to_string())
    print(pd.DataFrame(r["abl"]).T[["sharpe", "msr", "dohodnost_pct_god", "mdd", "dolya_rynka"]].to_string())
    print(r["cmp"].to_string())
