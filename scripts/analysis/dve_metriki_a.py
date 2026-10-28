"""Две метрики для A_strogaya и якоря A, train4 (2016–2019), справочно, без выбора (Ж-102 п. 2, Ж-103 п. 1, 4).
Запуск: FASTBT_BARS1M=<D25/bars1m> FASTBT_CACHE=<копия кэша> python3 scripts/analysis/dve_metriki_a.py
Пишет docs/tables/dve_metriki/a_strogaya_yakor.csv. Определения — fastbt/metriki_e.py (Д7): Sharpe = mean/sd(ddof=1)·√252
дневного excess над КС, mSR — Корниш–Фишер, skew/kurt — смещённые моментные (kurt избыточный). Данные ≥ 2020 не читаются.

Строки: (а) полный ряд; (б) без 10 худших / без 10 лучших ДНЕЙ портфеля (дни вычеркиваются из ряда); (в) стоп 2σ поверх точки
(k_s = 2, остальное как у точки: родные выходы Sk остаются, вход/выход строгие open d+1); (г) размер 1/σ_d: razmer = 20
(цель 20 % годовых, вес = цель/(σ_d√252), L = 1; значение из сетки zh047). Плата за шорт 5 % — основная, 0 — справочно.
«Шарп в деле» — только дни, где у портфеля есть позиция (expo > 0 хотя бы у одной пары любого Sk).
«Экспозиция» — доля дней ряда с позицией (дни «в деле» / все дни). «Шарп без ФДР» снят владельцем (21:03)."""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import a_strogaya as AS, etap1, metriki_e as me, progon_urovni as urovni  # noqa: E402
from fastbt.etap1_eval_v2 import EvaluatorV2  # noqa: E402

OUT = ROOT / "docs" / "tables" / "dve_metriki"
STRICT = {"t_vyhod": "open2", "proverka": "bary"}
TOCHKI = {"baza": {}, "stop": {"k_s": 2}, "razmer": {"razmer": 20}, "kontrol": {"razmer": 1e6}}     # kontrol: вес = 1 → как база
STROKI = [("baza", "(а) полные", None), ("baza", "(б) без 10 худших дней", "hud"), ("baza", "(б) без 10 лучших дней", "luch"),
          ("stop", "(в) стоп 2σ поверх точки", None), ("razmer", "(г) размер 1/σ_d (razmer = 20)", None)]


class RodnyeVyhody:
    """Позиции Sk с их родными выходами (как у якоря), хотя точка с новой осью — не «якорь»: стоп/размер добавляются поверх."""

    def _positions(self, t, sp, q, anchor, gi1):
        return super()._positions(t, sp, q, True, gi1)


class Yakor(RodnyeVyhody, EvaluatorV2): pass            # noqa: E701
class Strogaya(RodnyeVyhody, AS.AStrogaya): pass        # noqa: E701


def ryad(ev, pt: dict) -> dict:
    """Ряды Sk для точки pt: как AS.schitat, плюс экспозиция по датам."""
    row, x, tik = ev.evaluate_v2(pt, AS.PER)
    if x is None:
        raise ValueError(f"{ev.sid} {pt}: {row.get('prichina')}")
    df, inv, eng, _, rs = ev._last
    _, _, dts, _, _, rdates = ev._series(AS.PER, ev.comm)
    cat = lambda k: np.concatenate([getattr(r, k) for r in rs])                       # noqa: E731
    keep = np.isin(df.index.year, etap1.years_of(AS.PER))
    return dict(x=x, tik=tik, baza=np.asarray(ev.zaem), dates=dts.to_numpy().astype("datetime64[D]"),
                expo=cat("expo"), cnt=cat("cnt"), god=rdates.year.to_numpy(), expo_d=np.bincount(inv, cat("expo"), len(eng.dates))[keep])


def schitat(sid: str) -> dict:
    """{портфель: {точка: ряды Sk}} для одного Sk."""
    inp, base, out = etap1.build_inputs(sid, period=AS.PER), etap1.base_point(sid), {}
    for imya, strogo in (("yakor", False), ("strogaya", True)):
        cls = Strogaya if strogo else Yakor
        ev0 = AS.AStrogaya(sid, inp, base, AS.COMM) if strogo else EvaluatorV2(sid, inp, base, AS.COMM)
        ev = cls(sid, inp, base, AS.COMM)
        out[imya] = {"baza": ryad(ev0, {"t_vyhod": "open2"} if strogo else {})}
        for k, pt in TOCHKI.items():
            if k == "baza" or (k == "kontrol" and not strogo):
                continue
            # якорь: «A со стопом» — строгое исполнение; (г) для якоря — исполнение как у якоря (close), меняется только размер
            out[imya][k] = ryad(ev, {**pt, **(STRICT if strogo or k == "stop" else {})})
        print(sid, imya, "готово", flush=True)
    return out


def portfel(res: dict, k: str) -> dict:
    """A из шести Sk (равный вес, без таргета): x, база платы за шорт, дни с позицией."""
    rr = {s: res[s][k] for s in AS.SIDS}
    a = AS.portfel_A(rr)
    a["delo"] = np.sum([r["expo_d"] for r in rr.values()], axis=0) > 0
    return a


def stroka(x, delo, hvost: str | None, plata: float, baza) -> dict:
    """Метрики одного ряда; hvost: 'hud' / 'luch' — вычеркнуть 10 худших / лучших дней (по x с платой за шорт)."""
    x = x - plata * baza
    keep = np.ones(len(x), bool)
    if hvost:
        keep[np.argsort(x)[:10] if hvost == "hud" else np.argsort(x)[-10:]] = False
    xs = x[keep]
    sh = lambda v: float(me.sr_dnevnoy(v) * np.sqrt(252))                             # noqa: E731
    m = me.msr(xs)
    return dict(sharpe=sh(xs), msr=m, razryv=m - sh(xs), skew=me.asimmetriya(xs), kurt=me.ekscess(xs),
                sharpe_v_dele=sh(x[keep & delo]), ekspoziciya=float((keep & delo).sum() / keep.sum()),
                dni=int(keep.sum()), dni_v_dele=int((keep & delo).sum()))


def main() -> None:
    with Pool(2) as p:
        per = dict(zip(AS.SIDS, p.map(schitat, AS.SIDS)))
    rows = []
    for imya in ("yakor", "strogaya"):
        res = {s: per[s][imya] for s in AS.SIDS}
        port = {k: portfel(res, k) for k in res["S1"]}
        for k, nazv, hv in STROKI:
            for plata in (AS.PLATA, 0.0):
                a = port[k]
                rows.append(dict(portfel=imya, stroka=nazv, plata_zaem=plata, **stroka(a["x"], a["delo"], hv, plata, a["baza"])))
        if "kontrol" in port:
            d = np.abs(port["kontrol"]["x"] - port["baza"]["x"])
            print(f"контроль razmer=1e6 против базы ({imya}): max|Δ| = {d.max():.3e}, средний {d.mean():.3e}")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "a_strogaya_yakor.csv", index=False)
    pas = pd.read_csv(ROOT / "docs" / "tables" / "a_strogaya" / "pasport.csv")
    for imya in ("yakor", "strogaya"):
        a = pas[(pas.ryad == "A") & (pas.variant == imya) & (pas.god == "все") & (pas.plata_zaem == 0)].msr.iloc[0]
        b = df[(df.portfel == imya) & df.stroka.str.startswith("(а)") & (df.plata_zaem == 0)].msr.iloc[0]
        print(f"сверка с паспортом {imya}: паспорт {a:.6f}, расчёт {b:.6f}, |Δ| = {abs(a - b):.2e}")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
