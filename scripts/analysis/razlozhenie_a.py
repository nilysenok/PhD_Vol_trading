"""Разложение mSR и Σ PnL за train4 (2016–2019) для A_strogaya и якоря A (справочный расчёт, без выбора).
Запуск: FASTBT_BARS1M=<D25/bars1m> python3 scripts/analysis/razlozhenie_a.py  (нужны data/fastbt/s9/cache_latest_*.txt, копия на чтение).
Пишет docs/tables/a_strogaya/razlozhenie.csv и razlozhenie_sdelki.csv. Определения — fastbt/metriki_e.py (mSR, Шарп ddof=1, √252, Д7).
Σ PnL % = сумма дневного excess за 4 года × 100 (то же, что `dohodnost` паспорта). Данные ≥ 2020 не читаются (период train4)."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import a_strogaya as AS, etap1, etap1_eval as E, metriki_e as me  # noqa: E402
from fastbt.etap1_eval_v2 import EvaluatorV2  # noqa: E402
from fastbt.etap1_run import _spans  # noqa: E402

VARIANTY = {"baza": {}, "long": {"storona": 1}, "short": {"storona": -1}, "comm0": {"comm": 0.0}, "div0": {"div0": True}}
STROKI = [("baza", "база: как в паспорте (комиссия 0,04 %, дивиденды, обе стороны)"), ("long", "(а) только лонги"),
          ("short", "(а) только шорты"), ("baza", "(б) с комиссией 0,04 %"), ("comm0", "(б) без комиссии"),
          ("baza", "(в) с дивидендами"), ("div0", "(в) без дивидендов"), ("bez10", "(д) без 10 худших сделок")]


TEK = {}                      # текущий оценщик процесса (для записи сделок из подменённого run_e1 якорного пути)


def zapis(ev, rows, t, days, p, win) -> None:
    """Сделки тикера с PnL пары (g минус комиссия на входе и выходе, как в port_rows) и вкладом по дням окна."""
    a, b = win
    ix = [i for i, d in enumerate(days) if not d.pusto]
    dates, out = ev.inp.meta[t]["dates"], []
    for tr, r0, last_g, er in _spans(rows, ix, p, len(days)):
        if not (r0 < b and last_g >= a):
            continue
        lo, hi = max(r0, a), min(last_g, b - 1)
        dn = {i: float(rows.g[i - a]) for i in range(lo, hi + 1)}
        for r in (lo, min(er, b - 1)):
            dn[r] = dn.get(r, 0.0) - ev.comm
        out.append(dict(tik=t, side=tr.side, vhod=str(dates[tr.entry_day])[:10], pnl=sum(dn.values()),
                        dni={str(dates[i])[:10]: v for i, v in dn.items()}))
    ev.sdelki[t] = out


def run_e1_zapis(t, days, sig, sg, p, **kw):
    """Тот же run_e1 (якорный путь Evaluator._rows), плюс запись сделок в TEK['ev']."""
    rows = _run_e1(t, days, sig, sg, p, **kw)
    zapis(TEK["ev"], rows, t, days, p, kw["win"])
    return rows


_run_e1 = E.run_e1


@contextmanager
def zapis_sdelok():
    """Подмена E.run_e1 на run_e1_zapis только на время расчёта; на выходе — исходная функция (импорт модуля движок не трогает)."""
    E.run_e1 = run_e1_zapis
    try:
        yield
    finally:
        E.run_e1 = _run_e1


def klass(cls):
    """Подкласс оценщика: фильтр стороны позиции и запись сделок — движок не правится (подмена только в этом процессе)."""
    class Rec(cls):
        storona, sdelki = 0, None

        def _positions(self, *a):
            pos = super()._positions(*a)
            return np.where(pos * self.storona > 0, pos, 0.0) if self.storona else pos

        def _ladder(self, t, days, sig, sg, p, w, lest, kw):
            rows = super()._ladder(t, days, sig, sg, p, w, lest, kw)
            zapis(self, rows, t, days, p, kw["win"])
            return rows
    return Rec


def schitat(sid: str) -> dict:
    """Все варианты для одного Sk: {портфель: {вариант: ряд schitat + сделки}}."""
    inp, base, out = etap1.build_inputs(sid, period=AS.PER), etap1.base_point(sid), {"yakor": {}, "strogaya": {}}
    with zapis_sdelok():
        for imya, cls, strogo in (("yakor", EvaluatorV2, False), ("strogaya", AS.AStrogaya, True)):
            for v, kw in VARIANTY.items():
                i2 = replace(inp, div={t: np.zeros_like(np.asarray(d, float)) for t, d in inp.div.items()}) if kw.get("div0") else inp
                ev = klass(cls)(sid, i2, base, kw.get("comm", AS.COMM))
                ev.storona, ev.sdelki = kw.get("storona", 0), {}
                TEK["ev"] = ev
                r = AS.schitat(ev, strogo)
                r["sdelki"] = ev.sdelki
                r["tikery"] = list(ev.sk.tickers)
                out[imya][v] = r
            print(sid, imya, "готово", flush=True)
    return out


def metriki(x: np.ndarray, baza: np.ndarray, plata: float) -> dict:
    x = x - plata * baza
    return dict(msr=me.msr(x), sharpe=float(me.sr_dnevnoy(x) * np.sqrt(252)), sum_pnl_pct=float(x.sum() * 100))


def portfel(res: dict, v: str, tik_pr: dict | None = None) -> dict:
    """A из шести Sk для варианта v; tik_pr — подмена рядов пар (для «без 10 худших»)."""
    rr = {s: dict(res[s][v]) for s in AS.SIDS}
    if tik_pr is not None:
        for s in AS.SIDS:
            rr[s]["tik"] = tik_pr[s]
    return AS.portfel_A(rr)


def hudshie(res: dict, n: int = 10):
    """(10 худших сделок по PnL пары %, ряды пар без них). Вклад сделки вычитается из ряда пары по дням."""
    vse = [dict(sk=s, **tr) for s in AS.SIDS for ss in res[s]["baza"]["sdelki"].values() for tr in ss]
    vse.sort(key=lambda t: t["pnl"])
    plohie, tik = vse[:n], {s: res[s]["baza"]["tik"].copy() for s in AS.SIDS}
    for tr in plohie:
        r = res[tr["sk"]]["baza"]
        ti, dts = r["tikery"].index(tr["tik"]), r["dates"].astype("datetime64[D]")
        for d, v in tr["dni"].items():
            c = int(np.searchsorted(dts, np.datetime64(d)))
            assert dts[c] == np.datetime64(d), d
            tik[tr["sk"]][ti, c] -= v
    return plohie, tik


def main() -> None:
    with Pool(4) as p:
        per = dict(zip(AS.SIDS, p.map(schitat, AS.SIDS)))
    rows, sdelki = [], []
    for imya in ("yakor", "strogaya"):
        res = {s: per[s][imya] for s in AS.SIDS}
        port = {v: portfel(res, v) for v in VARIANTY}
        plohie, tik_pr = hudshie(res)
        port["bez10"] = port["baza"] | dict(x=portfel(res, "baza", tik_pr)["x"])
        for k, (v, nazv) in enumerate(STROKI):
            for plata in (0.0, AS.PLATA):
                rows.append(dict(portfel=imya, stroka=nazv, kod=v, plata_zaem=plata, **metriki(port[v]["x"], port[v]["baza"], plata)))
        sdelki += [dict(portfel=imya, sk=t["sk"], tiker=t["tik"], storona=t["side"], data_vhoda=t["vhod"], pnl_pct=t["pnl"] * 100) for t in plohie]
    df = pd.DataFrame(rows)
    df.to_csv(AS.OUT / "razlozhenie.csv", index=False)
    pd.DataFrame(sdelki).to_csv(AS.OUT / "razlozhenie_sdelki.csv", index=False)
    pas = pd.read_csv(AS.OUT / "pasport.csv")
    for imya in ("yakor", "strogaya"):
        a = pas[(pas.ryad == "A") & (pas.variant == imya) & (pas.god == "все") & (pas.plata_zaem == 0)].msr.iloc[0]
        b = df[(df.portfel == imya) & (df.kod == "baza") & (df.plata_zaem == 0)].msr.iloc[0]
        print(f"сверка с паспортом {imya}: паспорт {a:.6f}, расчёт {b:.6f}, |Δ| = {abs(a - b):.2e}")
    print(df.to_string())
    print(pd.DataFrame(sdelki).to_string())


if __name__ == "__main__":
    main()
