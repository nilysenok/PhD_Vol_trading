"""К-3 (σ̂ C/D на wf, шаги 9–10): D13 `f_intraday` на 10-минутках без выходных сессий MOEX (ДСВД).

Справка: docs/wf_sigma_910_dsvd.md, п. 2 («gap_on») и п. 3. Формулы — d13_lib.intraday_one без правки; меняется только вход:
из баров убираются даты ДСВД (по календарю MOEX, не по дню недели; рабочая суббота 01.11.2025 остаётся) и всё позже KONEC.
gap_on понедельника после ДСВД поэтому считается от закрытия пятницы. Дни ≤ 2024 не затронуты.

Запуск: FASTBT_MAIN=<корень данных> python3 scripts/dannye/wf_intraday_bez_dsvd.py --out <каталог> [--konec 2026-02-03]
Пишет только <out>/data/regen/D13/intraday1850.parquet и <out>/proverka_k3.txt; в data/ основной копии не пишет.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/analysis"), str(REPO / "scripts/dannye")]
import d13_lib as T  # noqa: E402
import l2_lib as L  # noqa: E402
import wf_baza_bez_dsvd as B  # noqa: E402

KONEC = pd.Timestamp("2026-02-03")  # данные с 2026-02-04 не читаются (решение 30.10)
NAME = "intraday1850.parquet"


def bary_bez_dsvd(c, dsvd, konec=KONEC):
    """Бары дат ДСВД и позже konec убираются; остальное (в том числе рабочие субботы) — без изменений."""
    d = pd.to_datetime(c["begin"]).dt.normalize()
    return c[(d <= konec) & ~d.isin(dsvd)]


def sobrat(bars, dsvd, konec=KONEC, filtr=bary_bez_dsvd):
    """bars: {ticker: бары}. filtr — точка подмены для мутационных тестов."""
    r = [T.intraday_one(filtr(c, dsvd, konec), tk) for tk, c in bars.items()]
    return pd.concat(r, ignore_index=True)


def chitat(tickers, konec=KONEC):
    """Бары тикеров; строки позже konec отбрасываются сразу после чтения."""
    out = {}
    for tk in tickers:
        c = pd.read_parquet(T.BARS / f"{tk}.parquet")
        out[tk] = c[pd.to_datetime(c["begin"]).dt.normalize() <= konec]
    return out


def sdvig(new, rv_matrix, kolonka="rv_half2", ks=(-2, -1, 0, 1)):
    """Д1: корреляция ln(колонка_t) с ln(rv_{t+k}) матрицы (k строк тикера); максимум обязан быть на k=0."""
    m = new.merge(rv_matrix[["date", "ticker", "rv"]], on=["date", "ticker"]).sort_values(["ticker", "date"])
    m = m[(m[kolonka] > 0) & (m.rv > 0)]
    x = np.log(m[kolonka])
    res = {}
    for k in ks:
        y = np.log(m.rv).groupby(m.ticker).shift(-k)
        res[k] = float(x.corr(y))
    return res


def gap_nezavisimo(bars, dsvd, daty, konec=KONEC):
    """Независимый расчёт gap_on для дат `daty` (по тикеру): ln(open первого бара окна / close последнего бара прошлого дня без ДСВД)."""
    res = {}
    for tk, c in bars.items():
        c = bary_bez_dsvd(c, dsvd, konec).assign(d=lambda x: pd.to_datetime(x["begin"]).dt.normalize())
        mins = pd.to_datetime(c["begin"]).dt.hour * 60 + pd.to_datetime(c["begin"]).dt.minute
        days = sorted(c.d.unique())
        for dt in daty:
            if dt not in days or days.index(dt) == 0:
                continue
            prev = c[c.d == days[days.index(dt) - 1]].sort_values("begin")["close"].iloc[-1]
            w = c[(c.d == dt) & (mins >= 590) & (mins < 1130)].sort_values("begin")
            res[(tk, dt)] = float(np.log(w["open"].iloc[0] / prev)) if len(w) else np.nan
    return res


def ponedelniki_posle_dsvd(dsvd, kal_dni):
    """Торговые даты (из ряда kal_dni), непосредственно следующие за серией ДСВД."""
    return [d for d in kal_dni if (d - pd.Timedelta(days=1)) in dsvd or (d - pd.Timedelta(days=2)) in dsvd]


def proverka(new, old, bars, dsvd, rv_matrix, konec=KONEC):
    """G1–G4 на реальных данных; возвращает строки отчёта и словарь булевых итогов."""
    k = new.sort_values(["ticker", "date"]).reset_index(drop=True)
    o = old[(old.date < "2025-01-01")].sort_values(["ticker", "date"]).reset_index(drop=True)
    k0 = k[k.date < "2025-01-01"].reset_index(drop=True)
    g1 = k0.shape == o.shape and k0.equals(o[k0.columns])
    ds_in = int(new.date.isin(dsvd).sum())
    g2 = ds_in == 0 and bool((new.date == "2025-11-01").any())
    pm = ponedelniki_posle_dsvd(dsvd, sorted(new.date.unique()))
    ind = gap_nezavisimo(bars, dsvd, pm, konec)
    got = new.set_index(["ticker", "date"]).gap_on
    d3 = [abs(got.get(key, np.nan) - v) for key, v in ind.items() if np.isfinite(v) and abs(v) < 0.4]
    g3 = len(d3) > 0 and max(d3) < 1e-12
    s = sdvig(new, rv_matrix)
    g4 = max(s, key=s.get) == 0
    lines = [f"G1 <=2024 побайтно с текущим f_intraday: {g1} (строк {len(k0)})",
             f"G2 дат ДСВД в выходе: {ds_in}; 01.11.2025 есть: {bool((new.date == '2025-11-01').any())}; итог {g2}",
             f"G3 понедельников после ДСВД: {len(pm)}, сверено gap_on: {len(d3)}, макс. расхождение {max(d3) if d3 else None}: {g3}",
             f"G4 Д1 corr(ln rv_half2_t, ln rv_(t+k)) {s}; argmax k = {max(s, key=s.get)}: {g4}",
             f"последняя дата выхода {new.date.max().date()}; строк {len(new)}"]
    return lines, dict(G1=g1, G2=g2, G3=g3, G4=g4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--konec", default=KONEC.strftime("%Y-%m-%d"))
    a = ap.parse_args()
    konec = pd.Timestamp(a.konec)
    dsvd = B.daty_dsvd()
    tickers = sorted(L.SECTOR)
    bars = chitat(tickers, konec)
    new = sobrat(bars, dsvd, konec)
    dst = Path(a.out) / "data/regen/D13"
    dst.mkdir(parents=True, exist_ok=True)
    new.to_parquet(dst / NAME, index=False)
    old = pd.read_parquet(T.INTRA_CACHE)
    rv = pd.read_parquet(L.REGEN / "features_pit_v3.parquet", columns=["date", "ticker", "rv"])
    lines, ok = proverka(new, old, bars, dsvd, rv, konec)
    (Path(a.out) / "proverka_k3.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
