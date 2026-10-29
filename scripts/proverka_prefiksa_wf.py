"""К-wf4 (docs/preregistraciya_v2_chernovik.md §9(а) п. 1, §11): прогон до даты cut — префикс прогона до 31.12.2021.

Для 20 точек Sk считаются дневные избыточные доходности на ряду, обрезанном по cut (19, 2019-12-31 и 2021-12-31), и на
полном ряду (≤ 2021). Сравниваются дни строго до последних `ХВОСТ` баров префикса (на последнем баре окно платит выход,
Л7b) и суммы по кварталам; top-10 точек по Шарпу окна должен совпасть. Режим `znanie` — события дивидендов с датой
реестра > cut префиксу неизвестны (так в жизни); режим `naiv` — таблица целиком (показывает артефакт обрезки).
Данные ≤ 2021 (zona dev), сеть не нужна. Запуск: FASTBT_MAIN=<своя копия> python3 scripts/proverka_prefiksa_wf.py S1 [20]
Самопроверка (без аргументов): синтетика, мутация «процентиль по всему ряду» ловится."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
TOL, HVOST, TOP = 1e-12, 3, 10
CUTS = ("2019-12-31", "2020-12-31", "2021-12-31")      # последний ряд кончается 30.12.2021: cut 2021 тривиален (контроль нуля)


def max_delta(a: pd.Series, b: pd.Series, hvost: int = HVOST) -> float:
    """max |a−b| по общим датам без последних `hvost` баров a; пустое пересечение или разные даты — inf."""
    a = a.iloc[:-hvost] if hvost else a
    if a.empty or not a.index.isin(b.index).all():
        return float("inf")
    return float(np.abs(a.to_numpy() - b.reindex(a.index).to_numpy()).max())


def kvartaly(s: pd.Series, do: str) -> pd.Series:
    """Суммы по кварталам дат <= do без последнего квартала (в нём окно платит выход на конце, Л7б)."""
    s = s[s.index <= pd.Timestamp(do)]
    return s.groupby(s.index.to_period("Q")).sum().iloc[:-1]


def top_k(sharpes: np.ndarray, k: int = TOP) -> list:
    return sorted(np.argsort(-sharpes, kind="stable")[:k].tolist())


def tochki(sid: str, n: int, seed: int = 0) -> list:
    """n точек: сигнальные параметры и RM — случайные узлы сеток (плоские ключи sweep_post.split)."""
    from fastbt import grids
    from fastbt.sigprov import SIG_GRIDS
    rng = np.random.default_rng(seed)
    rm = grids.rm_grid(sid)
    out = []
    for _ in range(n):
        p = {k: v[int(rng.integers(len(v)))] for k, v in SIG_GRIDS[sid].items()}
        r = dict(rm[int(rng.integers(len(rm)))])
        tip = r.pop("_rm_type", None)
        out.append({**p, **r, **({"rm_tip": tip} if tip else {})})
    return out


@contextmanager
def ryad_do(cut: str | None, znanie: bool):
    """Дневной ряд и таблицы дивидендов обрезаны по cut (None — без обрезки); кэши загрузчиков сброшены."""
    from fastbt import data, dividendy
    orig_df, orig_read, orig_cal = data.daily_df, dividendy._read, dividendy._calendar

    def df():
        d = orig_df()
        return d if cut is None else d[d["date"] <= pd.Timestamp(cut)].reset_index(drop=True)

    def read(path, ticker):
        if cut is None or not znanie:
            return orig_read(path, ticker)
        t = pd.read_csv(path)
        t = t[(t["ticker"] == ticker) & (t["dps_rub"] > 0) & (pd.to_datetime(t["record_date"]) <= pd.Timestamp(cut))]
        if t.empty:
            return pd.DataFrame({"last_cum_date": pd.Series(dtype="datetime64[ns]"), "dps_rub": pd.Series(dtype=float)})
        return pd.DataFrame({"last_cum_date": dividendy.last_cum(pd.to_datetime(t["record_date"])), "dps_rub": t["dps_rub"]})
    def cal():                                       # календарь торговых дней тоже обрезан (как при физической обрезке файла)
        c = orig_cal()
        return c if cut is None else c[c <= np.datetime64(pd.Timestamp(cut))]
    cal.cache_clear = lambda: None
    data.daily_df, dividendy._read, dividendy._calendar = df, read, cal
    data.load_ticker.cache_clear()
    try:
        yield
    finally:
        data.daily_df, dividendy._read, dividendy._calendar = orig_df, orig_read, orig_cal
        data.load_ticker.cache_clear()


def pnl_tochek(sid: str, pts: list, cut: str | None, znanie: bool, start_year: int = 2016) -> list:
    from fastbt import etap1, sweep_post
    with ryad_do(cut, znanie):
        ev = etap1.sk_evaluator(sid)
        end = pd.Timestamp(cut) + pd.Timedelta(days=1) if cut else pd.Timestamp("2022-01-01")
        out = []
        for p in pts:
            sigs = ev.signals({k: v for k, v in p.items() if k in ev.sig_names})
            out.append(ev.pnl(sigs, p, 0.0004, end, start_year=start_year)["excess"])
    return out


def sravnenie(sid: str, n: int, znanie: bool) -> dict:
    from fastbt import metriki_e as me
    pts, res = tochki(sid, n), {}
    polnyj = pnl_tochek(sid, pts, None, znanie)
    for cut in CUTS:
        pref = pnl_tochek(sid, pts, cut, znanie)
        d = [max_delta(a, b) for a, b in zip(pref, polnyj)]
        dq = [float(np.abs(kvartaly(a, cut).to_numpy() - kvartaly(b, cut).reindex(kvartaly(a, cut).index).to_numpy()).max())
              for a, b in zip(pref, polnyj)]
        sh = lambda ser: np.array([me.sr_dnevnoy(s[s.index <= pd.Timestamp(cut)].iloc[:-HVOST].to_numpy()) for s in ser])  # noqa: E731
        res[cut] = dict(dni=max(d), kvartaly=max(dq), top_ok=top_k(sh(pref)) == top_k(sh(polnyj)), hudshie=int(np.argmax(d)))
    return res


def samoproverka() -> None:
    idx = pd.bdate_range("2016-01-01", periods=300)
    x = pd.Series(np.random.default_rng(1).normal(0, 1, 300), index=idx)
    zscore = lambda s: (s - s.expanding(20).mean()) / s.expanding(20).std()                 # noqa: E731 причинный
    zvse = lambda s: (s - s.mean()) / s.std()                                              # noqa: E731 мутация
    pr = lambda f: max_delta(f(x.iloc[:200]).dropna(), f(x).dropna(), hvost=0)             # noqa: E731
    assert pr(zscore) <= TOL, "причинный признак: префикс обязан совпасть"
    assert pr(zvse) > 1e-3, "мутация «по всему ряду» обязана ловиться"
    assert max_delta(x.iloc[:5], x.iloc[10:], 0) == float("inf")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        samoproverka()
        print("самопроверка: ок")
    else:
        sid, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
        for zn in (True, False):
            print(sid, "znanie" if zn else "naiv", sravnenie(sid, n, zn), flush=True)
