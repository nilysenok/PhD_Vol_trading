"""V6 лист Z2, G63: точный лучший результат SBER по правилу, объявленному до счёта. Минимум среднего QLIKE на общих датах среди PIT-сопоставимых рядов таблицы (K1, K2, K3, M, K8 и гибриды (а), (б)) отдельно для окон
«вся история», 2017–2021, 2022–2025; отрыв от второго ряда, от K2 и от K3 (Δ в % уровня K1, 90 % интервал, DM p); членство в MCS 90 %. Ряд класса «потолок» в выбор не входит (`assert` в `best_set`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_cmp_reg as R, v6_cmp_tab as T

REFS = ("второй", "K2", "K3")


def ext_set(u: dict) -> list:
    """Расширенное множество: все PIT-сопоставимые ряды листа 0 (+ контроли, D2) и конфигурации V6 кроме A0g."""
    pr = R.prior()
    ok = [s for s in u["f"] if s in R.CTRL or s == "D2" or (s.startswith("V6:") and s != "V6:A0g") or (s in pr.index and pr.loc[s]["class"] == R.PIT and pd.isna(pr.loc[s]["equal_to"]))]   # точные дубли контролей — не отдельные ряды
    return list(dict.fromkeys(ok))


def sets(u: dict, reg: pd.DataFrame) -> dict:
    cand = R.candidates(reg)
    return {T.SET_MAIN: cand, T.SET_NO6: [s for s in cand if not s.startswith("V6:")], T.SET_EXT: ext_set(u)}


def best_set(u: dict, ser: list, w: str) -> list:
    """-> ряды набора по возрастанию QLIKE окна [(ряд, QLIKE)]."""
    assert all(R.klass(s)[0] in (R.PIT, R.CTL) for s in ser),"в выбор лучшего попал ряд класса «потолок»"
    return sorted(((s, float(T.win_q(u, s, w).mean())) for s in ser), key=lambda x: (x[1], x[0]))


def verdict(best: str, members: list) -> str:
    if not members:
        return "MCS для этого множества не считался"
    if len(members) == 1:
        return "единственный в MCS 90 %"
    rest = [m for m in members if m != best]
    return "лучший по точке, статистически не отделим от: " + ", ".join(rest) if best in members else "лучший по точке не входит в MCS 90 %"


def rows_of(u: dict, name: str, ser: list, w: str, mc: pd.DataFrame) -> list:
    rk = best_set(u, ser, w)
    (b, qb), (s2, q2) = rk[0], rk[1]
    mm = mc[(mc["set"] == name) & (mc["window"] == w) & mc["in_mcs90"]]["model"].tolist()
    base = dict(set=name, window=w, best=b, best_cls=R.klass(b)[0], best_qlike=qb, n_series=len(ser), mcs_size=len(mm) if name == T.SET_MAIN else np.nan,
                mcs_members=";".join(mm) if name == T.SET_MAIN else "", best_in_mcs90=(b in mm) if name == T.SET_MAIN else np.nan, verdict=verdict(b, mm) if name == T.SET_MAIN else verdict(b, []))
    out = []
    for kind, ref in zip(REFS, (s2, "K2", "K3")):
        r = dict(base, ref_kind=kind, ref=ref, ref_qlike=float(T.win_q(u, ref, w).mean()))
        r.update(T.one_test(u, b, ref, w) if ref != b else dict(delta_pct=np.nan, lo90=np.nan, hi90=np.nan, p_one=np.nan, n=np.nan))
        out.append(r)
    return out


def best(u: dict, reg: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    rows = [r for name, ser in sets(u, reg).items() for w in R.Q_WIN for r in rows_of(u, name, ser, w, mc)]
    cols = ["set", "window", "best", "best_cls", "best_qlike", "n_series", "ref_kind", "ref", "ref_qlike", "delta_pct", "lo90", "hi90", "p_one", "n", "mcs_size", "mcs_members", "best_in_mcs90", "verdict"]
    return pd.DataFrame(rows)[cols]
