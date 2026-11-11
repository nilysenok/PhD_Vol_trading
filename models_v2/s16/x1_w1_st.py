"""W1 (Ж-250 п. 4): причинно доступные состояния S1…S8 (всё известно на закрытии d) для разности потерь X1 − K2 и X1 − K3 на окне M; смещение; потолок-оракул.
Определения терцилей — как DE0 (`models_v2/s15/de0_groups.terciles_past`: порог дня i по значениям признака в днях < i), но MIN_PAST = 252 (задание W1); пока прошлых < 252 — группа «нет порога».
Уже было (DE0 `de0_groups`, ZD/ZE `x1_*_states`, DX V7): уровень ln RV, изменение к 5 дням, знак доходности, экстремальный бар, расхождение |ln(K2/H)|, e_d X1. Новое здесь: S1 (уровень по K1_d), S3 (шок к 21 дню), S4 (доля скачка),
S5 (ln X1 − ln K), S6 (разрыв календаря, день недели t⁺), S7 (|доходность IMOEX|), S8 (недавняя относительная ошибка по 20 закрытым парам), смещение ln(RV/F)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s13 import v6_0_q as Q
from models_v2.s15 import de0_groups as G
from models_v2.s16 import x1_t0 as T0, x1_w1_join as J

MIN_PAST, LAST_PAIRS, MIN_G = 252, 20, 10
NAMES = ("T1 низ", "T2 середина", "T3 верх")
NONE = "нет порога"
DESC = {"S1": "уровень: терцили прогноза K1_d", "S2": "состояние e_d X1 (E/N)", "S3": "шок ln RV_d − ln mean(RV_{d−21…d−1}), терцили", "S4": "доля скачка J_d/RV_d, терцили", "S5s": "знак ln X1_d − ln K_d",
        "S5t": "терцили ln X1_d − ln K_d", "S6g": "разрыв календаря d → t⁺", "S6w": "день недели t⁺", "S7": "|доходность IMOEX дня d|, терцили", "S8s": "знак среднего (l_X1 − l_K) по 20 закрытым парам", "S8t": "терцили среднего (l_X1 − l_K) по 20 закрытым парам"}


def tercile(x: np.ndarray) -> np.ndarray:
    """Метки 0/1/2 или −1 (нет порога); порог дня i — только по x[:i] (DE0)."""
    return G.terciles_past(np.asarray(x, float), MIN_PAST)


def lab(ix: np.ndarray, names=NAMES) -> np.ndarray:
    return np.array([names[i] if i >= 0 else NONE for i in ix], dtype=object)


def closed_end(tp: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Число закрытых к d пар: строки с t⁺ ≤ d (строка дня d сама не закрыта: её t⁺ > d)."""
    return np.searchsorted(tp, d, side="right")


def s8_feature(dif: np.ndarray, tp: np.ndarray, d: np.ndarray, last: int = LAST_PAIRS) -> np.ndarray:
    """Среднее (l_X1 − l_K) по последним `last` закрытым парам (t⁺ ≤ d); меньше `last` закрытых — NaN."""
    k = closed_end(tp, d)
    return np.array([dif[e - last:e].mean() if e >= last else np.nan for e in k])


def daily_inputs(fr: dict) -> pd.DataFrame:
    """Дни SBER: rv, j (архив daily_SBER + продолжение: rv — из рядов по t⁺, j — запечатанная RV, замок открыт) и |доходность IMOEX| дня (последний бар основной сессии / предыдущий)."""
    from models_v2.s17.fwd_src import Src
    a = pd.read_parquet(T0.REG / "harj_panel_hybrid/daily_SBER.parquet")[["rv", "j"]]
    s = Src().sealed_rv()[["rv", "j"]]
    k1 = fr["K1"].set_index("t_plus")["rv"]
    new = pd.DataFrame({"rv": k1[k1.index > J.ARCH_END]}).join(s["j"])
    x = pd.concat([a, new]).sort_index()
    x.index = pd.to_datetime(x.index)
    b = D.main_session(Src().bars("IMOEX", pd.Timestamp("2026-11-10")))
    c = b.groupby(b["begin"].dt.normalize())["close"].last()
    x["imoex_abs"] = np.abs(np.log(c).diff()).reindex(x.index)
    return x


def day_labels(dd: pd.DataFrame) -> dict:
    """S3, S4, S7 по дням (пороги — по прошлым дням)."""
    rv = dd["rv"].to_numpy(float)
    prev = pd.Series(rv).shift(1).rolling(21).mean().to_numpy()
    x3 = np.log(rv) - np.log(prev)
    x4 = dd["j"].to_numpy(float) / rv
    return {k: pd.Series(lab(tercile(x)), index=dd.index) for k, x in (("S3", x3), ("S4", x4), ("S7", dd["imoex_abs"].to_numpy(float)))}


def e_state(d) -> np.ndarray:
    s = pd.concat([pd.read_parquet(J.MODELS / "x1.parquet")[["d", "s"]], pd.read_parquet(J.NEW / "x1.parquet")[["d", "s"]]]).drop_duplicates("d").set_index("d")["s"]
    return np.where(s.reindex(pd.DatetimeIndex(d)).to_numpy() > 0, "E", "N")


def base_labels(fr: dict, al: dict) -> dict:
    """Состояния, не зависящие от базы."""
    d, tp = pd.DatetimeIndex(al["K1"]["d"]), pd.DatetimeIndex(al["K1"]["t_plus"])
    k1 = fr["K1"]
    out = {"S1": pd.Series(lab(tercile(k1["F"].to_numpy(float))), index=pd.DatetimeIndex(k1["d"])).reindex(d).to_numpy(), "S2": e_state(d)}
    out |= {k: v.reindex(d).fillna(NONE).to_numpy() for k, v in day_labels(daily_inputs(fr)).items()}
    gap = (tp - d).days.to_numpy()
    out["S6g"] = np.where(gap == 1, "1 день", "≥ 2 дней")
    out["S6w"] = np.array(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"], dtype=object)[tp.dayofweek.to_numpy()]
    return out


def pair_labels(al: dict, base: str) -> dict:
    """Состояния, зависящие от базы: S5 (расхождение прогнозов), S8 (недавняя относительная ошибка)."""
    d, tp = pd.DatetimeIndex(al["K1"]["d"]).to_numpy(), pd.DatetimeIndex(al["K1"]["t_plus"]).to_numpy()
    x5 = np.log(al["X1"]["F"].to_numpy(float)) - np.log(al[base]["F"].to_numpy(float))
    rv = al["K1"]["rv"].to_numpy(float)
    x8 = s8_feature(Q.ql(rv, al["X1"]["F"].to_numpy(float)) - Q.ql(rv, al[base]["F"].to_numpy(float)), tp, d)
    sgn = lambda x, n: np.where(np.isfinite(x), np.where(x > 0, n[0], n[1]), NONE).astype(object)  # noqa: E731
    return {"S5s": sgn(x5, ("X1 выше K", "X1 ниже K")), "S5t": lab(tercile(x5)), "S8s": sgn(x8, ("X1 хуже недавно", "X1 лучше недавно")), "S8t": lab(tercile(x8))}


ORDER = NAMES + ("E", "N", "X1 выше K", "X1 ниже K", "1 день", "≥ 2 дней", "Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс", "X1 лучше недавно", "X1 хуже недавно", NONE)


def order(g: str) -> int:
    return ORDER.index(g) if g in ORDER else len(ORDER)


def group_rows(pair: str, state: str, labels: np.ndarray, dif: np.ndarray, qk: float, yr: np.ndarray) -> list:
    n, tot = len(dif), 100.0 * float(dif.sum()) / (len(dif) * qk)
    rows = []
    for g in sorted(set(labels), key=order):
        m = labels == g
        mean = float(dif[m].mean())
        ys = [y for y in np.unique(yr) if (m & (yr == y)).sum() >= MIN_G]
        same = sum(1 for y in ys if np.sign(dif[m & (yr == y)].mean()) == np.sign(mean))
        rows.append(dict(pair=pair, state=state, desc=DESC[state], group=g, n=int(m.sum()), share=m.sum() / n, mean_diff=mean, contrib_pct=100.0 * float(dif[m].sum()) / (n * qk),
                         years_same_sign=same, years_with_group=len(ys), delta_total_pct=tot))
    return rows


def oracle_row(state: str, labels: np.ndarray, lx: np.ndarray, lk: np.ndarray) -> dict:
    """Потолок (использует исход): в каждой группе — лучшая из X1 и K по сумме потерь ex post."""
    q = sum(min(lx[labels == g].sum(), lk[labels == g].sum()) for g in sorted(set(labels))) / len(lx)
    return dict(pair="X1|K2", state=state, desc=DESC[state], group="ORACLE (потолок: использует исход)", n=len(lx), share=1.0, mean_diff=float(q - lk.mean()), contrib_pct=pct(q, lk.mean()),
                years_same_sign=-1, years_with_group=-1, delta_total_pct=np.nan)


def pct(q: float, qb: float) -> float:
    return 100.0 * (q - qb) / qb


def bias_rows(al: dict, labs: dict) -> list:
    rv, tp = al["K1"]["rv"].to_numpy(float), pd.DatetimeIndex(al["K1"]["t_plus"])
    keys = {"год t⁺": tp.year.astype(str).to_numpy(), "S1": labs["S1"], "S3": labs["S3"]}
    rows = []
    for s in ("X1", "K2", "K3"):
        F = al[s]["F"].to_numpy(float)
        ls, r = Q.ql(rv, F), np.log(rv / F)
        for kind, lb in keys.items():
            for g in sorted(set(lb)):
                m = lb == g
                u = m & (rv > F)
                rows.append(dict(series=s, kind=kind, group=g, n=int(m.sum()), mean_ln_rv_f=float(r[m].mean()), share_under=float((rv[m] > F[m]).mean()), qlike=float(ls[m].mean()), loss_share_under=float(ls[u].sum() / ls[m].sum())))
    return rows


def states_bias(fr: dict, al: dict) -> tuple:
    rv, tp = al["K1"]["rv"].to_numpy(float), pd.DatetimeIndex(al["K1"]["t_plus"])
    bl, yr, rows = base_labels(fr, al), pd.DatetimeIndex(al["K1"]["t_plus"]).year.to_numpy(), []
    lx = Q.ql(rv, al["X1"]["F"].to_numpy(float))
    for b in ("K2", "K3"):
        lk = Q.ql(rv, al[b]["F"].to_numpy(float))
        labs = {**bl, **pair_labels(al, b)}
        for st in DESC:
            rows += group_rows("X1|" + b, st, labs[st], lx - lk, float(lk.mean()), yr)
            if b == "K2":
                rows.append(oracle_row(st, labs[st], lx, lk))
    return pd.DataFrame(rows), pd.DataFrame(bias_rows(al, bl))
