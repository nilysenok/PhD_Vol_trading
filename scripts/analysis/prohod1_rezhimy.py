"""Разбор прохода 1: режимы (Ж-088 п. 6) и причины выхода (Ж-085 п. 6) для центра региона. Справочно, без отбора, train4.
Ключевая ставка — fastbt.msp.rate_2013 (файл ряда читается целиком, значения берутся до 2019-12-31); IMOEX — только чтение."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import yaml

import prohod1_lib as L
from fastbt import ledger_v2 as lm
from fastbt.pbo_pravilo import msr_stolbcy

IMOEX = L.MAIN / "data/iss_mix_2015_2019/IMOEX.csv"
MIN_DNEY = 21
NAN = float("nan")


def _msr(x: np.ndarray) -> float:
    return float(msr_stolbcy(np.asarray(x, float)[:, None])[0]) if len(x) >= MIN_DNEY else NAN


def stavka_po_dnyam(daty: np.ndarray) -> np.ndarray:
    """Ключевая ставка в силе на d−1 (point-in-time, без заглядывания в d); доля."""
    from fastbt import msp
    s = msp.rate_2013()
    s = s[s.index <= "2019-12-31"]
    pos = s.index.searchsorted(pd.DatetimeIndex(daty) - pd.Timedelta(days=1), side="right") - 1
    return np.where(pos >= 0, s.to_numpy()[np.maximum(pos, 0)], np.nan)


def imoex_vyshe_ma200(daty: np.ndarray) -> np.ndarray:
    """True — закрытие IMOEX выше MA200 на последнем торговом дне ДО d (MA200 по 200 закрытиям до него включительно)."""
    df = pd.read_csv(IMOEX, usecols=["TRADEDATE", "CLOSE"], parse_dates=["TRADEDATE"])
    df = df[df.TRADEDATE <= "2019-12-31"].sort_values("TRADEDATE")
    c = df.set_index("TRADEDATE")["CLOSE"].astype(float)
    flag = (c > c.rolling(200).mean()).to_numpy()
    ok = c.rolling(200).mean().notna().to_numpy()
    pos = c.index.searchsorted(pd.DatetimeIndex(daty), side="left") - 1             # последний день строго раньше d
    return np.where(pos >= 0, flag[np.maximum(pos, 0)] & ok[np.maximum(pos, 0)], False), np.where(pos >= 0, ok[np.maximum(pos, 0)], False)


def razrezy(x: np.ndarray, daty: np.ndarray) -> list:
    """mSR по годам, по ставке (ниже / не ниже медианы train4) и по IMOEX (выше / ниже MA200). Строка = (разрез, группа, дней, mSR)."""
    god = np.asarray(daty).astype("datetime64[Y]").astype(int) + 1970
    st = stavka_po_dnyam(daty)
    med = float(np.nanmedian(st))
    vyshe, est_ma = imoex_vyshe_ma200(daty)
    out = [("god", str(y), int((god == y).sum()), _msr(x[god == y])) for y in L.GODY]
    gr = [("stavka", f"nizhe_mediany_{med * 100:.2f}%", st < med), ("stavka", f"ne_nizhe_mediany_{med * 100:.2f}%", st >= med),
          ("imoex", "vyshe_MA200", est_ma & vyshe), ("imoex", "nizhe_MA200", est_ma & ~vyshe), ("imoex", "MA200_net_dannyh", ~est_ma)]
    return out + [(r, g, int(m.sum()), _msr(x[m])) for r, g, m in gr if m.sum()]


def tochka(sid: str, kl: dict) -> dict:
    """Точка сетки из ключа ledger: токен -> исходное значение оси по кубу (kub_{sid}.yaml); сверка klyuch(…) == ключ."""
    kub = yaml.safe_load((L.ROOT / f"fastbt/designs/progon/kub_{sid}.yaml").read_text(encoding="utf-8"))
    look = {n: {lm.tok(v): v for v in sp["setka"]} for sec in ("signal", "obshchie", "novye") for n, sp in kub[sec].items()}
    pt = {k: look[k][v] for k, v in kl.items() if k not in ("period", "komissiya", "fdr")}
    if lm.klyuch(sid, pt, "train4", 0.0004, None) != kl:
        raise ValueError(f"{sid}: точка из ключа не воспроизводит ключ")
    return pt


def prichiny_vyhoda(sid: str, row: dict, port_ledger: np.ndarray) -> tuple:
    """Одним вызовом EvaluatorV2 пересчитывает точку и считает причины закрытия сделок журнала.
    Возврат: (доли причин среди закрытых сделок, закрытых, открытых на конце, max|порт − ledger|)."""
    from fastbt import etap1
    from fastbt.etap1_eval_v2 import EvaluatorV2
    pt = tochka(sid, row["klyuch"])
    evr = EvaluatorV2(sid, etap1.build_inputs(sid, period="train4"), etap1.base_point(sid), 0.0004)
    _, x, _ = evr.evaluate_v2(pt, "train4")
    trades = [tr for r in evr._last[4] for tr in r.trades]
    take = "tejk_signal" if pt.get("tejk_signal", "net") != "net" else "take"
    done = [tr for tr in trades if tr.reason]                                       # как в Evaluator.evaluate: открытые не считаются
    c = Counter(take if tr.reason == "take" else tr.reason for tr in done)
    dolya = {k: c.get(k, 0) / max(len(done), 1) for k in ("signal", "vremya", "stop", take, "trail", "bu")}
    return dolya, len(done), len(trades) - len(done), float(np.abs(np.asarray(x) - port_ledger).max())
