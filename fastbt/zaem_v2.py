"""Плата за заём шорта как отдельный ряд рядом с ledger v2.0 (Ж-088 п. 5, поправка Ж-089; формат ledger не меняется).

Правило: плата только за перенос короткой позиции через ночь. Вход и выход в один торговый день — 0. Иначе
ставка × стоимость шорта × число календарных дней между датой входа и датой выхода / 365 (выходные и праздники
включены). Лонг — 0. Плата линейна по ставке, поэтому пишется база `zaem_baza[T]` (float64, доли капитала портфеля):
excess с платой r (доля, 0.05 = 5 %) = `excess − r · zaem_baza`.

Начисление распределено по ночам удержания: строка k (интервал [close k; close k+1)) получает
вес × (дата k+1 − дата k) / 365 для k от дня входа до дня выхода − 1. Сумма по сделке = вес × календарные дни / 365 ровно;
сделка, вошедшая до начала окна (прогрев), платит с первой строки окна (как комиссия входа по Н5).
Стоимость шорта = вес сделки (доля капитала слота × цена входа / цена входа), деление на число тикеров даёт доля портфеля.

Шарды: p{NN}/zaem/{shard:05d}.npy, float64 [≤ 256, T], номер и порядок строк те же, что у p{NN}/port/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from fastbt import rezhim_e1
from fastbt.etap1_eval import NET, years_of

SUB, SUMS, DNEI = "zaem", "SHA256SUMS_zaem.txt", 365.0


def baza_tikera(trades, dates, w, a: int, b: int) -> np.ndarray:
    """float64 [b − a]: база платы тикера по строкам окна [a; b), в весах сделок (до деления на число тикеров)."""
    gap = np.diff(np.asarray(dates).astype("datetime64[D]")).astype(np.int64)      # календарных дней строки k → k+1
    out = np.zeros(b - a)
    for tr in trades:
        if tr.side >= 0:                                                            # лонг — всегда 0
            continue
        lo = max(tr.entry_day, a)
        hi = min(b if tr.exit_day is None else tr.exit_day, b)                      # открытая на конце — до дня b
        if hi > lo:                                                                 # вход и выход в один день → пусто
            out[lo - a:hi - a] += float(w[tr.d_sig]) * gap[lo:hi] / DNEI
    return out


def ves(ev, pt: dict, t: str, b: int, anc: bool) -> np.ndarray:
    """Вес сделки по дню сигнала — как в Evaluator._rows/EvaluatorV2._rows2; лестница — половина на каждую ногу."""
    w = np.ones(b + 1)
    if not anc and pt["razmer"] != "ravnyy":
        sg = ev.inp.sigma[pt["sigma"]][t][:b + 1] * float(ev._ctx["vozm"].get("sigma_mult", 1.0))
        w = rezhim_e1.razmer_ves(sg, "obratnyj", float(pt["razmer"]) / 100.0)
    return w / 2 if ev._ctx["nov"]["lestnica"] != NET else w


def ryad(ev, period: str) -> np.ndarray:
    """float64 [T] — база платы последнего ev.excess(), в порядке port (годы периода); в строку ledger не входит."""
    pt, anc, rows = ev._zctx
    df, inv, eng, _, _ = ev._last
    ch = [baza_tikera(r.trades, ev.inp.meta[ev.sk.tickers[ti]]["dates"][:b + 1],
                      ves(ev, pt, ev.sk.tickers[ti], b, anc), a, b) for ti, r, a, b in rows]
    return baza_portfelya(ch, inv, eng.cnt, np.isin(df.index.year, years_of(period)))


def baza_portfelya(chasti: list, inv: np.ndarray, cnt: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """chasti — baza_tikera по тикерам в порядке строк `inv`; результат — по датам портфеля, только `keep` (годы периода)."""
    return (np.bincount(inv, np.concatenate(chasti), len(cnt)) / cnt)[keep]


class ZaemPisatel:
    """Шарды zaem/ в ногу с ledger_v2.Pisatel (тот же номер шарда и порядок строк); сам Pisatel не меняется."""

    def __init__(self, pisatel):
        self.p, self.buf = pisatel, []
        self.d = pisatel.pd / SUB
        self.d.mkdir(parents=True, exist_ok=True)
        zak = {int(f.stem) for f in (pisatel.pd / "port").glob("*.npy")}
        for f in self.d.iterdir():                                                   # обрыв: шард без port — удаляется
            if not f.name.endswith(".npy") or int(f.stem) not in zak:
                f.unlink()

    def add(self, row: dict, zaem) -> None:
        """Вызывать ДО Pisatel.add той же строки: zaem на диске раньше port, значит у каждого port он есть."""
        z = zaem if row.get("status") == "ok" and zaem is not None else np.full(self.p.T, np.nan)
        if z.dtype != np.float64 or z.shape != (self.p.T,):
            raise TypeError("zaem: float64 [T]")
        self.buf.append(z)
        if len(self.p.buf) + 1 >= self.p.shard:                                      # эта строка закроет шард port
            self.close()

    def close(self) -> None:
        if not self.buf:
            return
        p = self.d / f"{self.p.next:05d}.npy"
        with open(f"{p}.part", "wb") as f:
            np.save(f, np.stack(self.buf))
        Path(f"{p}.part").rename(p)
        with open(self.p.d / SUMS, "a", encoding="utf-8") as f:
            f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {self.p.potok}/{SUB}/{p.name}\n")
        self.buf = []


def _zp(pisatel) -> ZaemPisatel:
    if getattr(pisatel, "_zaem", None) is None:
        pisatel._zaem = ZaemPisatel(pisatel)
    return pisatel._zaem


def pisat(pisatel, row: dict, ex_port, ex_tik, zaem) -> None:
    """Вместо Pisatel.add: пишет ряд платы и ту же строку в ledger."""
    _zp(pisatel).add(row, zaem)
    pisatel.add(row, ex_port, ex_tik)


def zakryt(pisatel) -> None:
    """Перед Pisatel.close: дописывает хвост шарда zaem/."""
    _zp(pisatel).close()


def zaem_ryad(root, sid: str, ref: dict) -> np.ndarray:
    """float64 [T] — zaem_baza строки ledger; ref — строка ledger или её поле `excess` ({potok, shard, stroka})."""
    r = ref.get("excess", ref)
    f = Path(root) / sid / r["potok"] / SUB / f"{int(r['shard']):05d}.npy"
    return np.array(np.load(f, mmap_mode="r")[int(r["stroka"])])


def excess_s_platoy(port, baza, stavka: float) -> np.ndarray:
    """excess с платой: stavka — доля годовых (0.05 = 5 %); 0 → port без изменений."""
    return np.asarray(port, float) - float(stavka) * np.asarray(baza, float)
