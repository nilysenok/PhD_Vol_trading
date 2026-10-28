"""Оценщик итераций 3, 4 и 10 Ж-051 (К0-х): подкласс `AnsamblEvaluator` + сторона входа (К4) + распределитель (К3).

Оси выключены (`top_k`, `sektor_trend`, `shirina` = net) — `_rows` отдаёт родителя без единого вызова новых модулей,
результат побайтно совпадает с `AnsamblEvaluator`. Включены — два прохода: (1) сигнал тикера (обычный или ансамбль
`self.ens`) и фильтр входа по стороне; (2) распределитель по всем тикерам; затем `run_e1`. Фильтры только запрещают вход
(сделка Sk пропускается целиком), открытую позицию не закрывают. `etap1_eval.py` не правится.

Модули К3/К4 (#363, #364), ленивый импорт:
 - `e1_sektor.allow(pt, t, dates, ctx) -> (long, short)`; `ctx` — `Ctx` из индексов секторов и close Н6 (`_ctx`).
   Контроль к2 (`pt["kontrol"] = ("krug_sdvig", lag)`) делает оценщик: маски разрешения сдвигаются циклически;
 - `e1_portfel.raspredelit(sig, ball, k)` — массивы (T, n) на общем календаре тикеров (`_slots`), балл — `ball_sily`.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from fastbt import rezhim_e1
from fastbt.etap1_ansambl_eval import AnsamblEvaluator
from fastbt.etap1_data import _one_bar, align_b, gate_trades
from fastbt.etap1_eval import NET, PERIODS, REZHIM, RM_OFF, params_of
from fastbt.etap1_run import run_e1
from fastbt.sweep_post import build_view

PLUS_NET = dict(top_k=NET, sektor_trend=NET, shirina=NET, ma_n=200, porog_shiriny=0.5)
PLUS_OSI = ("top_k", "sektor_trend", "shirina")


OBE, LONG = "obe", "long"          # правило стороны итерации 26 (не ось сетки): pt["storona"]; нет ключа = обе стороны


def plus_on(pt: dict) -> bool:
    """Оси К3/К4 включены, либо правило «только лонг» (итерация 26), либо плата за шорт > 0."""
    return any(pt.get(a, NET) != NET for a in PLUS_OSI) or pt.get("storona", OBE) != OBE or float(pt.get("plata_shorta", 0.0)) > 0


def modul(name: str):
    """Ленивый импорт модуля К3/К4; тесты подменяют эту функцию."""
    try:
        return importlib.import_module(f"fastbt.{name}")
    except ImportError as e:
        raise NotImplementedError(f"fastbt.{name} ещё не слит (К3/К4)") from e


def side_gate(d: np.ndarray, long_ok: np.ndarray, short_ok: np.ndarray) -> np.ndarray:
    """Лонг-сделки гейтятся маской лонга, шорт-сделки — маской шорта; смена знака без нуля — две сделки."""
    return gate_trades(np.maximum(d, 0.0), long_ok) + gate_trades(np.minimum(d, 0.0), short_ok)


def only_long(d: np.ndarray) -> np.ndarray:
    """Правило «только лонг» (К-б1): `gate_trades` с allow_long=True, allow_short=False — шорт-сделки Sk пропущены целиком."""
    return side_gate(d, np.ones(len(d), bool), np.zeros(len(d), bool))


def charge_short(g: np.ndarray, trades: list, days: list, w: np.ndarray, win: tuple, plata: float) -> None:
    """Справочная плата за шортовую ногу `plata` (годовых): вес сделки × plata/252 за каждую торговую строку шорта, из g окна."""
    a, b = win
    for tr in trades:
        if tr.side >= 0:
            continue
        stop = len(days) if tr.exit_day is None else tr.exit_day          # строки дохода: row0 … последняя до дня выхода
        for r in range(max(tr.row0, a), min(stop, b)):
            if not days[r].pusto:
                g[r - a] -= float(w[tr.d_sig]) * plata / 252.0


class PlusEvaluator(AnsamblEvaluator):
    def full(self, pt: dict) -> dict:
        return {**PLUS_NET, **super().full(pt)}

    def _strength(self, t: str, sp: dict, q: float, s0: int, b: int) -> np.ndarray:
        """Сила сигнала Sk (`view.strength[0]`) по дням окна. Вызывать после `_positions` (заполняет `_sigs`)."""
        td, ind = self.sk._td[t]
        p = {**self.base, **sp, "q_sily": q, **RM_OFF[self.sid]}
        v, _ = build_view(self.sid, self._sigs[tuple(sorted(sp.items()))][t], td, ind, p, self.sk.sig_names)
        st = v.strength[0]
        return np.asarray(st[t] if isinstance(st, dict) else st, float)[s0:s0 + b + 1]

    def _signal(self, pt: dict, t: str, sp: dict, q: float, dates, s0: int, b: int):
        if self.ens is not None:
            pts, mode = self.ens
            return self._ens_signal(t, pts, mode, dates, s0, b)
        sig = self._positions(t, sp, q, False, s0 + b)[s0:s0 + b + 1].astype(float)
        sig[-1] = 0.0                                        # день b — только close для строки b−1
        allow = ~self.inp.sobytiya[pt["sobytiya"]][t]
        if pt["rezhim"] != NET:
            allow = allow & align_b(self.inp.rezhim[REZHIM.get(pt["rezhim"], pt["rezhim"])], dates)
        return gate_trades(sig, allow), np.ones(len(sig))

    def _ball(self, pt: dict, t: str, sp: dict, q: float, s0: int, b: int, sg) -> np.ndarray:
        """Балл |strength| / σ на close d (σ известна на close d). Контроль: перемешивание между тикерами делает `_rows`."""
        return modul("e1_portfel").ball_sily(self._strength(t, sp, q, s0, b), sg)

    def _ctx(self):
        """Ряды К4: индексы секторов (< 2022), close Н6 17 тикеров, карта тикер → индекс. Строится один раз."""
        if getattr(self, "_c", None) is None:
            E = modul("e1_sektor")
            idx = E.load_indexes(sorted(set(pd.read_csv(E.SEKTORY_CSV).indeks)), end=pd.Timestamp("2020-02-01"))   # 2020–2021 в память не идут
            close = {}
            for t in self.sk.tickers:
                d, c = self.inp.meta[t]["dates"], self.inp.close_n6[t]
                n = min(len(d), len(c))
                close[t] = pd.Series(np.asarray(c, float)[:n], index=pd.DatetimeIndex(d[:n]))
            self._c = E.Ctx(idx, close, E.sektor_map(E.SEKTORY_CSV, idx))
        return self._c

    @staticmethod
    def _kontrol_sdvig(pt: dict, lg, sh):
        """Контроль к2 «krug_sdvig»: маски разрешения циклически сдвинуты на lag дней (≥ 63)."""
        k = pt.get("kontrol")
        if not (k and k[0] == "krug_sdvig"):
            return lg, sh
        ok = lg | sh
        v = int(np.argmax(ok)) if ok.any() else len(ok)      # начало валидной части: до него — прогрев MA (запрет), его не двигаем
        out = []
        for m in (lg, sh):
            m = m.copy()
            if len(m) - v > 0:
                m[v:] = np.roll(m[v:], int(k[1]) % (len(m) - v))
            out.append(m)
        return tuple(out)

    @staticmethod
    def _slots(S: dict, tickers: list, ball: dict, k: int) -> None:
        """Распределитель К3 на общем календаре: массивы (T, n); результат возвращается в S[t]['d'] по датам тикера."""
        cal = np.unique(np.concatenate([S[t]["dates"] for t in tickers]))
        pos = {t: np.searchsorted(cal, S[t]["dates"]) for t in tickers}

        def col(fill, get):
            out = np.full((len(cal), len(tickers)), fill, float)
            for j, t in enumerate(tickers):
                out[pos[t], j] = get(t)
            return out
        new = modul("e1_portfel").raspredelit(col(0.0, lambda t: S[t]["d"]), col(np.nan, lambda t: ball[t]), k)
        for j, t in enumerate(tickers):
            S[t]["d"] = new[pos[t], j]

    def _rows(self, pt: dict, period: str, anchor: bool):
        if not plus_on(pt):
            return super()._rows(pt, period, anchor)         # выключено = родитель, модули не вызываются
        y0, _ = PERIODS[period]
        sp = {k: pt[k] for k in self.sk.sig_names}
        q = float(pt.get("q_sily", 0.0))
        h = 5.0 if (pt["tau"] != NET or pt["R"] == NET) else self._h(sp, q)
        S = {}
        for t in self.sk.tickers:                             # проход 1: сигнал и сторона
            m, inp = self.inp.meta[t], self.inp
            s0, b = m["s0"], self._b(t, period)
            dates = m["dates"][:b + 1]
            d, mag = self._signal(pt, t, sp, q, dates, s0, b)
            if pt["sektor_trend"] != NET or pt["shirina"] != NET:
                lg, sh = modul("e1_sektor").allow(pt, t, dates, self._ctx())
                lg, sh = self._kontrol_sdvig(pt, np.asarray(lg, bool), np.asarray(sh, bool))
                d = side_gate(d, lg, sh)
            if pt.get("storona", OBE) == LONG:
                d = only_long(d)
            sg = inp.sigma[pt["sigma"]][t][:b + 1]
            S[t] = dict(d=d, mag=mag, dates=dates, b=b, s0=s0, sg=sg)
        if pt["top_k"] != NET:                                # проход 2: слоты по всем тикерам
            ball = {t: self._ball(pt, t, sp, q, s["s0"], s["b"], s["sg"]) for t, s in S.items()}
            ball = self._kontrol_rang(pt, ball)
            self._slots(S, list(self.sk.tickers), ball, int(pt["top_k"]))
        out = []
        for ti, t in enumerate(self.sk.tickers):
            s, inp = S[t], self.inp
            a = int(np.searchsorted(s["dates"], np.datetime64(f"{y0}-01-01")))
            days = inp.bars[t][:s["b"]] + [_one_bar(float(inp.close_n6[t][s["b"]]))]
            w = np.ones(len(s["d"]))
            if pt["razmer"] != "ravnyy":
                w = rezhim_e1.razmer_ves(s["sg"], "obratnyj", float(pt["razmer"]) / 100.0)
            p = params_of(pt, inp.tick[t], h, False, self.comm)
            ww = w * s["mag"]
            res = run_e1(t, days, s["d"], s["sg"], p, close=inp.close_n6[t][:s["b"] + 1], w=ww, win=(a, s["b"]),
                          div=inp.div[t][:s["b"] + 1], fdr=self.fdr)
            if float(pt.get("plata_shorta", 0.0)) > 0:
                charge_short(res.g, res.trades, days, ww, (a, s["b"]), float(pt["plata_shorta"]))
            out.append((ti, res, a, s["b"]))
        return out

    @staticmethod
    def _kontrol_rang(pt: dict, ball: dict) -> dict:
        """Контроль «случайный ранг» (20 seed): баллы перемешаны между тикерами внутри дня. Без `kontrol` — как есть."""
        k = pt.get("kontrol")
        if not k or k[0] != "sluchaynyy_rang":
            return ball
        rng, ts = np.random.default_rng(int(k[1])), list(ball)
        n = min(len(v) for v in ball.values())
        out = {t: ball[t].copy() for t in ts}
        for i in range(1, n + 1):                             # выравнивание по хвосту: дни считаются от конца окна
            perm = rng.permutation(len(ts))
            vals = [ball[t][-i] for t in ts]
            for j, t in enumerate(ts):
                out[t][-i] = vals[perm[j]]
        return out
