"""Оценщик итераций 3, 4 и 10 Ж-051 (К0-х): подкласс `AnsamblEvaluator` + сторона входа (К4) + распределитель (К3).

Оси выключены (`top_k`, `sektor_trend`, `shirina` = net) — `_rows` отдаёт родителя без единого вызова новых модулей,
результат побайтно совпадает с `AnsamblEvaluator`. Включены — два прохода: (1) сигнал тикера (обычный или ансамбль
`self.ens`) и фильтр входа по стороне; (2) распределитель по всем тикерам; затем `run_e1`. Фильтры только запрещают вход
(сделка Sk пропускается целиком), открытую позицию не закрывают. `etap1_eval.py` не правится.

Интерфейс модулей (проект #360, §2.3–2.4); модуль не слит — ленивый импорт, при вызове `NotImplementedError`:
 - `fastbt.e1_sektor.allow(pt, t, dates) -> (long: bool[n], short: bool[n])`, n = len(dates); разрешение входа по close d;
   контроль (к2) — по `pt["kontrol"] = ("krug_sdvig", lag)`;
 - `fastbt.e1_portfel.raspredelit(sig, ball, k, sektor=None, n_sek=None, dates=None) -> dict`: `sig`, `ball`, `dates` —
   словари тикер → массив; возвращает словарь тикер → позиции после слотов (без вытеснения, вес не меняется).
"""
from __future__ import annotations

import importlib

import numpy as np

from fastbt import rezhim_e1
from fastbt.etap1_ansambl_eval import AnsamblEvaluator
from fastbt.etap1_data import _one_bar, align_b, gate_trades
from fastbt.etap1_eval import NET, PERIODS, REZHIM, RM_OFF, params_of
from fastbt.etap1_run import run_e1
from fastbt.sweep_post import build_view

PLUS_NET = dict(top_k=NET, sektor_trend=NET, shirina=NET, ma_n=200, porog_shiriny=0.5)
PLUS_OSI = ("top_k", "sektor_trend", "shirina")


def plus_on(pt: dict) -> bool:
    return any(pt.get(a, NET) != NET for a in PLUS_OSI)


def modul(name: str):
    """Ленивый импорт модуля К3/К4; тесты подменяют эту функцию."""
    try:
        return importlib.import_module(f"fastbt.{name}")
    except ImportError as e:
        raise NotImplementedError(f"fastbt.{name} ещё не слит (К3/К4)") from e


def side_gate(d: np.ndarray, long_ok: np.ndarray, short_ok: np.ndarray) -> np.ndarray:
    """Лонг-сделки гейтятся маской лонга, шорт-сделки — маской шорта; смена знака без нуля — две сделки."""
    return gate_trades(np.maximum(d, 0.0), long_ok) + gate_trades(np.minimum(d, 0.0), short_ok)


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
        return np.abs(self._strength(t, sp, q, s0, b)) / np.maximum(np.asarray(sg, float), 1e-12)

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
                lg, sh = modul("e1_sektor").allow(pt, t, dates)
                d = side_gate(d, np.asarray(lg, bool), np.asarray(sh, bool))
            sg = inp.sigma[pt["sigma"]][t][:b + 1]
            S[t] = dict(d=d, mag=mag, dates=dates, b=b, s0=s0, sg=sg)
        if pt["top_k"] != NET:                                # проход 2: слоты по всем тикерам
            ball = {t: self._ball(pt, t, sp, q, s["s0"], s["b"], s["sg"]) for t, s in S.items()}
            ball = self._kontrol_rang(pt, ball)
            new = modul("e1_portfel").raspredelit({t: s["d"] for t, s in S.items()}, ball, int(pt["top_k"]),
                                                  dates={t: s["dates"] for t, s in S.items()})
            for t, s in S.items():
                s["d"] = np.asarray(new[t], float)
        out = []
        for ti, t in enumerate(self.sk.tickers):
            s, inp = S[t], self.inp
            a = int(np.searchsorted(s["dates"], np.datetime64(f"{y0}-01-01")))
            days = inp.bars[t][:s["b"]] + [_one_bar(float(inp.close_n6[t][s["b"]]))]
            w = np.ones(len(s["d"]))
            if pt["razmer"] != "ravnyy":
                w = rezhim_e1.razmer_ves(s["sg"], "obratnyj", float(pt["razmer"]) / 100.0)
            p = params_of(pt, inp.tick[t], h, False, self.comm)
            res = run_e1(t, days, s["d"], s["sg"], p, close=inp.close_n6[t][:s["b"] + 1], w=w * s["mag"], win=(a, s["b"]))
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
