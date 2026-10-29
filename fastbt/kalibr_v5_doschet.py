"""К-2 (29.10, docs/otchety/2026-10-29_gotovnost_servera.md §5 п. 6): досчёт членов калибратора v5 оценщиком — на сервере, не на Маке.
Три работы поверх `kalibr_v5` (Pul.dosschitat, `god(…, poz_fn)`) и `kalibr_v5_golos`:
1. соседи фазы 2 вне ledger — `Doschet.dosschitat(pt)`: считает точку, пишет строку в ledger (поток POTOK, v = версия ledger), ряд [17, T] — в Pul.novye (это N);
2. позиции top-10 — `Doschet.poz_fn(kl, k)`: знаковые бинарные позиции автомата (sign(expo − 2·short)), для `golos_v2` (= `execlayer.vote`);
3. ряды членов — `Doschet.chlen(kl)`, `ryady_chlenov(res, dc, daty)`: дневной избыточный ряд [17, T] для к2 и «одной точки».
Данных не читает сам: оценщик и даты приходят в `sr` (Sreda драйвера); даты ≥ 2020 — отказ. Контракт К-1 (пул на суммах 2.1) —
`Pul.dosschitat = Doschet.dosschitat` возвращает (ex[17, T] дневной, None); пул на суммах переводит его `ledger_v2.summy` сам."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fastbt import canon_v5 as cv5, ledger_v2 as lm, progon_sobol as PS
from fastbt.data import year_bounds
from fastbt.etap1_eval import years_of
from fastbt.kalibr_v5 import klyuch

POTOK, ISTOCHNIK = 90, "doschet_faza2"      # p90/ — отдельный поток писателя; процессы счёта Соболя занимают p00…p(N−1)


def dekod_iz_kuba(kub: dict) -> dict:
    """{ось: {tok(значение): значение}} по осям куба v5 и якорям: ключи пула — строки `tok`, оценщику нужны исходные значения."""
    d = {a: {lm.tok(v): v for v in vals} for a, vals in cv5.osi(kub)}
    for a, v in kub.get("yakorya_progon", {}).items():
        d.setdefault(a, {})[lm.tok(v)] = v
    return d


def raskodirovat(pt: dict, dekod: dict) -> dict:
    """Строки `tok` → исходные значения: по карте осей; вне карты (канон-поля 'net', 'sigma') — число, если разбирается, иначе строка."""
    out = {}
    for a, v in pt.items():
        if not isinstance(v, str):
            out[a] = v
        elif v in dekod.get(a, {}):
            out[a] = dekod[a][v]
        else:
            try:
                out[a] = float(v)
            except ValueError:
                out[a] = v
    return out


def poz_znak(evr, period: str) -> np.ndarray:
    """Знаковые бинарные позиции тикеров [17, T] последнего evaluate_v2: sign(expo − 2·short) на дне; дней без строки — 0.
    Раскладка по датам — как `EvaluatorV2._poz` (keep/col/inv); expo = |pos|·доля баров, short — его шорт-часть (etap1_run.py:35)."""
    df, inv, _, _, rs = evr._last
    keep = np.isin(df.index.year, years_of(period))
    col = np.cumsum(keep) - 1
    out, lo = np.zeros((len(rs), int(keep.sum()))), 0
    for ti, r in enumerate(rs):
        sl = slice(lo, lo + len(r.g))
        lo = sl.stop
        sh = 0.0 if r.short is None else np.asarray(r.short, float)
        ok = keep[inv[sl]]
        out[ti, col[inv[sl]][ok]] = np.sign(np.asarray(r.expo, float) - 2.0 * sh)[ok]
    return out


class Doschet:
    """Досчёт на сервере: sr — Sreda драйвера (evr, kod, vhody, daty, tikery); root — ledger стратегии; rz — Rezhim прохода."""

    def __init__(self, sr, sid: str, root: Path, rz, dekod: dict, potok: int = POTOK, znak=poz_znak):
        if sr.daty.size == 0 or np.asarray(sr.daty, "datetime64[D]").max() >= np.datetime64("2020-01-01"):
            raise ValueError(f"{sid}: пустые даты или даты ≥ 2020 — отказ")
        m = lm.Chitatel(root, sid).manifest
        if (m["kod"], m["vhody"]) != (sr.kod, sr.vhody):
            raise ValueError(f"{sid}: kod/vhody оценщика ≠ manifest ledger — отказ (досчёт не смешивается с чужим кодом)")
        self.sr, self.sid, self.rz, self.dekod, self.znak, self.v = sr, sid, rz, dekod, znak, m["v"]
        self.kod_slit, self.poz_konec, self.T = bool(m.get("kod_slit")), bool(m.get("poz_konec")), int(sr.daty.size)
        self.w = lm.Pisatel21(root, sid, potok, m, daty=sr.daty, poz=self.poz_konec) if self.v == lm.V21 else lm.Pisatel(root, sid, potok, m)
        self.zapisano, self._chleny = set(), {}

    def _schitat(self, pt: dict) -> tuple:
        """(строка ledger, ex_port, ex_tik[17, T]) оценщиком, медленным путём; точка — исходные значения."""
        return PS.schitat_tochku(self.sr, self.sid, -1, raskodirovat(pt, self.dekod), ISTOCHNIK, self.kod_slit, False, self.v, self.rz)

    def dosschitat(self, pt: dict):
        """Pul.dosschitat: (ex[17, T], None). Строка пишется в ledger один раз (по klyuch_sha); не посчитано — nan (сосед отпадает)."""
        r, ex_p, ex_t = self._schitat(pt)
        ok = r["status"] == "ok"
        if r["klyuch_sha"] not in self.w.done() | self.zapisano:
            poz = self.sr.evr.poz if ok and self.v == lm.V21 and self.poz_konec else None
            self.w.add(r, ex_p, ex_t, poz) if self.v == lm.V21 else self.w.add(r, ex_p, ex_t)
            self.zapisano.add(r["klyuch_sha"])
        ex = np.asarray(ex_t, float) if ok else np.full((lm.NTIK, self.T), np.nan)
        return (ex.astype(np.float32).astype(float) if self.v == lm.V else ex), None     # 2.0 хранит tik в float32 — сосед в той же точности

    def chlen(self, kl: str) -> tuple:
        """(ex[17, T], poz[17, T] знаковые бинарные) по ключу точки `kalibr_v5.klyuch`; в ledger не пишется (точка там уже есть)."""
        if kl not in self._chleny:
            r, _, ex_t = self._schitat(json.loads(kl))
            if r["status"] != "ok":
                raise ValueError(f"член {kl}: оценщик вернул {r['status']} — голосовать нечем")
            ex, poz = np.asarray(ex_t, float), self.znak(self.sr.evr, self.rz.period)
            if ex.shape != (lm.NTIK, self.T) or poz.shape != ex.shape:
                raise ValueError("ряды члена: [17, T] длины пула")
            self._chleny[kl] = (ex, poz)
        return self._chleny[kl]

    def poz_fn(self, kl: str, k: int) -> np.ndarray:
        """poz_fn для `kalibr_v5.god` / `kalibr_v5_golos.golos_v2`: позиция тикера k на всех днях пула."""
        return self.chlen(kl)[1][k]

    def zakryt(self) -> None:
        self.w.close()


def ryady_chlenov(res: dict, dc: Doschet, daty: np.ndarray) -> dict:
    """{год t: {тикер k: ex[члены, дни года t]}} по итогу `kalibrovat` — ряды для к2 и «одной точки»; тикер без членов пропущен."""
    out = {}
    for t, g in res.items():
        s, e = year_bounds(np.asarray(daty, "datetime64[D]"), t)
        out[t] = {k: np.stack([dc.chlen(kl)[0][k, s:e] for kl in tk["chleny"]]) for k, tk in g["tikery"].items() if tk}
    return out
