"""A_strogaya (Ж-086 п. 2, Ж-088 п. 2, 4): те же сигналы S1–S6 и та же точка A, исполнение строгое.

Вход `rynok` — по open 5-мин бара 10:00 дня d+1 (Params.vhod_open); выход по сигналу и τ — по open бара 10:00 дня x+1
(`t_vyhod = open2`, Ж-086 п. 4); close дня решения в основной строке не участвует. Лимитки k_e не используются (у A их нет).
Якорь A = s9 не трогается: режим «якорь» (EvaluatorV2 без новых осей) остаётся эталоном кода. Здесь якорь только читается — для паспорта.
Сигналы — позиции s9 с его риск-менеджментом (как у якоря); меняется исполнение, не сигнал. Данные: train4 (2016–2019, прогрев 2015).
Портфель A — как у A: равный вес по парам (тикер, Sk), без таргета (progon_urovni, `umolchaniya` = ravnyj_bez_targeta).
Плата за шорт — zaem_v2 (только за перенос через ночь); комиссия 0,04 %.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import etap1, metriki_e as me, progon_urovni as urovni
from fastbt.etap1_data import gate_trades
from fastbt.etap1_eval_v2 import EvaluatorV2
from fastbt.exec_e_types import Params

COMM, PER, GODY, PLATA = 0.0004, "train4", (2016, 2017, 2018, 2019), 0.05
SIDS = ("S1", "S2", "S3", "S4", "S5", "S6")
OUT = Path(__file__).resolve().parent.parent / "docs" / "tables" / "a_strogaya"


class AStrogaya(EvaluatorV2):
    """Режим «якорь» с новыми осями: сигнал — позиции s9, исполнение — Params.vhod_open + t_vyhod из точки."""
    STROGAYA = True

    def _params(self, pt, t, h, anchor, nov, vz):
        if anchor:
            return Params(check="bary", comm=self.comm, t_vyhod=str(nov["t_vyhod"]), vhod_open=True)
        return super()._params(pt, t, h, anchor, nov, vz)

    def _sig(self, t, sp, q, anchor, s0, b, st):
        """Вход по open дня b (первый день после окна) не входит в окно: сделка, стартовавшая в последний день окна, пропускается."""
        pos = super()._sig(t, sp, q, anchor, s0, b, st)
        return gate_trades(pos, np.arange(len(pos)) != len(pos) - 2)


def evaluatory(sid: str, ev=None, period: str | None = None) -> tuple:
    """(якорь, строгий) на общих входах периода (train4 — умолчание; test1 — реверс раунд 1): EvaluatorV2 без осей — эталон;
    AStrogaya — A_strogaya."""
    period = PER if period is None else period             # PER читается при вызове: AS.PER = "test1" действует
    inp = etap1.build_inputs(sid, ev=ev, period=period)
    base = etap1.base_point(sid)
    return EvaluatorV2(sid, inp, base, COMM), AStrogaya(sid, inp, base, COMM)


def schitat(ev: EvaluatorV2, strogo: bool, period: str | None = None) -> dict:
    """Ряд Sk через EvaluatorV2: x [T], tik [17, T], baza платы за шорт [T], занятость и счёт сторон по строкам, даты."""
    period = PER if period is None else period
    row, x, tik = ev.evaluate_v2({"t_vyhod": "open2"} if strogo else {}, period)
    _, _, dts, _, _, rdates = ev._series(period, ev.comm)
    rs = ev._last[4]
    return dict(x=x, tik=tik, baza=np.asarray(ev.zaem), dates=dts.to_numpy().astype("datetime64[D]"), row=row,
                expo=np.concatenate([r.expo for r in rs]), cnt=np.concatenate([r.cnt for r in rs]), god=rdates.year.to_numpy())


def portfel_A(res: dict) -> dict:
    """A из рядов пар по progon_urovni (равный вес, без таргета). Плата за шорт: пары одного числа тикеров/дату -> среднее баз."""
    cnt = {s: np.isfinite(r["tik"]).sum(0) for s, r in res.items()}
    if any(not np.array_equal(cnt[s], cnt[SIDS[0]]) for s in cnt):
        raise ValueError("число пар по датам различается между Sk: A не равен среднему рядов Sk")
    dates = res[SIDS[0]]["dates"]
    x = urovni.a_shtrih({s: r["tik"] for s, r in res.items()}, urovni.setka()["umolchaniya"], dates)
    cat = lambda k: np.concatenate([res[s][k] for s in SIDS])                  # noqa: E731
    return dict(x=x, tik=None, baza=np.mean([res[s]["baza"] for s in SIDS], axis=0), dates=dates,
                expo=cat("expo"), cnt=cat("cnt"), god=cat("god"))


def metriki_goda(x: np.ndarray) -> dict:
    """Одно определение метрик (Д7, metriki_e): mSR, Шарп (ddof=1, √252), доходность = сумма дневного excess, MDD."""
    x = np.asarray(x, float)
    return dict(msr=me.msr(x), sharpe=me.sr_dnevnoy(x) * np.sqrt(252), dohodnost=float(x.sum()), mdd=me.mdd(x))


def pasport(imya: str, variant: str, r: dict, gody: tuple | None = None) -> list:
    """Строки паспорта по годам периода (умолчание — train4) и «все» — с платой 5 % за шорт и без; доля времени в позиции и сделки в год."""
    from fastbt.etap1_eval import years_of
    GODY = years_of(PER) if gody is None else tuple(gody)          # noqa: N806  годы — из текущего PER (при вызове)
    yr, rows = r["dates"].astype("datetime64[Y]").astype(int) + 1970, []
    for plata in (0.0, PLATA):
        x = r["x"] - plata * r["baza"]
        for g in (*GODY, "все"):
            m, mr = (yr == g, r["god"] == g) if g != "все" else (np.ones(len(yr), bool), np.ones(len(r["god"]), bool))
            n_god = 1 if g == "все" else 0
            rows.append(dict(ryad=imya, variant=variant, plata_zaem=plata, god=g, **metriki_goda(x[m]),
                             dolya_vremeni=float(r["expo"][mr].mean()), sdelok=float(r["cnt"][mr].sum()) / 2 / (len(GODY) if n_god else 1)))
    return rows


def zapis_ryada(imya: str, r: dict, out: Path = OUT) -> Path:
    """docs/tables/a_strogaya/ryad_<imya>.parquet: date, excess, excess_zaem5 (дневной excess train4, плата 5 % — zaem_v2)."""
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": pd.to_datetime(r["dates"]), "excess": r["x"], "excess_zaem5": r["x"] - PLATA * r["baza"]})
    p = out / f"ryad_{imya}.parquet"
    df.to_parquet(p, index=False)
    return p
