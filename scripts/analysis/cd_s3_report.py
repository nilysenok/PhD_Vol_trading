"""Отчёт cd_s3_sigma: все числа берутся из посчитанных таблиц (не переписываются руками)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _tab(df: pd.DataFrame, cols: list) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], (float, np.floating)) and pd.notna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def napisat(path: Path, df: pd.DataFrame, sd: pd.DataFrame, msr_k2: np.ndarray, p_k2: float, nan: dict, sek: float) -> None:
    m = lambda key: float(df[(df.god == "2018-2019") & df.stroka.str.contains(key, regex=False)].msr.iloc[0])  # noqa: E731
    base, hyb, harj, nai = m("(база)"), m("(H2)"), m("HAR-J"), m("наивной")
    first = (f"**Δ mSR σ̂ против базы: {hyb - base:+.3f}; против HAR-J (б): {hyb - harj:+.3f}; против наивной √RV: {hyb - nai:+.3f}** "
             f"(S3, 2018–2019, плата за шорт 5 %, Δ = разность табличных mSR; p к2 по 100 сдвигам = {p_k2:.2f}).")
    parts = []
    for col, nm in (("corr", "уровни"), ("corr_prirasch", "приращения")):
        t = sd.pivot(index="ryad", columns="sdvig", values=col).reset_index()
        t.columns = [str(c) for c in t.columns]
        mx = {r: int(g.loc[g[col].idxmax(), "sdvig"]) for r, g in sd.groupby("ryad")}
        parts.append(f"{nm}; максимум по k: {mx}\n{_tab(t, list(t.columns))}")
    sd_txt = "\n\n".join(parts)
    txt = f"""{first}

**Статус: разработка train4, окно 2018–2019, потолок, H2-строка.** Не результат теста. Выбор порога p67 сделан по диагностике прохода 1 на тех же данных
train4 (`docs/otchety/2026-10-28_prohod1_diagnostika.md`) — честной половины нет, цифры выше — потолок. Данные ≥ 2020 не читались.

## Проверка сдвига σ̂ (Д1)
Корреляция σ̂ (строка d, известна на close d) с √RV дня d+1+k, пул по 17 тикерам, 2018–2019. k = −1 — день d, последний известный: максимум обязан быть там
(оконные ряды — по приращениям); k = 0 — день прогноза d+1, k = +1 — d+2.
{sd_txt}

## Что считалось
- База: A_strogaya «только S3» (`fastbt/a_strogaya.py`, точка A, вход open 10:00 d+1), совпала с `docs/tables/a_strogaya/ryad_S3.parquet` (≤ 1e-9).
- Затвор: решение на close d не исполняется (сделка пропущена целиком), если ранг σ тикера ≥ p67 в окне 252 значений, оканчивающемся днём d
  (доля строго ниже; нужно ≥ 126 значений). Открытая позиция не закрывается. Ранг не определён → затвор открыт: дней 2018–2019 с неопределённым рангом (σ̂ / HAR-J / √RV): {nan}.
- Реализованная σ — `rv22` из входов Sk (справочно, не прогноз). σ̂ — гибрид h1, `data/regen/ch4_dev/sigma_hyb_h1_dev.parquet`; HAR-J (б) — `sigma_harj_b_h1_dev`; наивная — `sigma_naive_dev`.
- к2: круговой сдвиг маски затвора σ̂ по тикеру на lag ∈ [21, n−21], 100 повторов, seed 20261028; p = доля повторов с mSR ≥ mSR σ̂. mSR к2: среднее {msr_k2.mean():.3f}, 5 % {np.quantile(msr_k2, .05):.3f}, 95 % {np.quantile(msr_k2, .95):.3f}.
- Метрики: mSR (`fastbt/metriki_e.py`, одно определение Д7) на дневном excess с платой за шорт 5 % (`zaem_v2`), комиссия 0,04 %; сделок = стороны/2.

## Результат (`docs/tables/cd_s3/sigma_zatvor.csv`)
{_tab(df, ['stroka', 'tip', 'god', 'msr', 'sdelok', 'd_vs_baza'])}

## Оговорки
- ↺ Окно 2018–2019, а не 2016–2019: ряды σ̂ начинаются 03.01.2017, рангу нужен прогрев (план C/E, отступление 1).
- Два года и 17 тикеров, одна пара S3: не статистика; Δ без доверительного интервала. Метрика по году на ~250 днях шумна.
- Затвор только по входу; перемешивание — маски, а не ряда σ̂ (сохраняет долю закрытых дней и автокорреляцию).
- Число сделок падает вместе с затвором: Δ mSR смешивает качество отбора и сокращение экспозиции; к2 это и контролирует.
- Не проверено: другие p (только p67 из задания), другие горизонты σ̂, остальные Sk.
- Время счёта: {sek:.0f} с.
"""
    path.write_text(txt, encoding="utf-8")
