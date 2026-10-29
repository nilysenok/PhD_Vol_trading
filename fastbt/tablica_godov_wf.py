"""Главная таблица по годам (предрегистрация v2.0, §5–§7; Ж-136 п. 4). Формат `v: "1"`.

Контракт входа (его же выдаёт `fastbt/svod_wf.py`): `ryady: dict[str, pd.Series]` — имя ансамбля →
дневная избыточная доходность (DatetimeIndex по торговым дням, 2017-01…2026-02, без NaN;
день без позиции = 0). Обязательные имена — `ANSAMBLI`; лишние допустимы, недостающие — KeyError.
Даты вне 2017-01-01…2026-02-28 — ошибка (защита границы).

Метрики (Д7, одно определение каждой; код — `metriki_e`): Sharpe дневной (ddof=1, ·√252), mSR недельный
(суммы W-FRI, Фавр–Галеано, √52; вне области — NaN), Сортино дневное (g=252). По календарным годам;
«янв–фев 2026» — неполный, только в «итог»; «медиана 9 лет» — 2017…2025 без 2026 (NaN пропускаются).
Δ в таблице сравнений — разность табличных Шарпов (`sravneniya_wf`, мутация № 13).
Правило чтения (§7): итоговый Sharpe > 0 и медиана годовых Sharpe по 9 полным годам > 0.
Предзащита: `docs/tables/tabl_4_a_utechka.csv` (разложение #570: та же процедура с утечкой окна), как есть.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import sravneniya_wf as SW
from fastbt.metriki_e import metriki_ned, sortino_e, sr_dnevnoy

V = "1"
STATUS = "разработка, до теста; предрегистрация v2.0 (Ж-138)"
ANSAMBLI = ("A'-базовая", "A'-расширенная", "A_strogaya", "C", "D", "МСП λ=0", "МСП λ=1")
GODY = tuple(range(2017, 2026))
START, KONEC = pd.Timestamp("2017-01-01"), pd.Timestamp("2026-02-28")
METRIKI = ("sharpe", "msr_ned", "sortino")
STROKI = tuple(str(g) for g in GODY) + ("янв–фев 2026", "итог", "медиана 9 лет")
KOREN = Path(__file__).resolve().parents[1]
PREREG = KOREN / "docs" / "preregistraciya_v2.md"
PREDZASHCHITA = KOREN / "docs" / "tables" / "tabl_4_a_utechka.csv"


def metriki(x: pd.Series) -> dict:
    """Три метрики на куске ряда; меньше 3 наблюдений или нулевая дисперсия — NaN."""
    a = x.to_numpy(float)
    if len(a) < 3 or not np.std(a, ddof=1) > 1e-12:
        return dict.fromkeys(METRIKI, np.nan)
    return dict(sharpe=sr_dnevnoy(a) * np.sqrt(252), msr_ned=metriki_ned(a, x.index.values)["msr_ned"],
                sortino=sortino_e(a, 252))


def proverit(ryady: dict) -> None:
    for n in ANSAMBLI:
        if n not in ryady:
            raise KeyError(f"нет ряда {n!r}")
        s = ryady[n]
        if not isinstance(s.index, pd.DatetimeIndex):
            raise TypeError(f"{n}: индекс не DatetimeIndex")
        if len(s) and (s.index.min() < START or s.index.max() > KONEC):
            raise ValueError(f"{n}: даты вне {START.date()}…{KONEC.date()}")
        if s.isna().any():
            raise ValueError(f"{n}: NaN в ряду")


def tablica(ryady: dict) -> pd.DataFrame:
    """Строки STROKI × столбцы MultiIndex (ансамбль, метрика)."""
    proverit(ryady)
    t = pd.DataFrame(np.nan, index=list(STROKI), columns=pd.MultiIndex.from_product([ANSAMBLI, METRIKI]))
    god9 = [str(g) for g in GODY]
    for n in ANSAMBLI:
        s = ryady[n].sort_index()
        kuski = {str(g): s[s.index.year == g] for g in GODY}
        kuski["янв–фев 2026"], kuski["итог"] = s[s.index.year == 2026], s
        for k, x in kuski.items():
            for m, v in metriki(x).items():
                t.loc[k, (n, m)] = v
        for m in METRIKI:
            v9 = t.loc[god9, (n, m)].to_numpy(float)
            t.loc["медиана 9 лет", (n, m)] = np.nanmedian(v9) if np.isfinite(v9).any() else np.nan
    return t


def pravilo(t: pd.DataFrame) -> dict:
    """§7: по ансамблю — итоговый Sharpe > 0 и медиана годовых Sharpe по 9 годам > 0 (NaN — «нет»)."""
    return {n: "выполнено" if t.loc["итог", (n, "sharpe")] > 0 and t.loc["медиана 9 лет", (n, "sharpe")] > 0
            else "нет" for n in ANSAMBLI}


def predzashchita(put: Path | None = None) -> pd.DataFrame:
    """Строки разложения #570 как есть; нет файла — пустая таблица (пометка в шапке)."""
    put = Path(put or PREDZASHCHITA)
    return pd.read_csv(put) if put.exists() else pd.DataFrame()


def sha_predreg(put: Path = PREREG) -> str:
    """git-blob SHA-1 файла предрегистрации (= `git hash-object`); коммит — в журнале, Ж-138."""
    b = Path(put).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def hash_vhodov(ryady: dict) -> str:
    h = hashlib.sha256()
    for n in sorted(ryady):
        s = ryady[n].sort_index()
        h.update(n.encode() + s.index.values.astype("datetime64[D]").tobytes() + s.to_numpy(float).tobytes())
    return h.hexdigest()[:16]


def kod() -> str:
    try:
        return subprocess.run(["git", "-C", str(KOREN), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "n/a"


def shapka(ryady: dict, net_predzashchity: bool = False) -> list[str]:
    s = [f"v: {V}", f"статус: {STATUS}", f"код: {kod()}", f"хэш входов: {hash_vhodov(ryady)}",
         f"SHA предрегистрации (blob): {sha_predreg()}"]
    return s + (["предзащита: файл разложения #570 не найден"] if net_predzashchity else [])


def zapisat(ryady: dict, papka: Path, pary=SW.PARY_V2) -> dict:
    """Пишет tablica_godov.csv (+ строка «правило чтения»), sravneniya.csv, predzashchita.csv; шапка — `# `-строки."""
    t, p, papka = tablica(ryady), predzashchita(), Path(papka)
    papka.mkdir(parents=True, exist_ok=True)
    sr, pr = SW.sravnit(ryady, pary), pravilo(t)
    sh = shapka(ryady, p.empty)
    plain = t.copy()
    plain.columns = [f"{a}|{m}" for a, m in t.columns]
    plain.loc["правило чтения"] = [pr[c.split("|")[0]] if c.endswith("|sharpe") else "" for c in plain.columns]
    cmp_ = pd.DataFrame(sr["rows"]).assign(m=sr["m"])
    for nm, df, extra, idx in (("tablica_godov", plain, [], True), ("sravneniya", cmp_, [f"m: {sr['m']}"], False),
                               ("predzashchita", p, ["строка: предзащита (утечка окна)"], False)):
        with open(papka / f"{nm}.csv", "w", encoding="utf-8") as f:
            f.write("".join(f"# {x}\n" for x in sh + extra))
            df.to_csv(f, index=idx, index_label="строка" if idx else None)
    return dict(tablica=t, sravneniya=sr, pravilo=pr, predzashchita=p)
