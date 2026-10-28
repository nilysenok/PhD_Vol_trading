"""Матрица корреляций дневных избыточных рядов S1–S6, A_strogaya, S7–S9, train4 (2016–2019) — Ж-077.
Запуск: python3 scripts/analysis/korr_s1_s9.py [--out путь.csv]. Без аргументов — самопроверка и сверка с закоммиченным CSV.
Ряды: docs/tables/a_strogaya/ryad_*.parquet (колонка excess), docs/tables/golaya_s7_s9/krivaya.csv. Данные ≥ 2020 не читаются."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "docs" / "tables" / "golaya_s7_s9" / "korr_s1_s9.csv"
KRIVAYA = ROOT / "docs" / "tables" / "golaya_s7_s9" / "krivaya.csv"
RYAD = ROOT / "docs" / "tables" / "a_strogaya" / "ryad_{}.parquet"
NAMES = ["S1", "S2", "S3", "S4", "S5", "S6", "A_strogaya", "S7", "S8", "S9"]
START, END = "2016-01-01", "2019-12-31"


def korr_matrix(ryady: pd.DataFrame) -> pd.DataFrame:
    """Корреляция Пирсона по столбцам; пропуск в любом ряду — ошибка, не молчаливая обрезка."""
    if ryady.isna().any().any():
        raise ValueError("пропуски в рядах")
    return ryady.corr()


def _excess(name: str) -> pd.Series:
    f = "A" if name == "A_strogaya" else name
    d = pd.read_parquet(str(RYAD).format(f))
    s = d["excess"] if "date" not in d.columns else d.set_index("date")["excess"]
    s.index = pd.to_datetime(s.index)
    return s.rename(name)


def load() -> pd.DataFrame:
    cols = [_excess(n) for n in NAMES[:7]]
    k = pd.read_csv(KRIVAYA, index_col="date", parse_dates=True)
    cols += [k[n] for n in NAMES[7:]]
    df = pd.concat(cols, axis=1).loc[START:END]
    return df.dropna()


def to_csv_text(m: pd.DataFrame) -> str:
    return m.to_csv(float_format="%.4f", lineterminator="\n")


def _selftest() -> None:
    import numpy as np
    x = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)[::-1] + np.sin(np.arange(10))})
    m = korr_matrix(x)
    assert abs(m.loc["a", "a"] - 1) < 1e-12 and abs(m.loc["a", "b"] - m.loc["b", "a"]) < 1e-12
    print("самопроверка: ок")


def main() -> int:
    _selftest()
    df = load()
    print(f"дней: {len(df)}, {df.index.min().date()} … {df.index.max().date()}")
    text = to_csv_text(korr_matrix(df))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(text)
        return 0
    same = text == CSV.read_text()
    print("побайтно совпало" if same else "НЕ совпало с закоммиченным CSV")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
