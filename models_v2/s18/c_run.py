"""S-C-DATA: запуск. `python3 -m models_v2.s18.c_run --write` пишет `data/regen/harj_struct/C/`; `--check` только читает:
пересчёт в памяти == файлы, хэш каталога выходов до и после совпадает. Прогнозов нет, реестр и CHECKPOINT не трогаются."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

from models_v2.s17.fwd_src import ROOT
from models_v2.s18 import c_data, c_decomp, c_sync

OUT = ROOT / "data/regen/harj_struct/C"
CSV = {"coverage_paper_year.csv": c_sync.paper_coverage, "schedule_year.csv": c_sync.schedule, "short_days.csv": c_sync.short_days}


def dir_hash(path: Path = OUT) -> str:
    """Хэш имён и содержимого всех файлов каталога выходов."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")) if path.exists() else []:
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def compute(inp=None) -> dict:
    """Все выходы листа в памяти: имя файла -> DataFrame."""
    inp = inp or c_data.build_inputs()
    df = c_decomp.run(inp)
    out = {"C_sessions.parquet": df, "decomp_year.csv": c_sync.decomposition_by_year(df)}
    out.update({name: fn(inp) for name, fn in CSV.items()})
    return out


def write(out: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        tmp = OUT / ("tmp_" + name)
        df.to_parquet(tmp) if name.endswith(".parquet") else df.to_csv(tmp, index=False)
        tmp.replace(OUT / name)


def read(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / name) if name.endswith(".parquet") else pd.read_csv(OUT / name)


def check(out: dict) -> None:
    """Файлы == пересчёт (CSV — по округлённым значениям записи); только чтение."""
    for name, df in out.items():
        on_disk = read(name)
        if name.endswith(".parquet"):
            pd.testing.assert_frame_equal(on_disk, df)
        else:
            pd.testing.assert_frame_equal(on_disk, pd.read_csv(pd.io.common.StringIO(df.to_csv(index=False))))


def main(argv: list) -> int:
    out = compute()
    df = out["C_sessions.parquet"]
    print(f"сессий {df['d'].nunique()} {df['d'].min().date()}..{df['d'].max().date()}; max|ident|={df['ident_err'].abs().max():.3e}")
    if "--write" in argv:
        write(out)
    if "--check" in argv:
        before = dir_hash()
        check(out)
        assert dir_hash() == before, "выходы изменились при --check"
        print("--check: файлы == пересчёт, хэш выходов не изменился: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
