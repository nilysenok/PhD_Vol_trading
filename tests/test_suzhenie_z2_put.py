"""Путь части Z2 не должен попадать под замок чтения (zamki.json); синтетика во временном каталоге.

Мутация: старая chasti_puti (Z2/p.parent/...) оставляет сегмент test_locked -> тест (1) падает.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dannye"))
import suzhenie_rezka as rz  # noqa: E402
import suzhenie_test_locked as st  # noqa: E402

PRAVILA = [re.compile(x) for x in json.loads((ROOT / ".claude/hooks/zamki.json").read_text())["zakryto_chtenie"]]
Z2, Z3, ISH = "data/regen/test1_2022_2024", "data/regen/hold25", "data/regen/ish"
SRC = ["data/regen/test_locked/final3/final_h1_test.parquet",
       "data/models_cache/test_locked/boost/final_h1_test.parquet",
       "data/regen/test_locked/x/a_test.parquet"]


def test_z2_ne_pod_zamkom():
    for rel in SRC:
        p1, _ = rz.chasti_puti(rel, Z2, Z3)
        assert "test_locked" not in p1.parts
        assert not any(r.search(p1.as_posix()) for r in PRAVILA), p1


def test_dva_kornya_ne_sovpadayut():
    a = rz.chasti_puti("data/regen/test_locked/x/a_test.parquet", Z2, Z3)[0]
    b = rz.chasti_puti("data/models_cache/test_locked/x/a_test.parquet", Z2, Z3)[0]
    assert a != b
    assert a.as_posix() == f"{Z2}/regen/x/a__t1.parquet"
    assert b.as_posix() == f"{Z2}/models_cache/x/a__t1.parquet"


def test_skvoznoy_rezat_i_manifest(tmp_path, capsys):
    rel = "data/regen/test_locked/final3/final_h1_test.parquet"
    (tmp_path / rel).parent.mkdir(parents=True)
    idx = pd.bdate_range("2021-12-01", "2025-02-28")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"date": idx, "val": 1.0}), preserve_index=False),
                   tmp_path / rel)
    d = ["--z2-dir", Z2, "--z3-dir", Z3, "--ishodnye-dir", ISH, "--koren", str(tmp_path), "--vkhod", "data"]
    do = "docs/do.tsv"
    assert st.main(["--manifest-do", "--primenit", "--out", do, *d]) == 0
    assert st.main(["--rezat", "--primenit", *d]) == 0
    p1, _ = rz.chasti_puti(rel, Z2, Z3)
    assert (tmp_path / p1).exists()
    capsys.readouterr()
    rc = st.main(["--manifest-posle", "--do-tsv", str(tmp_path / do), *d])
    out = capsys.readouterr().out
    assert rc == 0 and "status=OK" in out
