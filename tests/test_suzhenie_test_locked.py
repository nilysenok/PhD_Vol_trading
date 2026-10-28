"""Тесты сужения test_locked: только синтетика во временном каталоге (Ж-112 п. 1, Ш6).

Мутации (образцы mutations.md № 15, 17, 30, 36): строка 2025 в __t1; окно h за 31.12.2024;
значения напечатаны; исходник удалён вместо переноса; сумма строк не сходится.
"""
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dannye"))
import suzhenie_obshee as ob  # noqa: E402
import suzhenie_rezka as rz  # noqa: E402
import suzhenie_test_locked as st  # noqa: E402

SENT = "777000"  # префикс значений синтетики: не должен появляться в выводе
Z3, ISH, Z2 = "z3", "ish", "z2"
DIRS = ["--z2-dir", Z2, "--z3-dir", Z3, "--ishodnye-dir", ISH]


def mk(path, d0="2021-12-01", d1="2025-02-28", col="date"):
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range(d0, d1)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({col: idx, "val": 777000.0 + pd.RangeIndex(len(idx)).to_numpy()}),
                                        preserve_index=False), path, row_group_size=100)


def run(tmp, capsys, *mode, extra=()):
    rc = st.main([*mode, "--koren", str(tmp), "--vkhod", "data", *DIRS, *extra])
    return rc, capsys.readouterr().out


def utechka(out):
    return SENT in out


@pytest.fixture
def tree(tmp_path):
    mk(tmp_path / "data/mix.parquet")
    mk(tmp_path / "data/final_h22_test.parquet")
    return tmp_path


def read(p):
    return pq.read_table(p).to_pandas()


def test_plan_klassy(tmp_path, capsys):
    mk(tmp_path / "data/a.parquet", "2022-03-01", "2023-05-01")
    mk(tmp_path / "data/b.parquet", "2025-02-03", "2025-02-28")
    mk(tmp_path / "data/v.parquet")
    mk(tmp_path / "data/g.parquet", col="nazvanie")
    (tmp_path / "data/OPENED_x.json").write_text("{}")
    _, out = run(tmp_path, capsys, "--plan")
    k = {ln.split("\t")[0].split("=")[1]: ln.split("\t")[1].split("=")[1] for ln in out.splitlines()}
    assert k == {"data/a.parquet": "а", "data/b.parquet": "б", "data/v.parquet": "в",
                 "data/g.parquet": "г", "data/OPENED_x.json": "д"}


def test_manifest_do_granitsy_bez_znacheniy(tree, capsys):
    rc, out = run(tree, capsys, "--manifest-do")
    assert rc == 0 and not utechka(out)
    assert f"dmin={date(2021, 12, 1)}\tdmax={date(2025, 2, 28)}" in out


def test_rezka_granitsy_summa_ishodnik(tree, capsys):
    src = tree / "data/mix.parquet"
    orig, sha = read(src), ob.sha_file(src)
    rc, out = run(tree, capsys, "--rezat", extra=["--primenit"])
    t1, h25 = read(tree / Z2 / "data/mix__t1.parquet"), read(tree / Z3 / "data/mix__h25.parquet")
    old = read(src)
    assert t1.date.min() >= pd.Timestamp("2022-01-01") and t1.date.max() <= pd.Timestamp("2024-12-31")
    assert h25.date.min() >= pd.Timestamp("2025-01-01")
    assert len(old) + len(h25) == len(orig) and old.date.max() <= pd.Timestamp("2024-12-31")
    assert ob.sha_file(tree / ISH / "data/mix.parquet") == sha and not utechka(out)


def test_okno_h22_ne_zahodit_za_2024(tree, capsys):
    run(tree, capsys, "--rezat", extra=["--primenit"])
    t1 = read(tree / Z2 / "data/final_h22__t1.parquet")
    assert pd.Timestamp("2024-12-20") not in set(t1.date)
    dates = list(ob.to_dates(read(tree / ISH / "data/final_h22_test.parquet").date))
    assert ob.okno_ok(dates, max(ob.to_dates(t1.date)), 22)


def test_dry_run_ne_pishet(tree, capsys):
    before = sorted(p.name for p in tree.rglob("*"))
    run(tree, capsys, "--rezat")
    run(tree, capsys, "--manifest-do")
    assert before == sorted(p.name for p in tree.rglob("*"))


def test_mut_stroka_2025_v_t1(tree, capsys, monkeypatch):
    real = rz.razrezat

    def bad(src, col, cut):
        pre, t1, h25 = real(src, col, cut)
        return pre, pa.concat_tables([t1, h25.slice(0, 1)]), h25.slice(1)

    monkeypatch.setattr(rz, "razrezat", bad)
    with pytest.raises(ValueError, match="Z2"):
        run(tree, capsys, "--rezat", extra=["--primenit"])
    assert (tree / "data/mix.parquet").exists() and not (tree / ISH).exists()


def test_mut_okno_h_za_2024(tree, capsys, monkeypatch):
    monkeypatch.setattr(ob, "granitsa", lambda dates, h: [d for d in dates if d <= ob.T1_HI][-1])
    with pytest.raises(ValueError, match="Z2"):
        run(tree, capsys, "--rezat", extra=["--primenit"])
    assert not (tree / Z2).exists()


def test_mut_znacheniya_napechatany(tree, capsys, monkeypatch):
    with pytest.raises(ValueError):
        ob.vyvod(path="x", val=SENT)
    with pytest.raises(TypeError):
        ob.vyvod(path=[SENT])
    monkeypatch.setattr(ob, "vyvod", lambda **r: print(r, read(tree / "data/mix.parquet").val[0]))
    assert utechka(run(tree, capsys, "--plan")[1])  # проверка утечки ловит мутацию


def test_mut_ishodnik_udalyon(tree, capsys, monkeypatch):
    monkeypatch.setattr(rz, "perenos", lambda src, dst: os.remove(src))
    with pytest.raises(RuntimeError, match="_ishodnye"):
        run(tree, capsys, "--rezat", extra=["--primenit"])


def test_mut_summa_ne_shoditsya(tree, capsys, monkeypatch):
    real = rz.razrezat

    def bad(src, col, cut):
        pre, t1, h25 = real(src, col, cut)
        return pre, t1, h25.slice(1)

    monkeypatch.setattr(rz, "razrezat", bad)
    with pytest.raises(ValueError, match="сумма"):
        run(tree, capsys, "--rezat", extra=["--primenit"])
    assert (tree / "data/mix.parquet").exists()


def test_posle_sohranenie_i_hesh_holdout(tree, capsys):
    do = "docs/do.tsv"
    run(tree, capsys, "--manifest-do", extra=["--primenit", "--out", do])
    run(tree, capsys, "--rezat", extra=["--primenit"])
    rc, out = run(tree, capsys, "--manifest-posle", extra=["--do-tsv", str(tree / do)])
    assert rc == 0 and "status=OK" in out and "holdout_sha=" in out and not utechka(out)
    os.remove(tree / ISH / "data/mix.parquet")  # откат невозможен: исходник потерян
    rc2, out2 = run(tree, capsys, "--manifest-posle", extra=["--do-tsv", str(tree / do)])
    assert rc2 == 1 and "status=FAIL" in out2
