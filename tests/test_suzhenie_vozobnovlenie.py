"""Сверка среза после резки не зависит от разбиения на чанки; возобновление прерванного прогона.

Причина (расследование): sha_tabl хэширует IPC-поток, а байты потока включают границы батчей;
срез из скана исходника и срез перечитанного файла делятся на чанки по-разному.
Мутации: изменённое значение, переставленные строки, сменённый тип -> сравнение падает.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dannye"))
import suzhenie_obshee as ob  # noqa: E402
import suzhenie_test_locked as st  # noqa: E402
import suzhenie_vozobn as rzv  # noqa: E402

Z2, Z3, ISH = "z2", "z3", "ish"
DIRS = ["--z2-dir", Z2, "--z3-dir", Z3, "--ishodnye-dir", ISH]
REL = "data/D17_h1/G1X_test.parquet"


def sintetika(n_tick=170, d0="2017-01-03", d1="2025-12-30"):
    days = pd.bdate_range(d0, d1)
    n = len(days) * n_tick
    rng = np.random.default_rng(1)
    base = rng.normal(size=n)
    base[::97] = np.nan
    df = pd.DataFrame({
        "date": np.repeat(days.values, n_tick),
        "ticker": np.tile([f"T{i}" for i in range(n_tick)], len(days)),
        "year": np.repeat(days.year.values, n_tick).astype("int64"),
        "step": 1, "model": np.where(rng.random(n) < .5, "har", "lstm"), "rank": np.tile(np.arange(n_tick), len(days)),
        "seed": 7, "rv_actual": rng.random(n), "rv_pred": rng.random(n), "base_ln": base})
    return pa.Table.from_pandas(df, preserve_index=False)


def zapisat(koren, rel=REL, t=None, rg=None):
    p = Path(koren) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t if t is not None else sintetika(), p, row_group_size=rg)
    return p


def zapusk(koren, capsys):
    rc = st.main(["--rezat", "--primenit", "--koren", str(koren), "--vkhod", "data", *DIRS])
    return rc, capsys.readouterr().out


def staryj_sha(t):
    """Прежний sha_tabl: сырые байты IPC-потока."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, t.schema) as w:
        w.write_table(t)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def test_prichina_staryj_sha_zavisit_ot_chankov_i_musora_pod_null():
    """H1 (чанки) и H3 (null в base_ln): при равных данных сырые IPC-байты различаются; новый sha_tabl — нет."""
    t = sintetika(n_tick=3, d1="2018-12-31")
    a = pa.Table.from_batches(t.to_batches(max_chunksize=500))
    b = pa.Table.from_batches(t.to_batches(max_chunksize=700))
    assert a.equals(b) and staryj_sha(a) != staryj_sha(b)  # H1: границы батчей попадают в байты
    assert ob.sha_tabl(a) == ob.sha_tabl(b)
    nul = np.zeros(len(t), dtype=bool)
    nul[::97] = True
    x = np.arange(len(t), dtype="float64")
    c1 = t.set_column(9, "base_ln", pa.array(np.where(nul, 1.0, x), mask=nul))
    c2 = t.set_column(9, "base_ln", pa.array(np.where(nul, 5.0, x), mask=nul))  # другой мусор под теми же null
    assert c1.to_pydict() == c2.to_pydict() and staryj_sha(c1) != staryj_sha(c2)  # H3: данные в пустых слотах
    assert ob.sha_tabl(c1) == ob.sha_tabl(c2)


@pytest.mark.parametrize("rg", [None, 1000, 20000, 131072, 200000])
def test_rezka_bolshogo_fajla_ne_padaet(tmp_path, capsys, rg):
    zapisat(tmp_path, rg=rg)
    rc, out = zapusk(tmp_path, capsys)
    assert rc == 0 and "fail=0" in out


def test_gipotezy_otbrosheny_tip_porjadok_nan_metadannye():
    """H2 (тип model), H3 (NaN), H4 (порядок), H5 (метаданные): схема и данные равны, расходится только sha по чанкам."""
    t = sintetika(n_tick=3, d1="2018-12-31")
    a = pa.Table.from_batches(t.to_batches(max_chunksize=500))
    b = pa.Table.from_batches(t.to_batches(max_chunksize=700))
    assert a.schema.equals(b.schema, check_metadata=True) and a.equals(b)
    assert str(a.schema.field("model").type) == "string" and a.column("base_ln").null_count + sum(
        np.isnan(a.column("base_ln").to_numpy())) > 0
    assert rzv.odinakovy(a, b) and ob.sha_tabl(a) == ob.sha_tabl(b)


def test_mutacii_sravneniya_lovyatsya():
    t = sintetika(n_tick=3, d1="2018-12-31")
    chanki = pa.Table.from_batches(t.to_batches(max_chunksize=500))
    assert rzv.odinakovy(t, chanki)
    rv = t.column("rv_pred").to_numpy().copy()
    rv[5] += 1e-9
    znach = t.set_column(t.schema.get_field_index("rv_pred"), "rv_pred", pa.array(rv))
    perestav = t.take(pa.array([1, 0, *range(2, len(t))]))
    tip = t.set_column(t.schema.get_field_index("model"), "model", t.column("model").cast(pa.large_string()))
    tip2 = t.set_column(t.schema.get_field_index("rv_pred"), "rv_pred", t.column("rv_pred").cast(pa.float32()))
    nan = t.set_column(t.schema.get_field_index("base_ln"), "base_ln", pa.array(t.column("base_ln").to_pylist()[:1] + [None] + t.column("base_ln").to_pylist()[2:], pa.float64()))
    for bad in (znach, perestav, tip, tip2, nan):
        assert not rzv.odinakovy(t, bad)


def mk_tree(koren, imena):
    for n in imena:
        zapisat(koren, f"data/{n}_test.parquet", sintetika(n_tick=4, d0="2020-01-01", d1="2025-03-31"), rg=700)


def snimok(koren):
    """mtime и sha всех parquet; файл C ещё не тронут в первом прогоне, поэтому исключён из сравнения «до/после»."""
    return {p.relative_to(koren).as_posix(): (p.stat().st_mtime_ns, ob.sha_file(p))
            for p in Path(koren).rglob("*.parquet") if "/C" not in p.as_posix()[len(str(koren)):]}


def zapusk_vkhod(koren, capsys, vkhod):
    rc = st.main(["--rezat", "--primenit", "--koren", str(koren), "--vkhod", *vkhod, *DIRS])
    return rc, capsys.readouterr().out


def test_vozobnovlenie_dovodit_do_konca_bez_perezapisi(tmp_path, capsys):
    mk_tree(tmp_path, ["A", "B", "C"])
    rc, _ = zapusk_vkhod(tmp_path, capsys, ["data/A_test.parquet", "data/B_test.parquet"])  # «прерванный» прогон
    assert rc == 0
    do = snimok(tmp_path)
    rc, out = zapusk_vkhod(tmp_path, capsys, ["data"])
    assert rc == 0 and "fail=0" in out
    assert out.count("status=uzhe_razrezan_provereno") == 2 and out.count("status=razrezan") == 1, out
    posle = snimok(tmp_path)
    assert all(posle[k] == v for k, v in do.items())  # ничего из прежнего не перезаписано
    rc, out = zapusk_vkhod(tmp_path, capsys, ["data"])  # и полностью завершённый прогон повторяется без ошибок
    assert rc == 0 and out.count("status=uzhe_razrezan_provereno") == 3 and snimok(tmp_path) == posle


@pytest.mark.parametrize("chto", ["t1", "h25", "src"])
def test_vozobnovlenie_lovit_porchu(tmp_path, capsys, chto):
    mk_tree(tmp_path, ["A"])
    assert zapusk_vkhod(tmp_path, capsys, ["data"])[0] == 0
    p = {"t1": tmp_path / Z2 / "data/A__t1.parquet", "h25": tmp_path / Z3 / "data/A__h25.parquet",
         "src": tmp_path / "data/A_test.parquet"}[chto]
    t = pq.read_table(p)
    pq.write_table(t.take(pa.array([1, 0, *range(2, len(t))])), p)  # переставлены две строки
    rc, out = zapusk_vkhod(tmp_path, capsys, ["data"])
    assert rc == 1 and "status=FAIL" in out
