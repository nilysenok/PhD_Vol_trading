"""Граница сужения test_locked считается по календарю торгов MOEX, а не по датам файла (синтетика).

Мутация: расчёт по датам файла — файл 2018 года получает класс «в», файл до 2022-12-30 теряет декабрь 2022.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dannye"))
import suzhenie_obshee as ob  # noqa: E402
import suzhenie_test_locked as st  # noqa: E402

CAL = list(ob.kalendar())
HI = date(2024, 12, 31)


def dates_to(lo, hi):
    return [d for d in CAL if lo <= d <= hi]


def klass(dates, h):
    m = dict(date_col="date", nrows=len(dates), dmin=dates[0], dmax=dates[-1])
    return ob.classify(f"data/x_h{h}_test.parquet", m, dates)


def pozzhe_2025():
    return [d.date() for d in pd.bdate_range("2025-01-03", "2025-02-28")]


def old_granitsa(dates, h):
    """Прежний расчёт по датам файла (мутация)."""
    ok = [d for d in dates if d <= ob.T1_HI]
    i = len(ok) - 1 - h
    return ok[i] if i >= 0 else None


def test_kalendar_pokryvaet_do_2024():
    assert CAL == sorted(set(CAL)) and CAL[0] <= date(2018, 1, 9) and CAL[-1] <= HI
    assert date(2024, 12, 30) in CAL and date(2025, 1, 3) not in CAL


def test_fail_2018_h5_klass_a_bez_rezki():
    dates = dates_to(date(2018, 1, 1), date(2018, 12, 29))
    k, cut, h = klass(dates, 5)
    assert (k, h) == ("а", 5) and cut == CAL[-1 - 5]


def test_fail_do_2022_12_30_h22_klass_a():
    dates = dates_to(date(2022, 1, 1), date(2022, 12, 30))
    k, cut, _ = klass(dates, 22)
    assert k == "а" and cut > dates[-1]


def test_fail_2022_2025_h22_klass_v_i_okno_ok():
    dates = dates_to(date(2022, 1, 1), HI) + pozzhe_2025()
    k, cut, _ = klass(dates, 22)
    assert k == "в" and cut == CAL[-1 - 22]
    assert ob.okno_ok(dates, cut, 22)
    sled = CAL[CAL.index(cut) + 1]
    assert not ob.okno_ok(dates, sled, 22)  # на день позже цель уже за 31.12.2024


@pytest.mark.parametrize("h", [1, 5, 22])
def test_cut_ne_zavisit_ot_fayla(h):
    d1 = dates_to(date(2018, 1, 1), date(2018, 12, 29))
    d2 = dates_to(date(2022, 1, 1), HI) + pozzhe_2025()
    assert ob.granitsa(d1, h) == ob.granitsa(d2, h) == CAL[-1 - h]


def test_h0_cut_posledniy_torgovy_den():
    assert ob.granitsa([], 0) == date(2024, 12, 30) == max(d for d in CAL if d <= HI)
    syn = [date(2024, 12, 27), date(2024, 12, 30), date(2025, 1, 3)]
    assert ob.granitsa([], 0, cal=syn) == date(2024, 12, 30)
    assert ob.granitsa([], 1, cal=syn) == date(2024, 12, 27)


def test_mutatsiya_po_datam_fayla_lovitsya(monkeypatch):
    d18 = dates_to(date(2018, 1, 1), date(2018, 12, 29))
    d22 = dates_to(date(2022, 1, 1), date(2022, 12, 30))
    assert klass(d18, 5)[0] == "а" and klass(d22, 22)[0] == "а"
    monkeypatch.setattr(ob, "granitsa", old_granitsa)
    assert klass(d18, 5)[0] == "в"  # тесты (1)/(2) падают на этой мутации
    assert klass(d22, 22)[0] == "в" and klass(d22, 22)[1] == date(2022, 11, 30)


def test_konveyer_fail_2018_ne_rezhetsya(tmp_path, capsys):
    p = tmp_path / "data/final_h5_test.parquet"
    p.parent.mkdir(parents=True)
    idx = pd.to_datetime(dates_to(date(2018, 1, 1), date(2018, 12, 29)))
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"date": idx, "val": 1.0}), preserve_index=False), p)
    rc = st.main(["--rezat", "--primenit", "--koren", str(tmp_path), "--vkhod", "data",
                  "--z2-dir", "z2", "--z3-dir", "z3", "--ishodnye-dir", "ish"])
    out = capsys.readouterr().out
    assert rc == 0 and "klass=а" in out
    assert not (tmp_path / "z3").exists() or not any((tmp_path / "z3").rglob("*.parquet"))


# ---- Ж-118/Ж-119: проверки частей в пробном прогоне, ITOG, n_h25_2024 ----
import suzhenie_rezka as rz  # noqa: E402

DIRS = ["--z2-dir", "z2", "--z3-dir", "z3", "--ishodnye-dir", "ish"]


def mk(tmp, name, dates):
    p = tmp / "data" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.to_datetime(dates)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"date": idx, "val": 1.0}), preserve_index=False), p)


def run(tmp, capsys, *mode, extra=()):
    rc = st.main([*mode, "--koren", str(tmp), "--vkhod", "data", *DIRS, *extra])
    return rc, capsys.readouterr().out


def sred():
    return dates_to(date(2024, 6, 1), HI) + pozzhe_2025()


def test_n_h25_2024_v_vyvode_i_manifeste(tmp_path, capsys):
    mk(tmp_path, "f_h5_test.parquet", sred())
    rc, out = run(tmp_path, capsys, "--rezat")
    assert rc == 0 and "n_h25_2024=5" in out and "min_h25=" in out
    assert "ITOG fajlov=1 razrezano=1 celikom_otkryto=0 zakryto=0 bez_daty=0 fail=0" in out
    run(tmp_path, capsys, "--manifest-do", extra=["--primenit", "--out", "do.tsv"])
    rc, out = run(tmp_path, capsys, "--rezat", extra=["--primenit"])
    assert rc == 0 and "n_h25_2024=5" in out
    run(tmp_path, capsys, "--manifest-posle", extra=["--primenit", "--do-tsv", str(tmp_path / "do.tsv"),
                                                     "--out", "posle.tsv"])
    rows = ob.chitai_tsv(tmp_path / "posle.tsv")
    assert [r["n_h25_2024"] for r in rows if r["path"].startswith("z3/")] == ["5"]


def test_itog_primer(tmp_path, capsys):
    mk(tmp_path, "a_h5_test.parquet", dates_to(date(2018, 1, 1), date(2018, 12, 29)))
    mk(tmp_path, "b_h22_test.parquet", sred())
    mk(tmp_path, "c_test.parquet", pozzhe_2025())
    rc, out = run(tmp_path, capsys, "--rezat")
    assert rc == 0 and "ITOG fajlov=3 razrezano=1 celikom_otkryto=1 zakryto=1 bez_daty=0 fail=0" in out


def lomaem(monkeypatch, tmp_path, capsys, name, mut):
    mk(tmp_path, name, sred())
    real = rz.razrezat
    monkeypatch.setattr(rz, "proveri", lambda *a: None)  # суммы и старые проверки не мешают новым
    monkeypatch.setattr(rz, "razrezat", lambda s, c, k: mut(*real(s, c, k)))
    return run(tmp_path, capsys, "--rezat")


def test_fail_peresechenie(monkeypatch, tmp_path, capsys):
    rc, out = lomaem(monkeypatch, tmp_path, capsys, "f_h5_test.parquet",
                     lambda pre, t1, h25: (pre, t1, pa.concat_tables([h25, t1.slice(0, 1)])))
    assert rc == 1 and "status=FAIL" in out and "пересекаются" in out and "fail=1" in out


def test_fail_otkrytaya_cel_za_2024(monkeypatch, tmp_path, capsys):
    rc, out = lomaem(monkeypatch, tmp_path, capsys, "f_h5_test.parquet",
                     lambda pre, t1, h25: (pre, pa.concat_tables([t1, h25.slice(0, 1)]), h25.slice(1)))
    assert rc == 1 and "status=FAIL" in out and "цель за 31.12.2024" in out


def test_fail_zakrytaya_min_h_bolshe_0(monkeypatch, tmp_path, capsys):
    rc, out = lomaem(monkeypatch, tmp_path, capsys, "f_h5_test.parquet",
                     lambda pre, t1, h25: (pre, t1, h25.slice(1)))
    assert rc == 1 and "status=FAIL" in out and "следующий торговый день" in out


def test_fail_zakrytaya_min_h0_ranshe_2025(monkeypatch, tmp_path, capsys):
    def mut(pre, t1, h25):
        return pre, t1.slice(0, len(t1) - 1), pa.concat_tables([t1.slice(len(t1) - 1), h25])
    rc, out = lomaem(monkeypatch, tmp_path, capsys, "f_test.parquet", mut)
    assert rc == 1 and "status=FAIL" in out and "раньше 01.01.2025" in out


def test_fail_ne_ostanavlivaet_ostalnye(monkeypatch, tmp_path, capsys):
    mk(tmp_path, "z_h5_test.parquet", sred())
    rc, out = lomaem(monkeypatch, tmp_path, capsys, "a_h5_test.parquet",
                     lambda pre, t1, h25: (pre, t1, h25.slice(1)))
    assert rc == 1 and out.count("status=FAIL") == 2 and "razrezano=2" in out and "fail=2" in out
