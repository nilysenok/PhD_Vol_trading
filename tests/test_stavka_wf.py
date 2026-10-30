"""Ставка ЦБ wf до 2026-02-28: дописывание по решениям с датой вступления (Д3), без ffill вперёд (Д4).

Исходник теста — файл репозитория, обрезанный до 2026-02-06 (то, что было до дописывания). Мутации — в памяти."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "dannye"))
import stavka_wf_dopisat as W  # noqa: E402
from fastbt import msp, zona  # noqa: E402

REPO = ROOT / "moex_discovery" / "data" / "external" / "macro" / "raw" / "key_rate_cbr.parquet"
S = pd.Timestamp("2026-02-06")
D = pd.Timestamp


def staryj() -> pd.DataFrame:
    r = pd.read_parquet(REPO)
    return r[r["date"] <= S].reset_index(drop=True)


def znach(n: pd.DataFrame) -> dict:
    return dict(zip(n["date"].dt.strftime("%Y-%m-%d"), n["value"]))


def test_prefiks_do_0206_pobajtno():
    s, n = staryj(), W.dopisat(staryj())
    assert len(s) == 3101 and s["date"].max() == S
    assert n.iloc[:len(s)].equals(s) and n.dtypes.equals(s.dtypes)
    assert n.equals(pd.read_parquet(REPO))          # то же, что лежит в репозитории


def test_znacheniya_fevral_2026():
    v = znach(W.dopisat(staryj()))
    assert [v[f"2026-02-{d:02d}"] for d in (9, 10, 11, 12, 13)] == [16.0] * 5      # решение 13.02 ещё не действует
    assert [v[f"2026-02-{d:02d}"] for d in (16, 17, 18, 19, 20, 24, 25, 26, 27)] == [15.5] * 9
    assert "2026-02-23" not in v and "2026-02-28" not in v                         # нет строки в hd_base; суббота


def test_obyavlenie_vmesto_vstupleniya_lovitsya():
    r = W.reshenia()
    mut = r.assign(data_vstupleniya=r["data_resheniya"])
    n = W.P.prodlit(staryj(), mut[["data_vstupleniya", "stavka_pct"]])
    assert znach(n)["2026-02-13"] == 15.5               # мутация действительно меняет ряд
    with pytest.raises(ValueError, match="2026-02-13"):
        W.proverit(n, r, S)


def test_vhod_pozzhe_resheniya(tmp_path, monkeypatch):
    r = pd.read_csv(W.P.CSV)
    r["data_vstupleniya"] = r["data_resheniya"].where(r["data_resheniya"].str.match(r"^\d{4}-\d\d-\d\d$"), r["data_vstupleniya"])
    f = tmp_path / "r.csv"
    r.to_csv(f, index=False)
    monkeypatch.setattr(W.P, "CSV", f)
    with pytest.raises(ValueError, match="point-in-time"):
        W.reshenia()


def test_ffill_vpered_bez_resheniya_lovitsya():
    n = W.dopisat(staryj())
    mut = n.assign(value=n["value"].where(n["date"] <= S, 16.0))     # константа вперёд: 14 наблюдений
    assert (mut["date"] > S).sum() > 1
    with pytest.raises(ValueError, match="2026-02-16"):
        W.proverit(mut, W.reshenia(), S)
    W.proverit(n, W.reshenia(), S)                                    # честный ряд проходит


def test_zona_rate_on_2026_02_27(tmp_path, monkeypatch):
    old, new = tmp_path / "old.parquet", tmp_path / "new.parquet"
    staryj().to_parquet(old, index=False)
    W.dopisat(staryj()).to_parquet(new, index=False)
    dni = [D("2026-02-13"), D("2026-02-16"), D("2026-02-27")]
    monkeypatch.setattr(msp, "KEY_RATE", old)
    with pytest.raises(ValueError, match="кончается"):
        msp.rate_on(dni)
    h_old = msp.rate_cbr()[lambda x: x.index < D("2022-01-01")]
    monkeypatch.setattr(msp, "KEY_RATE", new)
    assert list(msp.rate_on(dni)) == pytest.approx([0.16, 0.155, 0.155])
    assert msp.rate_cbr()[lambda x: x.index < D("2022-01-01")].equals(h_old)   # dev-часть ряда не тронута
    with pytest.raises(PermissionError):
        zona.check(D("2026-02-27"), "ставка")             # dev (граница 2025-01-01): конец ряда закрыт
    monkeypatch.setattr(zona, "_zone", "wf")
    zona.check(D("2026-02-27"), "ставка")                 # wf: открыт
    with pytest.raises(PermissionError):
        zona.check(D("2026-03-01"), "ставка")
