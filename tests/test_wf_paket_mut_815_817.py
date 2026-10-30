"""Тестировщик пакета wf (#815, #816, #817): мутации, не пойманные тестами авторов, и найденные дефекты (xfail strict).
Синтетика, в сеть не ходит; корни временные, основная копия только читается (маркер dev)."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import proverka_sdviga_wf as P  # noqa: E402
from fastbt import daily_1850 as D  # noqa: E402
from fastbt import istochniki_v6 as iv  # noqa: E402
from fastbt import vneshnie_1850 as v  # noqa: E402
from fastbt import vneshnie_usdrub as vu  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
END = pd.Timestamp("2022-01-01")
GR = pd.Timestamp("2026-03-01")


# ---------- #815: wf без CNYRUB_TOM ----------

def _kak_815(cls, tmp_path, period, monkeypatch):
    """Поведение indeks("cnyrub") без файла и с файлом: (исход без файла, что прочитано с файлом)."""
    seen = []
    monkeypatch.setattr(iv.v18, "read_bars", lambda p, end=None: seen.append(end) or "bars")
    monkeypatch.setattr(iv.v18, "granitsa", lambda period=None, otmetka=None: GR)
    f = cls(END, tmp_path, period)
    try:
        bez = f.indeks("cnyrub")
    except FileNotFoundError as e:
        bez = "otkaz:CNYRUB_TOM" if "CNYRUB_TOM" in str(e) else "otkaz:drugoj"
    p = tmp_path / v.PATHS["cnyrub"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return bez, f.indeks("cnyrub"), seen


class _OtkazVezde(iv.Fajly):
    def indeks(self, name):
        if name.lower() == "cnyrub" and not (self.main / v.PATHS["cnyrub"]).exists():
            raise FileNotFoundError("CNYRUB_TOM")     # мутация: отказ и на train4 (ломает ≤ 2021)
        return super().indeks(name)


class _ExistsPosle(iv.Fajly):
    def indeks(self, name):
        key = name.lower()
        p = self.main / (v.PATHS[key] if key in v.PATHS else f"{iv.sek.IDX_DIR}/{name}.parquet")
        r = v.read_bars(p, v.granitsa("wf") if self.period == "wf" else self.end)   # мутация: чтение до проверки
        if self.period == "wf" and key == "cnyrub" and not p.exists():
            raise FileNotFoundError(f"CNYRUB_TOM {p}")
        return r if p.exists() else None


def test_815_wf_s_cny_chitaet_kak_prezhde_a_bez_nego_otkaz(tmp_path, monkeypatch):
    bez, s, seen = _kak_815(iv.Fajly, tmp_path, "wf", monkeypatch)
    assert (bez, s, seen) == ("otkaz:CNYRUB_TOM", "bars", [GR])        # граница wf, как у прочих индексов wf


def test_815_train4_bez_cny_none_kak_prezhde(tmp_path, monkeypatch):
    bez, s, seen = _kak_815(iv.Fajly, tmp_path, "train4", monkeypatch)
    assert (bez, s, seen) == (None, "bars", [END])


def test_815_mutaciya_otkaz_i_na_train4_lovitsya(tmp_path, monkeypatch):
    assert _kak_815(_OtkazVezde, tmp_path, "train4", monkeypatch)[0] != None  # noqa: E711  (ломает ≤ 2021)


def test_815_mutaciya_exists_posle_chteniya_lovitsya(tmp_path, monkeypatch):
    def chitat(p, end=None):
        raise FileNotFoundError(f"parquet: {p}")                       # настоящее чтение без файла падает своим текстом
    monkeypatch.setattr(iv.v18, "read_bars", chitat)
    monkeypatch.setattr(iv.v18, "granitsa", lambda period=None, otmetka=None: GR)
    with pytest.raises(FileNotFoundError, match="CNYRUB_TOM"):
        iv.Fajly(END, tmp_path, "wf").indeks("cnyrub")                  # честный отказ — с причиной
    with pytest.raises(FileNotFoundError) as e:
        _ExistsPosle(END, tmp_path, "wf").indeks("cnyrub")
    assert "parquet" in str(e.value)                                    # мутация даёт другой текст, тест выше её ловит


# ---------- #816: проверка сдвига usdrub в wf ----------

DNI = pd.bdate_range("2024-05-01", "2024-07-31")
STYK = pd.Timestamp("2024-06-11")


def _bary(dni, uroven):
    ts = [d + pd.Timedelta(hours=10, minutes=10 * k) for d in dni for k in range(12)]
    c = np.repeat(np.asarray(uroven, float), 12)
    return pd.DataFrame({"timestamp": ts, "close": c, "open": c, "high": c, "low": c, "volume": 1})


def _mir(monkeypatch, tmp_path, bez_cny=False):
    """USD — до стыка, CNY — на всём отрезке (как в данных: CNY есть и до стыка), разные уровни."""
    usd = _bary(DNI[DNI <= STYK], 90 * np.exp(np.cumsum(np.full((DNI <= STYK).sum(), 0.01))))
    cny = _bary(DNI, 12 * np.exp(np.cumsum(np.full(len(DNI), -0.02))))
    for n in ("usdrub", "cnyrub"):
        p = tmp_path / v.PATHS[n]
        p.parent.mkdir(parents=True, exist_ok=True)
        if not (bez_cny and n == "cnyrub"):
            p.write_bytes(b"")
    monkeypatch.setattr(v, "read_bars", lambda p, end=None: usd if "USD000" in str(p) else cny)


def test_816_sklejka_po_dnyam_11_06_usd_12_06_cny(monkeypatch, tmp_path):
    _mir(monkeypatch, tmp_path)
    dx = P.dx_usdrub_schet(tmp_path, pd.Timestamp("2024-07-31"))
    assert dx.loc["2024-06-11"] == pytest.approx(0.01) and dx.loc["2024-06-12"] == pytest.approx(-0.02)


def test_816_mutaciya_sklejka_12_06_lovitsya(monkeypatch, tmp_path):
    _mir(monkeypatch, tmp_path)
    monkeypatch.setattr(vu, "USD_LAST", pd.Timestamp("2024-06-12"))    # мутация: стык на день позже
    dx = P.dx_usdrub_schet(tmp_path, pd.Timestamp("2024-07-31"))
    assert not (dx.get(pd.Timestamp("2024-06-12")) == pytest.approx(-0.02))   # 12.06 уже не из CNY — тест выше упал бы


def test_816_wf_bez_cny_otkaz(monkeypatch, tmp_path):
    _mir(monkeypatch, tmp_path, bez_cny=True)
    with pytest.raises(FileNotFoundError, match="CNYRUB_TOM"):
        P.dx_usdrub_schet(tmp_path, pd.Timestamp("2024-07-31"))


# ---------- #817: маркер wf, ensure ----------

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ryad(tag: float, god: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["X", "X"], "date": pd.to_datetime([f"{god}-02-26", f"{god}-02-27"]),
                         "open": [tag, tag], "high": [tag + 1, tag + 1], "low": [tag, tag], "close": [tag, tag], "volume": [5, 6]})


def _ne_zvat(*a, **k):
    raise AssertionError("ensure пересобрал готовый ряд")


def test_817_ensure_s_vernym_markerom_ne_pishet(tmp_path, monkeypatch):
    p = tmp_path / "ohlcv_daily_1850.parquet"
    D.write_wf(_ryad(2.0, "2026"), p)
    s0, m0, t0 = _sha(p), D.marker(p).read_bytes(), p.stat().st_mtime_ns
    monkeypatch.setattr(D, "build", _ne_zvat)
    D.ensure(p, tmp_path)
    assert (_sha(p), D.marker(p).read_bytes(), p.stat().st_mtime_ns) == (s0, m0, t0)


def test_817_mutaciya_ensure_bez_proverki_gotovnosti_lovitsya(tmp_path, monkeypatch):
    p = tmp_path / "ohlcv_daily_1850.parquet"
    D.write_wf(_ryad(2.0, "2026"), p)
    monkeypatch.setattr(D, "is_ready", lambda path: False)             # мутация: готовность не проверяется
    monkeypatch.setattr(D, "build", lambda main=None: _ryad(9.0, "2021"))
    D.ensure(p, tmp_path)
    assert pd.read_parquet(p)["open"].iloc[0] == 9.0                   # затёрт; тест выше (_ne_zvat) эту мутацию роняет


def test_817_nevernyj_marker_wf_gromkij_otkaz_fajl_ne_tronut(tmp_path, monkeypatch):
    p = tmp_path / "ohlcv_daily_1850.parquet"
    D.write_wf(_ryad(2.0, "2026"), p)
    D.marker(p).write_text("0" * 64)                                   # маркер от другого содержимого
    s0, t0 = _sha(p), p.stat().st_mtime_ns
    monkeypatch.setattr(D, "build", _ne_zvat)
    with pytest.raises(RuntimeError, match="маркер не сходится"):
        D.ensure(p, tmp_path)
    assert (_sha(p), p.stat().st_mtime_ns) == (s0, t0)


def test_817_nevernyj_marker_dev_pereborka_prezhnyaya_pobajtno(tmp_path, monkeypatch):
    p, ref = tmp_path / "ohlcv_daily_1850.parquet", tmp_path / "ref.parquet"
    dev = _ryad(5.0, "2021")
    D.write_wf(dev, p)
    D.marker(p).write_text("0" * 64)
    new = _ryad(9.0, "2021")
    new.to_parquet(ref, index=False)
    monkeypatch.setattr(D, "build", lambda main=None: new)
    D.ensure(p, tmp_path)
    assert p.read_bytes() == ref.read_bytes() and D.is_ready(p)        # пересобран из build, маркер заново


def test_817_mutaciya_otkaz_na_dev_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "END", pd.Timestamp("2000-01-01"))          # мутация: порог отказа сдвинут — dev тоже «wf»
    p = tmp_path / "ohlcv_daily_1850.parquet"
    D.write_wf(_ryad(5.0, "2021"), p)
    D.marker(p).write_text("0" * 64)
    monkeypatch.setattr(D, "build", lambda main=None: _ryad(9.0, "2021"))
    with pytest.raises(RuntimeError):
        D.ensure(p, tmp_path)                                          # прежний тест пересборки dev на ней упал бы


def test_817_mutaciya_otkaz_snyat_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_ne_zatirat_wf", lambda path: None)        # мутация: защита снята
    p = tmp_path / "ohlcv_daily_1850.parquet"
    D.write_wf(_ryad(2.0, "2026"), p)
    D.marker(p).write_text("0" * 64)
    monkeypatch.setattr(D, "build", lambda main=None: _ryad(9.0, "2021"))
    D.ensure(p, tmp_path)
    assert pd.read_parquet(p)["date"].max().year == 2021               # wf затёрт — тест отказа выше упал бы


def test_817_marker_dev_osnovnoj_kopii_ne_menyaetsya_pri_importe(tmp_path):
    main = REPO.parents[2]
    m = main / "data" / "fastbt" / "daily" / "ohlcv_daily_1850.parquet.sha256"
    if not m.exists():
        pytest.skip("нет маркера dev в основной копии (только чтение)")
    before = m.read_bytes()
    p = tmp_path / "data" / "fastbt" / "daily" / "ohlcv_daily_1850.parquet"
    p.parent.mkdir(parents=True)
    D.write_wf(_ryad(2.0, "2026"), p)
    script = tmp_path / "imp.py"
    script.write_text(f"import sys\nsys.path.insert(0, {str(REPO)!r})\nimport fastbt.data\n")
    r = subprocess.run([sys.executable, str(script)], cwd=REPO, env={**os.environ, "FASTBT_MAIN": str(tmp_path)},
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-300:]
    assert m.read_bytes() == before


def test_817_python_m_fastbt_data_bez_markera_ne_zavisaet(tmp_path):
    env = {**os.environ, "FASTBT_MAIN": str(tmp_path)}
    env.pop("FASTBT_DATA_ROOT", None)
    subprocess.run([sys.executable, "-m", "fastbt.data"], cwd=REPO, env=env, capture_output=True, text=True, timeout=20)


@pytest.mark.xfail(strict=True, raises=subprocess.TimeoutExpired,
                   reason="#823: `python -m fastbt.daily_1850` без маркера зависает (воспроизведено, таймаут 20 с): ensure держит flock, "
                          "build() импортирует fastbt.data, чей импорт зовёт ensure на том же lock")
def test_823_python_m_daily_1850_bez_markera_ne_zavisaet(tmp_path):
    env = {**os.environ, "FASTBT_MAIN": str(tmp_path)}
    env.pop("FASTBT_DATA_ROOT", None)
    subprocess.run([sys.executable, "-m", "fastbt.daily_1850"], cwd=REPO, env=env, capture_output=True, text=True, timeout=20)
