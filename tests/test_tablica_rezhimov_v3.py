"""Таблица дат режимов для v3 (Ж-159 п. 7, Ж-161 п. 1): режим дня d — компонента `trend` fazy (_LAG = 1), IMOEX ≤ d−1.
Синтетика с известным ответом, без сети. Мутации: лаг 0; MA50/MA200 переставлены; граница чтения снята.
python3 -m pytest tests/test_tablica_rezhimov_v3.py -q"""
import contextlib
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from fastbt import fazy, period_v2, tablica_rezhimov_v3 as T, zona

D = pd.Timestamp
READ: list = []


def _bary(tp: Path, close: pd.Series, nm: str = "IMOEX.parquet") -> Path:
    """10-минутки: два бара в день (10:00 и 18:40), close дня = close второго бара; формат внешних рядов (begin — строка)."""
    d = np.repeat(close.index.strftime("%Y-%m-%d"), 2)
    hm = np.tile(["10:00:00", "18:40:00"], len(close))
    c = np.repeat(close.to_numpy(float), 2)
    p = tp / nm
    pd.DataFrame({"begin": [f"{a} {b}" for a, b in zip(d, hm)], "open": c, "high": c, "low": c, "close": c,
                  "value": 1.0, "volume": 1}).to_parquet(p)
    return p


def _ryad(vals) -> pd.Series:
    return pd.Series(np.asarray(vals, float), index=pd.bdate_range("2014-08-26", periods=len(vals)))


ROST = _ryad(100 + np.arange(300))
PADENIE = _ryad(1000 - np.arange(300))
_b = 100 + np.arange(303.0)
_b[300] = 50.0                      # провал на одном дне: close < MA200, но MA50 ещё выше MA200
BOKOVIK = _ryad(_b)


def _izvestnyj(tp: Path) -> None:
    """Эталонный ответ, выведенный руками: день i смотрит на закрытие i−1; MA200 есть у закрытия i ≥ 199 → net_ma у дней i < 200."""
    for nm, ryad, rez in (("rost", ROST, "rost"), ("pad", PADENIE, "padenie")):
        tab, n = T.tablica(_bary(tp, ryad, nm + ".parquet"), "2014-08-26", "2016-12-31")
        r = tab.set_index("data")["rezhim"]
        assert n == 200 and len(r) == 300, nm
        assert (r.iloc[:200] == "net_ma").all() and (r.iloc[200:] == rez).all(), nm
    tab, n = T.tablica(_bary(tp, BOKOVIK, "bok.parquet"), "2014-08-26", "2016-12-31")
    r = tab.set_index("data")["rezhim"]
    assert n == 200
    assert (r.iloc[299], r.iloc[300], r.iloc[301], r.iloc[302]) == ("rost", "rost", "bokovik", "rost")


def test_izvestnyj_otvet_rost_padenie_bokovik(tmp_path):
    _izvestnyj(tmp_path)


def test_mutaciya_lag_nol_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(fazy, "_LAG", 0)           # режим по close дня d
    with pytest.raises(AssertionError):
        _izvestnyj(tmp_path)


def test_mutaciya_ma50_ma200_perestavleny_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(fazy, "OKNO_MA_D", fazy.OKNO_MA_K)
    monkeypatch.setattr(fazy, "OKNO_MA_K", 200)
    with pytest.raises(AssertionError):
        _izvestnyj(tmp_path)


def test_sovpadenie_s_fazy_na_sinteticheskom_ryade(tmp_path):
    rng = np.random.default_rng(7)
    ryad = _ryad(1000 * np.exp(np.cumsum(rng.normal(0, 0.01, 900))))
    tab, _ = T.tablica(_bary(tmp_path, ryad), "2014-08-26", "2018-12-31")
    ref = fazy._priznaki(ryad)["trend"].reindex(ryad.index)
    oj = ref.map({"рост": "rost", "падение": "padenie", "боковик": "bokovik"}).fillna("net_ma")
    assert list(tab["data"]) == list(ryad.index) and list(tab["rezhim"]) == list(oj)
    assert {"rost", "padenie", "bokovik"} <= set(tab["rezhim"])        # синтетика содержит все три режима


# ---- граница чтения (зона polnaya, < 2025-01-01 до акта) ----
def _zapomnit(t) -> None:
    t = t if isinstance(t, pd.DataFrame) else t.to_pandas()
    if "begin" in t.columns and len(t):
        READ.append(pd.to_datetime(t["begin"]).max())


@pytest.fixture
def zamok(tmp_path, monkeypatch):
    READ.clear()
    otm = tmp_path / "wf_zamok_snyat.json"
    otm.write_text('{"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "%s", "data_snyatiya": "2026-10-30"}' % ("a" * 64))
    monkeypatch.setattr(period_v2, "OTMETKA", otm)
    rt = pq.read_table

    def obertka(*a, **k):
        t = rt(*a, **k)
        _zapomnit(t)
        return t

    monkeypatch.setattr(pq, "read_table", obertka)
    yield
    zona.set_zone("dev")


def _s_hvostom(tp: Path) -> Path:
    dni = pd.bdate_range("2014-08-26", "2026-02-27")
    return _bary(tp, pd.Series(1000 + 50 * np.sin(np.arange(len(dni)) / 40), index=dni))


def _glavnaya(tp: Path, do: str) -> Path:
    out = tp / "t.csv"
    T.main(["--do", do, "--out", str(out), "--imoex", str(_s_hvostom(tp))])
    return out


def test_granitsa_do_2024_12_31_nichego_ne_ot_2025(tmp_path, zamok):
    out = _glavnaya(tmp_path, "2024-12-31")
    assert READ and max(READ) < D("2025-01-01")
    dd = pd.read_csv(out, comment="#")["data"]
    assert len(dd) > 2000 and pd.to_datetime(dd).max() < D("2025-01-01")


def test_granitsa_do_za_chteniem_otkaz_bez_fajla(tmp_path, zamok):
    with pytest.raises(PermissionError):
        _glavnaya(tmp_path, "2025-06-30")
    assert not (tmp_path / "t.csv").exists()


def test_mutaciya_granitsa_snyata_lovitsya(tmp_path, zamok, monkeypatch):
    monkeypatch.setattr(zona, "end", lambda: D("2026-03-01"))
    with contextlib.suppress(Exception):           # после чтения ≥ 2025 aggregate просит календарь MOEX — для проверки чтения не важно
        _glavnaya(tmp_path, "2024-12-31")
    with pytest.raises(AssertionError):
        assert READ and max(READ) < D("2025-01-01")


def test_shapka_hash_i_chislo_net_ma(tmp_path, zamok, capsys):
    out = _glavnaya(tmp_path, "2024-12-31")
    txt = out.read_text(encoding="utf-8").splitlines()
    sh = {ln[2:].split(": ", 1)[0]: ln[2:].split(": ", 1)[1] for ln in txt if ln.startswith("# ")}
    assert sh["v"] == "1" and sh["pravilo"] == "В-4 v3"
    assert sh["kod"] == hashlib.sha256(Path(fazy.__file__).read_bytes()).hexdigest()
    assert int(sh["net_ma"]) == sum(ln.endswith(",net_ma") for ln in txt) == 200
    assert "istochnik" in sh and len(sh["istochnik"]) == 64
    pech = capsys.readouterr().out
    assert hashlib.sha256(out.read_bytes()).hexdigest() in pech and "2014" in pech and "2024" in pech
