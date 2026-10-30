"""wf-int-a1: интеграция wf в источники v6 и пути σ̂ (Ж-140). Синтетика + проверка реальных файлов σ̂ wf (нет данных — skip).

Известный ответ: на wf dx оси usdrub после 11.06.2024 = лог-доходность CNYRUB_TOM внутри CNY-ряда (не пусто, не ffill);
`cnyrub` — в sha только на wf (Д5); на прочих периодах имена рядов sha и ось usdrub — прежние (train4 побайтно).
Мутации: «wf берёт ch4_dev» (katalog без периода), «cnyrub не в sha», «ось usdrub wf по USD-пути» — роняют тесты ниже.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import istochniki_v6 as iv, s9, sigma_e, vneshnie_1850 as v18

MIN = [9 * 60 + 50 + 10 * k for k in range(54)]
USD_DAYS = pd.bdate_range("2024-05-13", "2024-06-11")
CNY_DAYS = pd.bdate_range("2024-05-20", "2024-06-28")


def _bars(seed: int, days, s0: float = 90.0) -> pd.DataFrame:
    rng, rows, p = np.random.default_rng(seed), [], s0
    for d in days:
        for m in MIN:
            q = p * float(np.exp(rng.normal(0, 0.002)))
            rows.append((d + pd.Timedelta(minutes=m), p, max(p, q) * 1.0005, min(p, q) * 0.9995, q, 1000.0))
            p = q
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


class Fake(iv.Fajly):
    def __init__(self, idx: dict, period=None):
        self.end, self.main, self.i, self.period = pd.Timestamp("2026-03-01"), None, idx, period

    def akcii(self, t):
        return _bars(5, USD_DAYS)

    def indeks(self, name):
        return self.i.get(name.lower())

    def zakrytie(self, t):
        return v18.close_1850(self.akcii(t), t)

    def brent(self):
        return {}, pd.DataFrame({"secid": [], "lsttrade": [], "roll": []})

    def kalendar(self):
        return pd.DataFrame({"date": [], "known_from": []}), pd.DataFrame()


def _idx(cny_seed: int = 2):
    return {"usdrub": _bars(1, USD_DAYS), "cnyrub": _bars(cny_seed, CNY_DAYS, 12.0)}


def _dx(period):
    return iv._Sbor(Fake(_idx(), period), ["AAA"]).faktor("usdrub")[0]


def test_wf_usdrub_posle_12_06_iz_cny():
    dx = _dx("wf")
    cny = v18.close_1850(_idx()["cnyrub"], "CNYRUB")
    ожид = np.log(cny).diff().loc["2024-06-12":]
    assert len(ожид) > 10 and dx.loc["2024-06-12":].notna().all()
    assert np.allclose(dx.loc["2024-06-12":].to_numpy(), ожид.to_numpy())
    assert dx.loc[:"2024-06-11"].notna().sum() > 10       # до стыка — USD-ряд


def test_mutaciya_wf_po_usd_puti_lovitsya():
    """Не-wf: прежний путь — после 11.06.2024 значений нет (CNY не нужен); тест выше на нём падал бы."""
    assert _dx(None).loc["2024-06-12":].empty


def test_sha_cnyrub_tolko_v_wf():
    names = lambda p: [n for n, _ in Fake(_idx(), p).ryady(["AAA"])]    # noqa: E731
    assert "indeks:cnyrub" in names("wf") and "indeks:cnyrub" not in names(None) and "indeks:cnyrub" not in names("train4")
    assert [n for n in names("wf") if n != "indeks:cnyrub"] == names(None)


def test_sha_menyaetsya_ot_cnyrub_na_wf_i_ne_menyaetsya_na_train4():
    h = lambda p, s: iv.sha_ryadov(Fake(_idx(s), p), ["AAA"])    # noqa: E731
    assert h("wf", 2) != h("wf", 3)                     # мутация «cnyrub не в sha» роняет это
    assert h(None, 2) == h(None, 3)


def test_indeks_wf_chitaet_do_granicy_wf(tmp_path, monkeypatch):
    p = tmp_path / v18.PATHS["usdrub"]
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    seen = []
    monkeypatch.setattr(iv.v18, "read_bars", lambda path, end=None: seen.append(end) or "bars")
    monkeypatch.setattr(iv.v18, "granitsa", lambda period=None, otmetka=None: pd.Timestamp("2026-03-01"))
    assert iv.Fajly(pd.Timestamp("2022-01-01"), tmp_path, "wf").indeks("usdrub") == "bars"
    assert iv.Fajly(pd.Timestamp("2022-01-01"), tmp_path).indeks("usdrub") == "bars"
    assert seen == [pd.Timestamp("2026-03-01"), pd.Timestamp("2022-01-01")]


def _write(path: Path, val: float, last: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame({"date": pd.bdate_range("2021-12-27", last), "ticker": "AAA"})
    pd.DataFrame({"date": d["date"], "ticker": "AAA", "sigma_pred_h1": np.float32(val)}).to_parquet(path, index=False)


def test_sigma_wf_iz_kataloga_wf_ne_ch4_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(sigma_e, "CH4", tmp_path / "dev")
    monkeypatch.setattr(sigma_e, "WF_CH4", tmp_path / "wf")
    _write(tmp_path / "dev" / "sigma_hyb_h1_dev.parquet", 0.01, "2022-01-14")
    _write(tmp_path / "wf" / "sigma_hyb_h1_dev.parquet", 0.02, "2026-03-13")
    w, d = sigma_e.load_model("hyb_h1", "wf"), sigma_e.load_model("hyb_h1")
    assert np.allclose(w["sigma"], 0.02) and w["date"].max() == pd.Timestamp("2026-02-27")   # обрезка по концу wf 28.02
    assert np.allclose(d["sigma"], 0.01) and d["date"].max() == pd.Timestamp("2021-12-31")   # прежнее: ≤ 2021
    with pytest.raises(FileNotFoundError):
        sigma_e.load_model("harj_h1", "wf")             # в каталоге wf только гибрид — не молчаливая подмена


def test_s9_sigma_src_po_periodu():
    assert s9.sigma_src("train4") == (s9.SOURCES_V2, s9.CH4 / "SHA256.txt") and s9.sigma_src(None)[0] is s9.SOURCES_V2
    src, sha = s9.sigma_src("wf")
    assert sorted(src) == ["hyb_h1", "hyb_h5"] and sha == s9.CH4_WF / "SHA256SUMS"
    assert all(p.parent == s9.CH4_WF for p, _, _ in src.values())


MAIN = sigma_e.MAIN


@pytest.mark.skipif(not (sigma_e.WF_CH4 / "SHA256SUMS").exists(), reason="нет data/regen/wf/ch4_sigma")
def test_realnye_sigma_wf_sha_i_do_2022_pobajtno():
    src, sha = s9.sigma_src("wf")
    ref = {ln.split()[1]: ln.split()[0] for ln in sha.read_text().splitlines() if ln.strip()}
    for p, _, _ in src.values():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == ref[p.name]
    w, d = sigma_e.load_model("hyb_h1", "wf"), sigma_e.load_model("hyb_h1")
    assert w["date"].max() > pd.Timestamp("2026-01-01") and w["date"].max() <= pd.Timestamp("2026-02-28")
    pd.testing.assert_frame_equal(w[w["date"] <= sigma_e.LAST].reset_index(drop=True), d)
