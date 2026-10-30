"""σ̂ C/D на wf, версия 2 (scripts/dannye/wf_sigma_cd_v2.py): синтетика, реальные данные не читаются, сеть не нужна.
Мутации: шаг 10 читает ≥ 2026-02-04; дыра в склейке; σ̂ = rv_pred без корня; дата ДСВД в v2; шаги 1–8 не побайтно = v1; чужой SHA итога К-6.
python3 -m pytest tests/test_wf_sigma_cd_v2.py -q"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("wsc2", Path(__file__).resolve().parents[1] / "scripts" / "dannye" / "wf_sigma_cd_v2.py")
V2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V2)
W = V2.W

TICKERS = ("AAA", "BBB")
DAYS = pd.bdate_range("2016-11-01", "2026-02-03")
DSVD = (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-02"))     # сб, вс; понедельник после них — 2025-03-03
PN = pd.Timestamp("2025-03-03")
K_PN = 1.1                                                           # v1 на понедельнике после ДСВД отличается: rv_pred × 1,1


def _kal(seed=3):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame([(d, t) for t in TICKERS for d in DAYS], columns=["date", "ticker"])
    f["rv"] = np.exp(rng.normal(-8, 1, len(f)))                      # iid: сдвиг на день разрушает корреляцию
    return f.sort_values(["ticker", "date"]).reset_index(drop=True)


def _istochnik(h, kal, seed=5):
    rng = np.random.default_rng(seed + h)
    d = kal[kal["date"] >= "2017-01-01"][["date", "ticker", "rv"]].copy()
    d["year"], d["step"] = d["date"].dt.year, d["date"].dt.year - 2016
    d["leg_raw"] = d["rv"] * np.exp(rng.normal(0, 0.2, len(d)))
    d["F_sm"] = d["step"].map({k: 1.1 + 0.01 * k for k in range(1, 11)})
    d["rv_pred"] = d["leg_raw"] * d["F_sm"]
    d["F_sm_bez_emb"], d["design"] = d["F_sm"], W.DESIGN[h]
    return d.drop(columns="rv").sort_values(["step", "date", "ticker"]).reset_index(drop=True)


def _v1(d, h):
    """v1 = ряд с ДСВД: те же строки + сб/вс 2025-03-01/02; понедельник после них отличается."""
    ex = d[d["date"] == PN].assign(date=DSVD[0])
    ex = pd.concat([ex, ex.assign(date=DSVD[1])])
    x = pd.concat([d.assign(rv_pred=np.where(d["date"] == PN, d["rv_pred"] * K_PN, d["rv_pred"])), ex])
    return W.H.sigma_ch3(x, "rv_pred", W.oc(h))


def make_root(tmp, mut=None, bad_sha=False):
    """Мини-корень: kal без ДСВД, v1 с ДСВД, ch4_dev, итог К-6 + sborka_shapka.json. mut(d, h) подменяет итог К-6 после сборки остального."""
    root, kal, shapka = tmp / "root", _kal(), {}
    for p in (W.KAL, W.CH4.format(h=1), V2.V1.format(h=1), V2.K6 + "/x"):
        (root / p).parent.mkdir(parents=True, exist_ok=True)
    kal.to_parquet(root / W.KAL, index=False)
    for h in W.HORIZONS:
        d = _istochnik(h, kal)
        _v1(d, h).to_parquet(root / V2.V1.format(h=h), index=False)
        W.H.sigma_ch3(d[d["date"] < "2022-01-01"], "rv_pred", W.oc(h)).to_parquet(root / W.CH4.format(h=h), index=False)
        p = root / V2.K6 / V2.K6F.format(h=h)
        (mut(d, h) if mut else d).to_parquet(p, index=False)
        shapka[f"vyhod_h{h}"] = "0" * 64 if bad_sha else W.sha(p)
    (root / V2.K6 / "sborka_shapka.json").write_text(json.dumps(shapka))
    return root


@pytest.fixture(autouse=True)
def bez_zamka(monkeypatch):
    calls = []
    monkeypatch.setattr(W.period_v2, "proverit_chtenie", lambda period, put=None, **k: calls.append((period, str(put))))
    return calls


# ---------------- G1, G2, G3, G6: формат, = v1, ДСВД, Δ ----------------

def test_run_formatt_v1_dsvd_i_delta(tmp_path):
    out = tmp_path / "out"
    rep = V2.run(make_root(tmp_path), out)
    meta = json.loads((out / "wf_sigma_run.json").read_text())
    assert meta["v"] == 2 and meta["d1_ok"] is True and meta["konec"] == "2026-02-03"
    assert meta["dni"] == "osnovnaya_sessiya MOEX >= 2025, DSVD isklyucheny" and len(meta["kalendar_moex_sha256"]) == 64
    for h in W.HORIZONS:
        s = pd.read_parquet(out / f"sigma_hyb_h{h}_dev.parquet")
        assert list(s.columns) == ["date", "ticker", W.oc(h)] and s[W.oc(h)].dtype == np.float32
        assert s.equals(s.sort_values(["date", "ticker"]).reset_index(drop=True)) and not s.duplicated(["date", "ticker"]).any()
        assert not s["date"].isin(DSVD).any() and s["date"].max() == pd.Timestamp("2026-02-03")
        v = meta[f"hyb_h{h}"]["v1"]
        assert v["strok_v1_vne_v2"] == 2 * len(TICKERS) and v["dsvd_dat"] == ["2025-03-01", "2025-03-02"]
        dl = meta[f"hyb_h{h}"]["delta_v1_v2"]
        assert dl["po_shagam"]["3"]["med"] == 0.0 and dl["po_shagam"]["9"]["max"] == pytest.approx(0.5 * np.log(K_PN), abs=1e-6)
        assert dl["ponedelniki_posle_dsvd"]["n"] == len(TICKERS)
        assert dl["ponedelniki_posle_dsvd"]["med"] == pytest.approx(0.5 * np.log(K_PN), abs=1e-6)
        assert dl["ostalnye_dni_shagov_9_10"]["max"] == 0.0
    assert set(rep["_shagi"]["step"]) == set(range(1, 11)) and "SHA256SUMS" in {p.name for p in out.iterdir()}


def test_sigma_ravna_koren_iz_rv_pred(tmp_path):
    out = tmp_path / "out"
    V2.run(make_root(tmp_path), out)
    d = _istochnik(1, _kal())
    s = pd.read_parquet(out / "sigma_hyb_h1_dev.parquet").merge(d, on=["date", "ticker"])
    assert np.array_equal(s[W.oc(1)].to_numpy(), np.sqrt(s["rv_pred"].to_numpy()).astype("float32"))


def test_load_model_wf_chitaet_katalog_v2(tmp_path, monkeypatch):
    from fastbt import sigma_e
    out = tmp_path / "out"
    V2.run(make_root(tmp_path), out)
    monkeypatch.setattr(sigma_e, "WF_CH4", out)
    m = sigma_e.load_model("hyb_h1", "wf")
    assert len(m) > 0 and sigma_e.katalog("wf") == out


# ---------------- G5: мутации ----------------

def test_mutaciya_shag10_chitaet_posle_konca_lovitsya(tmp_path):
    def mut(d, h):
        x = d[d["date"] == "2026-02-03"].assign(date=pd.Timestamp("2026-02-04"))
        return pd.concat([d, x], ignore_index=True)
    with pytest.raises(SystemExit, match="после 2026-02-03"):
        V2.run(make_root(tmp_path, mut), None)


def test_mutaciya_dyra_v_sklejke_lovitsya(tmp_path):
    def mut(d, h):
        return d[d["date"] != "2025-06-17"].reset_index(drop=True)
    with pytest.raises(SystemExit, match="гейт 5"):
        V2.run(make_root(tmp_path, mut), None)


def test_mutaciya_sigma_bez_kornya_lovitsya(tmp_path, monkeypatch):
    def bez_kornya(df, col, out_col):
        sp = df.assign(**{out_col: df[col].values.astype("float32")}).sort_values(["date", "ticker"])
        return sp[["date", "ticker", out_col]].reset_index(drop=True)
    root = make_root(tmp_path)
    monkeypatch.setattr(W.H, "sigma_ch3", bez_kornya)
    with pytest.raises(SystemExit, match="ch4_dev|= v1"):
        V2.run(root, None)


def test_mutaciya_data_dsvd_v_v2_lovitsya(tmp_path):
    def mut(d, h):
        return pd.concat([d, d[d["date"] == PN].assign(date=DSVD[0])], ignore_index=True)
    with pytest.raises(SystemExit, match="ДСВД"):
        V2.run(make_root(tmp_path, mut), None)


def test_mutaciya_shag_1_8_ne_pobajtno_lovitsya(tmp_path):
    def mut(d, h):
        d = d.copy()
        i = d.index[d["date"] == "2023-05-10"][0]
        d.loc[i, "rv_pred"] *= 1 + 1e-6                               # ≈ 1 ulp float32 в σ̂
        d.loc[i, "leg_raw"] *= 1 + 1e-6
        return d
    with pytest.raises(SystemExit, match="= v1"):
        V2.run(make_root(tmp_path, mut), None)


def test_mutaciya_sha_ne_iz_shapki_lovitsya_do_chteniya(tmp_path, monkeypatch):
    root = make_root(tmp_path, bad_sha=True)

    def nelzya(*a, **k):
        raise AssertionError("данные прочитаны до сверки SHA")
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: nelzya() if "wf_k6" in str(p) else pd.DataFrame(columns=["date", "ticker", "rv"]))
    with pytest.raises(SystemExit, match="гейт 1"):
        V2.run(root, None)
