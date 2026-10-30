"""σ̂ C/D на walk-forward (scripts/dannye/wf_sigma_cd.py): синтетика, реальные данные не читаются, сеть не нужна.
Мутации: строка года t из шага, обученного с годом t; дыра и дубль в склейке; ≤ 2021 не побайтно; утечка +1 в Д1; чтение до сверки SHA.
python3 -m pytest tests/test_wf_sigma_cd.py -q"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_spec = importlib.util.spec_from_file_location("wsc", Path(__file__).resolve().parents[1] / "scripts" / "dannye" / "wf_sigma_cd.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

TICKERS = ("AAA", "BBB")
DAYS = pd.bdate_range("2016-11-01", "2026-02-03")


def _rv(seed=3):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame([(d, t) for t in TICKERS for d in DAYS], columns=["date", "ticker"])
    f["rv"] = np.exp(rng.normal(-8, 1, len(f)))                       # iid: сдвиг на день разрушает корреляцию
    return f.sort_values(["ticker", "date"]).reset_index(drop=True)


def _istochnik(h, kal, seed=5):
    """Выход сборки в миниатюре: шаг k — год 2016 + k, одна F_sm на шаг, rv_pred = leg_raw × F_sm (тот же день, что RV)."""
    rng = np.random.default_rng(seed + h)
    d = kal[kal["date"] >= "2017-01-01"][["date", "ticker", "rv"]].copy()
    d["year"] = d["date"].dt.year
    d["step"] = d["year"] - 2016
    d["leg_raw"] = d["rv"] * np.exp(rng.normal(0, 0.2, len(d)))
    d["F_sm"] = d["step"].map({k: 1.1 + 0.01 * k for k in range(1, 11)})
    d["rv_pred"] = d["leg_raw"] * d["F_sm"]
    d["F_sm_bez_emb"] = d["F_sm"]
    d["design"] = W.DESIGN[h]
    return d.drop(columns="rv").sort_values(["step", "date", "ticker"]).reset_index(drop=True)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_main(tmp):
    """Мини-«основная копия»: исходник сборки (_ishodnye), части нарезки, ch4_dev, features_pit_v3; спецификация с sobrano."""
    main, kal, sob = tmp / "main", _rv(), {}
    (main / Path(W.KAL).parent).mkdir(parents=True)
    kal.to_parquet(main / W.KAL, index=False)
    for h in W.HORIZONS:
        d = _istochnik(h, kal)
        p = main / W.ISTOCHNIK.format(h=h)
        p.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(p, index=False)
        sob[f"R2/final/final_d0_s2emb_h{h}_test.parquet"] = _sha(p)
        for nm, (lo, hi) in {"usechennaya": ("2017", "2024-12-31"), "t1_2022_2024": ("2022", "2024-12-31"),
                             "holdout_2025": ("2025", "2026-12-31")}.items():
            q = main / W.CHASTI[nm].format(h=h)
            q.parent.mkdir(parents=True, exist_ok=True)
            d[(d["date"] >= lo) & (d["date"] <= hi)].to_parquet(q, index=False)
        c = main / W.CH4.format(h=h)
        c.parent.mkdir(parents=True, exist_ok=True)
        W.H.sigma_ch3(d[d["date"] < "2022-01-01"], "rv_pred", W.oc(h)).to_parquet(c, index=False)
    spec = tmp / "spec.yaml"
    spec.write_text(yaml.safe_dump({"sobrano": sob}, allow_unicode=True))
    return main, spec, kal


@pytest.fixture
def bez_zamka(monkeypatch):
    """Проверка замка — через подмену с учётом вызовов (отметка владельца в тесте не нужна, но вызов обязателен)."""
    calls = []
    monkeypatch.setattr(W.period_v2, "proverit_chtenie", lambda period, put=None, **k: calls.append((period, str(put))))
    return calls


# ---------------- гейт 2: строка года t — из шага, обученного на данных < t ----------------

def test_shagi_ok_i_obuchen_do():
    d = _istochnik(5, _rv())
    t = W.proverit_shagi(d, 5)
    assert t["step"].tolist() == list(range(1, 11))
    assert t["obuchen_do"].tolist() == [f"{2016 + k}-01-01" for k in range(1, 11)]
    assert (pd.to_datetime(t["pervaya"]) >= pd.to_datetime(t["obuchen_do"])).all()


def test_mutaciya_shag_obuchen_s_godom_t_lovitsya():
    """Строки 2019 взяты из шага 4 (2019 — его год валидации/ранней остановки, тест 2020): ряд года t использует год t."""
    d = _istochnik(5, _rv())
    m = d["date"].dt.year == 2019
    d.loc[m, "step"] = 4
    with pytest.raises(SystemExit, match="обученного с годом t"):
        W.proverit_shagi(d, 5)


def test_mutaciya_metka_goda_podognana_lovitsya():
    """Тот же подлог с подогнанной меткой year = 2016 + step: ловит сверка года даты."""
    d = _istochnik(5, _rv())
    m = d["date"].dt.year == 2019
    d.loc[m, "step"] = 4
    d.loc[m, "year"] = 2020
    d.loc[m, "F_sm"] = d.loc[d["step"] == 4, "F_sm"].iloc[0]
    with pytest.raises(SystemExit, match="год даты"):
        W.proverit_shagi(d, 5)


def test_mutaciya_dve_F_na_shag_i_chuzhoj_design_lovyatsya():
    d = _istochnik(1, _rv())
    d.loc[d.index[:3], "F_sm"] = 9.0
    d.loc[d.index[:3], "rv_pred"] = d.loc[d.index[:3], "leg_raw"] * 9.0
    with pytest.raises(SystemExit, match="F_sm"):
        W.proverit_shagi(d, 1)
    d = _istochnik(1, _rv()).assign(design=W.DESIGN[5])
    with pytest.raises(SystemExit, match="design"):
        W.proverit_shagi(d, 1)


# ---------------- гейт 5: склейка без дыр и дублей ----------------

def _sig(h=5):
    kal = _rv()
    return W.H.sigma_ch3(_istochnik(h, kal), "rv_pred", W.oc(h)), kal


def test_skleyka_bez_dyr_i_dubley():
    s, kal = _sig()
    r = W.proverit_skleyku(s, kal, 5)
    assert r["strok"] == len(s) and r["pervaya"] == "2017-01-02" and r["poslednyaya"] == "2026-02-03"


def test_mutaciya_dyra_na_styke_shagov_lovitsya():
    s, kal = _sig()
    s = s[~((s["ticker"] == "AAA") & (s["date"] == pd.Timestamp("2022-01-03")))]   # первый день шага 6
    with pytest.raises(SystemExit, match="дыры"):
        W.proverit_skleyku(s, kal, 5)


def test_mutaciya_dubl_na_styke_shagov_lovitsya():
    s, kal = _sig()
    s = pd.concat([s, s[s["date"] == pd.Timestamp("2021-12-31")]], ignore_index=True)
    with pytest.raises(SystemExit, match="дубли"):
        W.proverit_skleyku(s, kal, 5)


def test_mutaciya_lishnyaya_data_lovitsya():
    s, kal = _sig()
    kal = kal[~((kal["ticker"] == "BBB") & (kal["date"] == pd.Timestamp("2025-06-02")))]
    with pytest.raises(SystemExit, match="лишние"):
        W.proverit_skleyku(s, kal, 5)


# ---------------- гейт 4: ≤ 2021 побайтно = ch4_dev ----------------

def test_dev_pobajtno_i_mutaciya_ulp():
    s, _ = _sig()
    dev = s[s["date"] < "2022-01-01"].copy()
    assert W.proverit_dev(s, dev, 5) == len(dev)
    i = dev.index[7]
    dev.loc[i, W.oc(5)] = np.nextafter(dev.loc[i, W.oc(5)], np.float32(1), dtype=np.float32)
    with pytest.raises(SystemExit, match="побайтно"):
        W.proverit_dev(s, dev, 5)


# ---------------- гейт 6: Д1 и контроли ----------------

def test_d1_ryad_prohodit_kontroli_blokiruyutsya():
    s, kal = _sig()
    t = W.d1(s, W.oc(5), kal, "hyb_h5")
    assert t.loc[t["rol"] == "ряд", "ok"].all() and len(t) == 3 * len(W.PERIODY_D1)
    assert (t.loc[t["rol"] == "ряд", "argmax"] == 0).all()
    assert not t.loc[t["rol"] == "контроль", "ok"].any()


def test_mutaciya_utechka_plus_odin_lovitsya():
    s, kal = _sig()
    with pytest.raises(SystemExit, match="Д1"):
        W.d1(W.sdvinut(s, W.oc(5), +1), W.oc(5), kal, "hyb_h5")


# ---------------- гейты 1, 3 и запуск целиком ----------------

def test_sha_sveryaetsya_do_chteniya(tmp_path, monkeypatch, bez_zamka):
    main, spec, _ = make_main(tmp_path)
    p = main / W.ISTOCHNIK.format(h=5)
    p.write_bytes(p.read_bytes() + b"\0")                     # исходник изменён после сборки

    def nelzya(*a, **k):
        raise AssertionError("данные прочитаны до сверки SHA")
    monkeypatch.setattr(W.pd, "read_parquet", nelzya)
    with pytest.raises(SystemExit, match="гейт 1"):
        W.chitat(main, 5, spec)
    assert bez_zamka and bez_zamka[0][0] == "wf"


def test_mutaciya_chast_narezki_ne_iz_ishodnika_lovitsya(tmp_path, bez_zamka):
    main, spec, _ = make_main(tmp_path)
    q = main / W.CHASTI["holdout_2025"].format(h=5)
    x = pd.read_parquet(q)
    x.loc[x.index[0], "rv_pred"] *= 1.0000001
    x.to_parquet(q, index=False)
    d, _ = W.chitat(main, 5, spec)
    with pytest.raises(SystemExit, match="гейт 3"):
        W.sverit_chasti(main, d, 5)


def test_run_celikom_versiya_i_vyhod(tmp_path, bez_zamka):
    main, spec, _ = make_main(tmp_path)
    out = tmp_path / "out"
    rep = W.run(main, out, spec)
    meta = json.loads((out / "wf_sigma_run.json").read_text())
    assert meta["v"] == 1 and meta["d1_ok"] is True
    assert meta["okno"] == "2017-01-02…2026-02-03"
    for h in W.HORIZONS:
        s = pd.read_parquet(out / f"sigma_hyb_h{h}_dev.parquet")
        assert list(s.columns) == ["date", "ticker", W.oc(h)] and s[W.oc(h)].dtype == np.float32
        assert not s.duplicated(["date", "ticker"]).any()
        assert meta[f"hyb_h{h}"]["stroki_do_2022_pobajtno_ch4_dev"] == int((s["date"] < "2022-01-01").sum())
    assert set(rep["_shagi"]["step"]) == set(range(1, 11))
    sums = (out / "SHA256SUMS").read_text()
    assert "wf_sigma_run.json" in sums and "sigma_hyb_h5_dev.parquet" in sums
