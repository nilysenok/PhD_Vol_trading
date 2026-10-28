"""σ̂ главы 4 зоны Z2 (scripts/dannye/z2_sigma_ch4.py): синтетика, реальные данные не читаются.
python3 -m pytest tests/test_z2_sigma_ch4.py -q"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("zs", Path(__file__).resolve().parents[1] / "scripts" / "dannye" / "z2_sigma_ch4.py")
zs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zs)

DAYS = pd.bdate_range("2021-11-01", "2022-03-31")
TICKERS = ("AAA", "BBB", "CCC")


def _frame(rng):
    rv = pd.DataFrame([(d, t) for t in TICKERS for d in DAYS], columns=["date", "ticker"])
    rv["rv"] = np.exp(rng.normal(size=len(rv)))                      # iid: сдвиг на день разрушает корреляцию
    rv["noise"] = np.exp(0.1 * rng.normal(size=len(rv)))
    return rv.sort_values(["ticker", "date"]).reset_index(drop=True)


def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def make_main(tmp, drop_day=None, drop_bar=None, bad_rv=False):
    """Мини-«основная копия»: ch4_dev (≤ 2021), части Z2 (2022+), rv, календарь, дневной ряд Z2. -> (main, сырые столбцы частей Z2)."""
    rng, main, raws = np.random.default_rng(7), tmp / "main", {}
    f = _frame(rng)
    f["pred"] = f["rv"] * f["noise"]
    cut = pd.Timestamp("2022-01-01")
    for name, (part, col, oc) in zs.MODELS.items():
        raw = f[["date", "ticker"]].assign(**{col: f["pred"].values})
        sig = zs.H.sigma_ch3(raw[raw["date"] < cut], col, oc)
        _write(sig, main / "data/regen/ch4_dev" / f"{name}.parquet")
        z = raw[raw["date"] >= cut]
        if drop_day is not None:
            z = z[z["date"] != drop_day]
        raws[name] = z
        _write(z, main / zs.Z2R2 / f"{part}.parquet")
    rv_dev = f[f["date"] < cut].sort_values(["ticker", "date"])
    _write(zs.H.naive_from_rv(rv_dev, cut=pd.Timestamp("2021-12-31")), main / "data/regen/ch4_dev/sigma_naive_dev.parquet")
    rvf = f[["date", "ticker", "rv"]].copy()
    if bad_rv:
        rvf.loc[rvf["date"] == pd.Timestamp("2021-12-15"), "rv"] *= 3     # naive разошлась с ch4_dev
    _write(rvf, main / "data/iss_check/rv_1850_noquar.parquet")
    (main / "data/calendar").mkdir(parents=True)
    pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in DAYS]}).to_csv(main / "data/calendar/torgovye_dni_moex.csv", index=False)
    bars = [d for d in DAYS if d != drop_bar]
    _write(pd.DataFrame({"date": bars, "ticker": "AAA"}), main / "data/regen/test1_2022_2024/daily/ohlcv_daily_1850.parquet")
    return main, raws


def check_values(out, main, raws):
    """Инвариант: ≤ 2021 побайтно = ch4_dev; 2022+ = √сырого ряда части Z2 (float32); сортировка (date, ticker)."""
    for name, (_p, col, oc) in zs.MODELS.items():
        new = pd.read_parquet(out / f"{name}.parquet")
        dev = pd.read_parquet(main / "data/regen/ch4_dev" / f"{name}.parquet")
        old = new[new["date"] < "2022-01-01"]
        assert np.array_equal(old[oc].values, dev[oc].values) and new["date"].is_monotonic_increasing
        z = new[new["date"] >= "2022-01-01"].reset_index(drop=True)
        want = raws[name].sort_values(["date", "ticker"]).reset_index(drop=True)
        assert z[oc].dtype == np.float32
        assert np.array_equal(z[oc].values, np.sqrt(want[col].values).astype("float32"))


def test_ravenstvo_dev_i_znacheniya_z2(tmp_path):
    main, raws = make_main(tmp_path)
    out = tmp_path / "out"
    rep = zs.run(main, out)
    check_values(out, main, raws)
    assert all(r["gate1_maxdiff"] == 0.0 and r["sdvig"]["ok"] and r["sdvig"]["argmax"] == 0 for r in rep.values())
    assert (out / "SHA256SUMS").read_text().count("\n") == 5
    assert all(pd.read_parquet(p)["date"].max() < pd.Timestamp("2025-01-01") for p in out.glob("*.parquet"))


def test_otkaz_pri_date_2025(tmp_path):
    main, _ = make_main(tmp_path)
    p = main / zs.Z2R2 / "final_d0_s2emb_h1__t1.parquet"
    z = pd.read_parquet(p)
    bad = pd.concat([z, z.iloc[[0]].assign(date=pd.Timestamp("2025-01-09"))], ignore_index=True)
    _write(bad, p)
    with pytest.raises(SystemExit, match="2025"):
        zs.run(main, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_mutaciya_bez_sqrt_lovitsya(tmp_path, monkeypatch):
    main, raws = make_main(tmp_path)
    monkeypatch.setattr(zs.H, "sigma_ch3", lambda df, col, oc: df.assign(**{oc: df[col].values.astype("float32")})
                        .sort_values(["date", "ticker"])[["date", "ticker", oc]].reset_index(drop=True))
    out = tmp_path / "out"
    zs.run(main, out)
    with pytest.raises(AssertionError):
        check_values(out, main, raws)


def test_mutaciya_sdvig_plus_odin_lovitsya(tmp_path, monkeypatch):
    main, _ = make_main(tmp_path)

    def leak(df, col, oc):                                              # σ_d := σ_{d+1}
        d = df.sort_values(["ticker", "date"]).copy()
        d[oc] = np.sqrt(d.groupby("ticker")[col].shift(-1)).astype("float32")
        return d.dropna(subset=[oc]).sort_values(["date", "ticker"])[["date", "ticker", oc]].reset_index(drop=True)
    monkeypatch.setattr(zs.H, "sigma_ch3", leak)
    with pytest.raises(SystemExit, match="сдвиг"):
        zs.run(main, tmp_path / "out")


def test_gate1_naive_rashoditsya_s_dev(tmp_path):
    main, _ = make_main(tmp_path, bad_rv=True)
    with pytest.raises(SystemExit, match="гейт 1"):
        zs.run(main, tmp_path / "out")


def test_dyra_v_kalendare_lovitsya_i_den_bez_torgov_net(tmp_path):
    day = pd.Timestamp("2022-02-10")
    main, _ = make_main(tmp_path, drop_day=day)
    with pytest.raises(SystemExit, match="гейт 3"):
        zs.run(main, tmp_path / "out")
    main2, _ = make_main(tmp_path / "b", drop_day=day, drop_bar=day)     # дня нет и в дневном ряде: не дыра, а календарь шире торгов
    rep = zs.run(main2, None)
    assert all(r["bez_torgov"] == ["2022-02-10"] for r in rep.values())
