"""Реверс: TEST_C и D (s9) на period="test1" читают σ̂ зоны Z2 (dev ≤ 2021 + final_h*__t1). Синтетика во временном каталоге.
Мутации: σ̂ сдвинута на +1 день; чтение без отсечки 2025. python3 -m pytest tests/test_s9_test1_cd.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import data, s9, sigma_z2, zona

DEV_DATES = pd.to_datetime(["2021-12-29", "2021-12-30"])
Z2_DATES = pd.to_datetime(["2022-01-04", "2022-01-05", "2023-06-01", "2024-12-30", "2024-12-31"])


def _rv(i, h):
    return 0.01 * (i + 1) * h          # разные для каждой даты и горизонта: сдвиг на день различим


@pytest.fixture
def z2(tmp_path, monkeypatch):
    dk = tmp_path / "d"
    dk.mkdir()
    for h in (1, 5):
        pd.DataFrame({"date": DEV_DATES, "ticker": "SBER", f"sigma_pred_h{h}": np.float32(0.1 * h)}
                     ).to_parquet(sigma_z2.dev_path(h, dk))
    kat = tmp_path / "z"
    kat.mkdir()
    for h in (1, 5):
        pd.DataFrame({"date": Z2_DATES, "ticker": "SBER", "rv_pred": [_rv(i, h) for i in range(len(Z2_DATES))],
                      "design": sigma_z2.DESIGN[h]}).to_parquet(sigma_z2.z2_path(h, kat))
    monkeypatch.setenv("FASTBT_DEV_SIGMA", str(dk))
    monkeypatch.setenv("FASTBT_Z2_R2FINAL", str(kat))
    data.vpred_df.cache_clear()
    old = s9.PERIOD, s9.YEARS, s9.VAL_START
    s9.set_period("test1")
    yield kat
    s9._restore_period(*old)
    data.vpred_df.cache_clear()


def _proverka(df, label="hyb_h5", h=5):
    """σ̂ на дату d (из Z2) отдаётся на баре d, не на d−1 и не на d+1; даты 2022–2024."""
    d = df[df["ticker"] == "SBER"]
    a = s9.align(Z2_DATES.to_numpy(), d)
    assert a == pytest.approx([np.sqrt(_rv(i, h)) for i in range(len(Z2_DATES))], rel=1e-6)
    assert d["date"].max() <= pd.Timestamp("2024-12-31")


@pytest.mark.parametrize("label,h", [("hyb_h5", 5), ("hyb_h1", 1)])     # TEST_C = hyb_h5; D — h1 и h5
def test_c_d_chitayut_sigma_z2(z2, label, h):
    assert s9.need_sigma("x") is None and s9.TEST_C[1] in s9.TEST1_LABELS
    df = s9.load_source(label)
    assert df["date"].min() == DEV_DATES[0] and df["date"].max() == pd.Timestamp("2024-12-31")
    assert (df["date"] >= pd.Timestamp("2022-01-01")).sum() == len(Z2_DATES)          # 2022–2024 из Z2
    _proverka(df, label, h)


def test_nelzya_drugie_istochniki_na_test1(z2):
    with pytest.raises(NotImplementedError):
        s9.load_source("naive_h1")


def test_align_posle_2024_nan(z2):
    df = s9.load_source("hyb_h5")
    a = s9.align(np.array(pd.to_datetime(["2024-12-31", "2025-01-02"]), dtype="datetime64[ns]"), df)
    assert not np.isnan(a[0]) and np.isnan(a[1])


def test_umolchanie_train_ne_menyaetsya(z2):
    s9.set_period("train")
    assert s9._cut() == s9.CUT == pd.Timestamp("2021-12-31")
    s9.set_period("train4")
    with pytest.raises(NotImplementedError):
        s9.load_source("hyb_h5")


def test_build_cache_test1_otdayot_sigma_z2(z2, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(s9, "_job", lambda a: seen.append(a) or (a[0], {}))
    monkeypatch.setattr(s9, "OUT", tmp_path)
    monkeypatch.setattr(s9, "cache_key", lambda: "synt")
    s9.build_cache(workers=1)
    assert seen and set(seen[0][1]) == {"hyb_h1", "hyb_h5"} and seen[0][3] == "test1"
    assert seen[0][1]["hyb_h5"][0] == "h5"


def test_mutaciya_sigma_plus_odin_den_lovitsya(z2, monkeypatch):
    """Мутация: σ̂ d+1 на баре d (ряд сдвинут на день назад) — проверка обязана падать."""
    _proverka(s9.load_source("hyb_h5"))
    real = s9.load_source

    def mut(label):
        d = real(label).copy()
        d["sigma"] = d.groupby("ticker")["sigma"].shift(-1)
        return d.dropna(subset=["sigma"])
    with pytest.raises(AssertionError):
        _proverka(mut("hyb_h5"))


def test_otkaz_2025_i_mutaciya_bez_otsechki(z2, monkeypatch):
    kat = z2
    for h in (1, 5):
        pd.DataFrame({"date": pd.to_datetime(["2024-12-31", "2025-01-02"]), "ticker": "SBER", "rv_pred": 0.04}
                     ).to_parquet(sigma_z2.z2_path(h, kat))
    data.vpred_df.cache_clear()
    with pytest.raises(PermissionError):
        s9.load_source("hyb_h5")                 # отсечка 2025 стоит
    monkeypatch.setattr(zona, "check", lambda *a, **k: None)      # мутация: проверки нет
    monkeypatch.setattr(s9, "_cut", lambda: pd.Timestamp("2100-01-01"))
    df = s9.load_source("hyb_h5")
    with pytest.raises(AssertionError):          # ловится проверкой границы зоны: данные ≥ 2025 дошли бы до счёта
        assert df["date"].max() < pd.Timestamp("2025-01-01")


def _klyuch(monkeypatch, tmp_path):
    """cache_key на test1 с подменой тяжёлых общих входов (бары, SOURCES) на крошечный файл: меняется только σ̂."""
    f = tmp_path / "x.bin"
    f.write_bytes(b"x")
    monkeypatch.setattr(zona, "path", lambda kind, p=None: f)
    monkeypatch.setattr(s9, "SOURCES", {"hyb_h1": (f, "sigma_pred_h1", "h1")})
    monkeypatch.setattr(s9, "MAIN_YAML", f)
    return s9.cache_key()


def _podmena(p, col):
    d = pd.read_parquet(p)
    d.iloc[0, d.columns.get_loc(col)] += 0.001
    d.to_parquet(p)


@pytest.mark.parametrize("chast", ["dev", "z2"])
def test_klyuch_test1_menyaetsya_pri_podmene_dev_ili_z2(z2, monkeypatch, tmp_path, chast):
    """Д5 (mutations № 7): подмена dev-файла или Z2-файла при прочих равных → другой ключ."""
    k0 = _klyuch(monkeypatch, tmp_path)
    assert _klyuch(monkeypatch, tmp_path) == k0
    if chast == "dev":
        _podmena(sigma_z2.dev_path(5), "sigma_pred_h5")
    else:
        _podmena(sigma_z2.z2_path(1), "rv_pred")
    assert _klyuch(monkeypatch, tmp_path) != k0


def test_mutaciya_dev_ne_v_klyuche_lovitsya(z2, monkeypatch, tmp_path):
    """Мутация: digest считает только Z2-файлы (dev не в ключе) — проверка «подмена dev меняет ключ» обязана падать."""
    def tolko_z2():
        return b"".join(sigma_z2.z2_path(h).read_bytes() for h in sigma_z2.HORIZONS).hex()[:40]
    monkeypatch.setattr(sigma_z2, "digest", tolko_z2)
    k0 = _klyuch(monkeypatch, tmp_path)
    _podmena(sigma_z2.dev_path(5), "sigma_pred_h5")
    with pytest.raises(AssertionError):
        assert _klyuch(monkeypatch, tmp_path) != k0


def test_load_source_sam_otkazyvaet_2025(z2, monkeypatch):
    """Мутация «удалён zona.check в load_source» падает здесь: без своей проверки источник с датой 2025 молча отрезался бы _cut."""
    bad = pd.DataFrame({"date": pd.to_datetime(["2024-12-31", "2025-01-02"]), "ticker": "SBER",
                        "sigma_pred_h1": 0.1, "sigma_pred_h5": 0.1})
    fake = lambda: bad                  # noqa: E731
    fake.cache_clear = lambda: None
    monkeypatch.setattr(data, "vpred_df", fake)
    with pytest.raises(PermissionError):
        s9.load_source("hyb_h5")
