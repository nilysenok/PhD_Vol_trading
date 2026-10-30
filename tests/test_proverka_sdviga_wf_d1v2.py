"""Д1 wf, лист v2: ставка по ряду счёта (csv < 2022 + parquet ≥ 2022) и вторая проверка σ̂ (частная корреляция). Синтетика."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import proverka_sdviga_wf as P  # noqa: E402
from fastbt import msp  # noqa: E402

DO = pd.Timestamp("2026-02-28")


def stavka_schet(smeny: dict) -> pd.Series:
    """Фальшивый msp.rate_cbr: csv владельца целиком (доли) плюс смены parquet ≥ 2022, которых в csv нет."""
    o = stavka_csv_mutant(DO).set_index("data_vstupleniya")["stavka_pct"] / 100.0
    p = pd.Series({pd.Timestamp(k): v for k, v in smeny.items()})
    return pd.concat([o, p]).sort_index()


def zased(tablica: pd.DataFrame, *dop: str) -> np.ndarray:
    z = list(stavka_csv_mutant(DO)["data_vstupleniya"]) + [pd.Timestamp(d) for d in dop]     # заседания = все даты csv (нулевой лаг)
    return np.array(sorted(z), dtype="datetime64[ns]")


def stavka_csv_mutant(do):          # мутация: ставка ≥ 2022 тоже из csv (parquet не виден)
    st = pd.read_csv(P.ROOT / "fastbt/rates/key_rate_cbr_vladelec.csv", parse_dates=["data_vstupleniya"], comment="#")
    return st[st["data_vstupleniya"] <= do]


def test_stavka_parquet_smena_bez_zasedaniya_lovitsya_a_csv_mutant_ee_ne_vidit(monkeypatch):
    monkeypatch.setattr(msp, "rate_cbr", lambda: stavka_schet({"2022-12-01": 0.30}))
    t = P.stavka_tablica(DO)
    assert pd.Timestamp("2022-12-01") in set(t["data_vstupleniya"])          # смена parquet попала в проверку
    z = zased(t)                                                             # заседания перед 01.12 нет
    assert not P.stavka_ok(t, z)
    assert P.stavka_ok(stavka_csv_mutant(DO), z)                             # мутация «из csv» — ложно зелёная


def test_stavka_parquet_smena_posle_zasedaniya_prohodit(monkeypatch):
    monkeypatch.setattr(msp, "rate_cbr", lambda: stavka_schet({"2022-12-05": 0.30}))
    t = P.stavka_tablica(DO)
    assert P.stavka_ok(t, zased(t, "2022-12-02"))


def test_stavka_do_2022_parquet_ne_chitaetsya(monkeypatch):
    def boom():
        raise AssertionError("parquet читать нельзя при --do < 2022")
    monkeypatch.setattr(msp, "rate_cbr", boom)
    assert P.stavka_tablica(pd.Timestamp("2021-12-31")).equals(stavka_csv_mutant(pd.Timestamp("2021-12-31")))


def seriya(n=900, m=6, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    y = pd.DataFrame(rng.normal(size=(n, m)), index=idx, columns=list("ABCDEF"))
    return y, rng


def test_sigma_s_utechkoj_plus_odin_vtoraya_krasnaya():
    y, rng = seriya()
    s = y.shift(-1) + 0.3 * rng.normal(size=y.shape)             # σ̂_d знает RV_{d+1}
    v = P.vtoraya_sigma(s, y)
    assert not v["ok2"] and abs(v["pcorr_fakt"]) > 0.3


def test_sigma_sglazhennyj_bez_utechki_argmax_km1_vtoraya_zelenaya():
    y, rng = seriya()
    s = 0.15 * y + 0.6 * y.shift(1) + 0.4 * y.shift(2) + 0.2 * y.shift(3) + 0.05 * rng.normal(size=y.shape)   # пик весов на d−1
    prof = P.profil(s.diff(), y.diff())
    assert max(prof, key=lambda k: abs(prof[k])) == -1            # первая проверка красная: argmax на k=−1
    v = P.vtoraya_sigma(s, y)
    assert v["ok2"] and abs(v["pcorr_fakt"]) < 0.5 * abs(v["pcorr_kontrol"])


def kod(monkeypatch, tmp_path, ryad, ok, ok2):
    t = pd.DataFrame([{**P.stroka(ryad, "t", "c", {-2: 0.1, -1: 0.3, 0: 0.9 if ok else 0.2, 1: 0.3}), "ok2": ok2}])
    monkeypatch.setattr(P, "OUT", tmp_path / "t.csv")
    monkeypatch.setattr(P, "shapka", lambda *a: "# s\n")
    monkeypatch.setattr(P, "sobrat", lambda m, d: t)
    return P.main(["--do", "2021-12-31", "--main", str(tmp_path)]), (tmp_path / "t.csv").read_text()


def test_kod_vyhoda_sigma_utechka_krasnaya(monkeypatch, tmp_path):
    assert kod(monkeypatch, tmp_path, "sigma_h5", False, False)[0] != 0


def test_kod_vyhoda_sigma_sglazhennyj_zelenyj_s_otmetkoj(monkeypatch, tmp_path):
    k, csv = kod(monkeypatch, tmp_path, "sigma_h22", False, True)
    assert k == 0 and "Д1 через вторую проверку" in csv


def test_kod_vyhoda_ne_sigma_ok2_ne_spasaet(monkeypatch, tmp_path):
    assert kod(monkeypatch, tmp_path, "atr14", False, True)[0] != 0


def test_pustoj_ryad_vtoraya_ne_prohodit():
    y, _ = seriya(n=10)
    assert not P.vtoraya_sigma(y, y)["ok2"]
