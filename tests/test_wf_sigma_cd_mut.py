"""Тестировщик PR #745: мутации уровня run() (проводка гейтов), которых нет в tests/test_wf_sigma_cd.py.
Там гейты проверены по отдельности; здесь — что run() их вызывает: убери вызов гейта из run() — тест упадёт.
Синтетика, реальные данные не читаются. python3 -m pytest tests/test_wf_sigma_cd_mut.py -q"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

_D = Path(__file__).resolve().parent


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


T = _load("twsc", _D / "test_wf_sigma_cd.py")
W = T.W


@pytest.fixture
def bez_zamka(monkeypatch):
    monkeypatch.setattr(W.period_v2, "proverit_chtenie", lambda *a, **k: None)


def _pereschitat_sha(main, spec, h):
    sob = yaml.safe_load(spec.read_text())["sobrano"]
    sob[f"R2/final/final_d0_s2emb_h{h}_test.parquet"] = T._sha(main / W.ISTOCHNIK.format(h=h))
    spec.write_text(yaml.safe_dump({"sobrano": sob}, allow_unicode=True))


def test_run_dubl_v_istochnike_lovitsya(tmp_path, bez_zamka):
    """Мутация: дубль дня на стыке шагов пропущен. SHA пересчитан под изменённый исходник — ловит только гейт склейки/шагов."""
    main, spec, _ = T.make_main(tmp_path)
    p = main / W.ISTOCHNIK.format(h=5)
    d = pd.read_parquet(p)
    pd.concat([d, d[d["date"] == "2021-12-31"]], ignore_index=True).to_parquet(p, index=False)
    _pereschitat_sha(main, spec, 5)
    with pytest.raises(SystemExit, match="дубл"):
        W.run(main, None, spec)


def test_run_dyra_v_kalendare_lovitsya(tmp_path, bez_zamka):
    """Мутация: вызов proverit_skleyku убран из run() — дыра/лишняя дата против календаря входов не ловится."""
    main, spec, kal = T.make_main(tmp_path)
    kal[~((kal["ticker"] == "AAA") & (kal["date"] == pd.Timestamp("2022-01-03")))].to_parquet(main / W.KAL, index=False)
    with pytest.raises(SystemExit, match="гейт 5"):
        W.run(main, None, spec)


def test_run_chast_narezki_lovitsya(tmp_path, bez_zamka):
    """Мутация: вызов sverit_chasti убран из run()."""
    main, spec, _ = T.make_main(tmp_path)
    q = main / W.CHASTI["t1_2022_2024"].format(h=1)
    x = pd.read_parquet(q)
    x.loc[x.index[3], "rv_pred"] *= 1.001
    x.to_parquet(q, index=False)
    with pytest.raises(SystemExit, match="гейт 3"):
        W.run(main, None, spec)


def test_run_dev_ne_pobajtno_lovitsya(tmp_path, bez_zamka):
    """Мутация: вызов proverit_dev убран из run() — ≤ 2021 разошёлся с ch4_dev."""
    main, spec, _ = T.make_main(tmp_path)
    c = main / W.CH4.format(h=5)
    x = pd.read_parquet(c)
    x.loc[x.index[10], W.oc(5)] *= 1.01
    x.to_parquet(c, index=False)
    with pytest.raises(SystemExit, match="гейт 4"):
        W.run(main, None, spec)


def test_run_utechka_ryada_lovitsya(tmp_path, bez_zamka, monkeypatch):
    """Мутация: вызов d1 убран из run(). Исходник сборки с утечкой +1 (σ_d построена из RV_{d+1}), согласованный с частями и ch4_dev."""
    orig = T._istochnik

    def s_utechkoj(h, kal, seed=5):
        d = orig(h, kal, seed).sort_values(["ticker", "date"])
        d["leg_raw"] = d.groupby("ticker")["leg_raw"].shift(-1).fillna(d["leg_raw"])
        d["rv_pred"] = d["leg_raw"] * d["F_sm"]
        return d.sort_values(["step", "date", "ticker"]).reset_index(drop=True)
    monkeypatch.setattr(T, "_istochnik", s_utechkoj)
    main, spec, _ = T.make_main(tmp_path)
    with pytest.raises(SystemExit, match="Д1"):
        W.run(main, None, spec)


def test_run_shag_t_plyus_odin_lovitsya(tmp_path, bez_zamka):
    """Мутация (а): для года t взят шаг t+1 (метки step и year подогнаны вместе, строки года 2019 — из шага 4)."""
    main, spec, _ = T.make_main(tmp_path)
    p = main / W.ISTOCHNIK.format(h=1)
    d = pd.read_parquet(p)
    m = d["date"].dt.year == 2019
    d.loc[m, ["step", "year"]] = (4, 2020)
    d.loc[m, "F_sm"] = d.loc[d["step"] == 4, "F_sm"].iloc[0]
    d.to_parquet(p, index=False)
    _pereschitat_sha(main, spec, 1)
    with pytest.raises(SystemExit, match="гейт 2"):
        W.run(main, None, spec)


def test_run_vyhod_polnyj_d1_18_strok(tmp_path, bez_zamka):
    """Мутация: в proverka_sdvig.csv попали не все периоды/ряды/контроли (3 периода × 3 ряда × 2 горизонта)."""
    main, spec, _ = T.make_main(tmp_path)
    W.run(main, tmp_path / "out", spec)
    t = pd.read_csv(tmp_path / "out" / "proverka_sdvig.csv")
    assert len(t) == 18 and set(t["rol"]) == {"ряд", "контроль"}
    assert t.groupby("rol").size().to_dict() == {"ряд": 6, "контроль": 12}


def test_vyhod_v_test_locked_zapreshchen(tmp_path, bez_zamka):
    main, spec, _ = T.make_main(tmp_path)
    with pytest.raises(SystemExit):
        W.run(main, main / "data/regen/test_locked/wf_sigma", spec)
