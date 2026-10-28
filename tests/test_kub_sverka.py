"""Куб прогона = эталонная сетка Ж-047: оси signal, значения, общие оси. Мутация «S3 ↔ S4» обязана падать."""
from pathlib import Path

import pytest
import yaml

import fastbt.progon_sobol as ps

D = Path(__file__).resolve().parents[1] / "fastbt/designs"
SIDS = [f"S{k}" for k in range(1, 7)]


def _y(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def kub(sid: str) -> dict:
    return _y(D / f"progon/kub_{sid}.yaml")


def etalon(sid: str) -> dict:
    return _y(D / f"sweep/zh047_{sid}.yaml")


def rashozhdeniya(k: dict, z: dict) -> list:
    """Список расхождений: signal (имена, порядок, ishodnoe, setka) и общие оси (setka, yakor; sobytiya → filtr)."""
    out = []
    if list(k["signal"]) != list(z["signal"]):
        out.append(f"оси signal: {list(k['signal'])} ≠ {list(z['signal'])}")
    for n in k["signal"].keys() & z["signal"].keys():
        for f in ("ishodnoe", "setka"):
            if k["signal"][n][f] != z["signal"][n][f]:
                out.append(f"signal.{n}.{f}")
    for n, v in z["obshchie"].items():
        if n == "sobytiya":
            continue
        if n not in k["obshchie"]:
            out.append(f"obshchie.{n}: нет в кубе")
            continue
        for f in ("setka", "yakor"):
            if k["obshchie"][n][f] != v[f]:
                out.append(f"obshchie.{n}.{f}")
    return out


@pytest.mark.parametrize("sid", SIDS)
def test_kub_raven_etalonu(sid):
    assert rashozhdeniya(kub(sid), etalon(sid)) == []


@pytest.mark.parametrize("sid", SIDS)
def test_sid_shapki_raven_obekt_sid(sid):
    assert kub(sid)["sid"] == etalon(sid)["obekt"]["sid"] == sid


@pytest.mark.parametrize("sid", SIDS)
def test_proverit_kub_prinimaet_osi_semyi(sid):
    ps.proverit_kub(kub(sid))


def test_mutaciya_s3_s4_perepytany_lovitsya():
    assert rashozhdeniya(kub("S3"), etalon("S4"))
    assert rashozhdeniya(kub("S4"), etalon("S3"))
    k = kub("S3")
    k["signal"] = kub("S4")["signal"]                        # куб S3 с осями Supertrend (старая ошибка)
    assert rashozhdeniya(k, etalon("S3"))
    with pytest.raises(ValueError, match="оси"):
        ps.proverit_kub(k)
