"""Прогрев Морриса v7 и пилот так, как их запускает сервер (scripts/server/perenos/zapusk_sobol.sh, PERENOS_KUB=v7 PERENOS_PILOT=1),
но на периоде train4 (данных 2022+ на Маке нет). Сухой режим скрипта этого не ловил: progon_morris_v7 подменяет canon_v6 на canon_v7,
а у того нет функции canon_v6 (AttributeError в плане Морриса). Сети нет.

Граница train4: IMOEX на нём читается < 2020-01-01, а ряды оценщика идут до 2021, поэтому у S1–S4 точки с otnositelno ≠ net
на train4 дают «фон: … NaN» (на polnaya границы чтения и рядов согласованы — не проверено на Маке). Тест S1 допускает только эту ошибку."""
import json

import pytest

import fastbt.canon_v7b as c7b
import fastbt.fast_etap1 as fe
import fastbt.progon_morris_v6 as pm
import fastbt.progon_morris_v7 as pm7
import fastbt.progon_sobol as ps
import fastbt.progon_v7 as p7

SIDS = ["S1", "S8", "S10"]


def _dannye():
    from fastbt import daily_1850 as d18
    return (d18.default_main() / "moex_discovery/data/raw/candles_10m/SBER.parquet").exists()


realno = pytest.mark.skipif(not _dannye(), reason="нет данных train4 (сервер/Мак без копии)")


def _plan(sid):
    with pm7.kanon_v7():
        kub, ox = pm.setka(sid)
        return kub, ox, pm.plan(sid, kub, ox, 1, 10, int(kub["sobol"]["morris_seed"]))


@pytest.mark.parametrize("sid", SIDS)
def test_plan_morris_pod_kanonom_v7(sid):
    kub, ox, (nom, por, pts) = _plan(sid)
    assert pts and all(isinstance(p, dict) for p in pts)
    assert {n for n, _, _ in ox} >= {"k_okup"}                     # ось v7 открыта в Моррисе (GOTOVY_V7)


def test_kanon_semya_s10_cherez_canon_v7b():
    kub, ox, (_, _, pts) = _plan("S10")
    with pm7.kanon_v7():
        for p in pts[:5]:
            assert pm.canon_v6.canon_v6("S10", p, kub) == c7b.canon_v7b("S10", p, kub)


def test_podmena_snimaetsya_i_spisok_osej():
    staryj, so = pm.canon_v6, pm.spisok_osej
    with pm7.kanon_v7():
        assert pm.canon_v6 is pm7.Kanon7 and pm.spisok_osej is pm7.spisok_osej7
    assert pm.canon_v6 is staryj and pm.spisok_osej is so


def test_vhod_T_ne_idyot_v_bystryj_sloj():
    """Быстрый run не знает cena_T: точка канона v7 с vhod = T (ключ `vhod`, не `vhod_T`) обязана идти эталоном."""
    assert fe.pokryto({}) and not fe.pokryto({"vhod": "T", "T": "18:00"})


@realno
@pytest.mark.parametrize("sid", ["S8", "S10"])
def test_progrev_morris_v7_realno(sid, tmp_path, capsys):
    assert pm7.main(["--kub", "v7", "--period", "train4", "--out", str(tmp_path), "--procs", "1", "--r", "1", "--strategii", sid]) == 0
    s = capsys.readouterr().out
    assert "'oshibka': 0" in s and "'ok': 0" not in s, s


@realno
def test_progrev_morris_v7_s1_tolko_dev_granica(tmp_path, capsys):
    assert pm7.main(["--kub", "v7", "--period", "train4", "--out", str(tmp_path), "--procs", "1", "--r", "1", "--strategii", "S1"]) == 0
    oshibki = [ln for ln in capsys.readouterr().out.splitlines() if "ошибка ×" in ln]
    assert all("фон:" in ln for ln in oshibki), oshibki            # других ошибок оценщика у S1 нет


@realno
@pytest.mark.parametrize("sid", ["S1", "S8", "S10"])
def test_pilot_realno(sid, tmp_path):
    assert p7.main(["--kub", "v7", "--period", "train4", "--pilot", "--n1", "2", "--semeystva", sid, "--procs", "1", "--out", str(tmp_path), "--fast"]) == 0
    z = json.loads((tmp_path / p7.ZAMER).read_text(encoding="utf-8"))[sid]
    assert z["sboy"] is None and z["n"] >= 1
    st = [json.loads(x)["status"] for p in sorted((tmp_path / sid).glob("p[0-9]*/rows.jsonl")) for x in p.read_text(encoding="utf-8").splitlines()]
    assert st and set(st) == {"ok"}, st
