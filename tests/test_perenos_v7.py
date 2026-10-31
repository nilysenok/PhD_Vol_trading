"""Перенос куба v7 (V7-8b): (1) манифест polnaya (--do 2026-02-28) принимает всё, что читают источники v7 (10-минутки 17 тикеров
2014–2026, IMOEX/секторные индексы, календарь, корпсобытия, свод Д8), и отвергает данные позже границы; (2) `zapusk_sobol.sh polnaya`
с PERENOS_KUB=v7 формирует команды progon_morris_v7 / progon_v7 (сухой прогон на заглушке Python). Мутации: v7 уходит в progon_server;
polnaya без v7 / v7 на train4 принят; прогрев без Морриса v7 — тесты падают. Синтетика, без сети."""
import fnmatch
import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fastbt.data import TICKERS

ROOT = Path(__file__).resolve().parents[1]
SRV, SH = ROOT / "scripts/server", ROOT / "scripts/server/perenos"
sys.path.insert(0, str(SRV))


def _mod(name):
    s = importlib.util.spec_from_file_location(name, SRV / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


MP, UP = _mod("manifest_perenos"), _mod("usech_pravila")
DO = pd.Timestamp("2026-02-28")
CAND, EXT = "moex_discovery/data/raw/candles_10m", "moex_discovery/data/final_v6/candles_10m/external"
CORP, KAL = "data/corporate_actions.csv", "data/calendar/torgovye_dni_moex.csv"


def _pq(p: Path, kol: str, daty):
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({kol: pd.to_datetime(daty), "close": 1.0}).to_parquet(p)


def _koren(r: Path, hvost=False) -> Path:
    for t in TICKERS:
        _pq(r / CAND / f"{t}.parquet", "timestamp", ["2014-06-02 10:00", "2026-02-27 18:40"])
    _pq(r / CAND / "ZZZ.parquet", "timestamp", ["2014-06-02 10:00", "2026-03-02 10:00" if hvost else "2026-02-27 18:40"])
    _pq(r / EXT / "IMOEX.parquet", "begin", ["2014-06-02 10:00", "2026-02-27 18:40"])
    (r / CORP).parent.mkdir(parents=True, exist_ok=True)
    (r / CORP).write_text("ticker,date,factor,status\nSBER,2024-07-01,1.0,applied\n")
    (r / KAL).parent.mkdir(parents=True, exist_ok=True)
    (r / KAL).write_text("date\n2014-06-02\n2026-02-27\n")
    return r


def _prinyato(r: Path):
    res = MP.sobrat(str(r), MP.razobrat(SRV / "perenos_spisok.tsv"), DO)
    return {x[0] for x in res["prinyato"]}, " ".join(str(x) for x in res["otvergnuto"])


def test_manifest_v7_10minutki_vseh_tikerov_imoex_kalendar_korpsobytiya(tmp_path):
    p, otv = _prinyato(_koren(tmp_path))
    assert len(TICKERS) == 17 and all(f"{CAND}/{t}.parquet" in p for t in TICKERS), otv
    assert f"{EXT}/IMOEX.parquet" in p and CORP in p and KAL in p, otv


def test_manifest_v7_dannye_posle_granicy_otvergayutsya(tmp_path):
    p, otv = _prinyato(_koren(tmp_path, hvost=True))
    assert f"{CAND}/ZZZ.parquet" not in p and "2026-03-02" in otv


def test_pravila_usecheniya_pokryvayut_chitaemoe_istochnikami_v7():
    """Файлы docs, которые читает цепочка источников v7 (сектора, свод Д8, дивиденды, календарь MOEX 2025–2026), есть в правилах усечения."""
    shabl = [x[0] for x in UP.PRAVILA]
    for need in ("docs/issledovaniya/2026-10-27_sektory.csv", "docs/issledovaniya/div_sverka_svod.csv",
                 "docs/issledovaniya/2026-10-27_dividendy_2017_2021.csv", "docs/issledovaniya/2026-10-28_dividendy_2015_2016.csv",
                 "docs/issledovaniya/2026-10-30_kalendar_moex_2025_2026.csv", "data/corporate_actions.csv"):
        assert any(fnmatch.fnmatch(need, s) for s in shabl), need
    assert (ROOT / "docs/issledovaniya/2026-10-27_sektory.csv").exists()


STUB = """#!/usr/bin/env bash
if [ "$*" = "-m fastbt.progon_server --help" ]; then echo "--k3"; exit 0; fi
echo "$*" >> "$STUB_LOG"
exit 0
"""


def _pakyet(tmp: Path, progon_v7=True) -> Path:
    for f in ("obshchee.sh", "pamyat.sh", "zapusk_sobol.sh"):
        (tmp / f).write_text((SH / f).read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\nkod_slit=0\n")
    (tmp / "kod/ff/fastbt").mkdir(parents=True)
    (tmp / "kod/ff/fastbt/progon_server.py").write_text("")
    if progon_v7:
        for m in ("progon_v7", "progon_morris_v7"):
            (tmp / f"kod/ff/fastbt/{m}.py").write_text("")
    (tmp / ".venv/bin").mkdir(parents=True)
    py = tmp / ".venv/bin/python3"
    py.write_text(STUB)
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    return tmp


def _ryad(tmp: Path, per="polnaya", kub="v7", **env):
    log = tmp / "calls.log"
    e = {**os.environ, "STUB_LOG": str(log), "PERENOS_YADRA": "132", "PERENOS_KUB": kub, **env}
    r = subprocess.run(["bash", "zapusk_sobol.sh", "__ryad", per], cwd=tmp, env=e, capture_output=True, text=True)
    return r, (log.read_text().splitlines() if log.exists() else [])


def test_zapusk_v7_komandy_progrev_morris_prohody(tmp_path):
    r, calls = _ryad(_pakyet(tmp_path), PERENOS_PROHODY="s1 s2")
    assert r.returncode == 0, r.stdout + r.stderr
    joined = "\n".join(calls)
    assert "progon_server" not in joined and "progon_morris_v6" not in joined                  # v7 не уходит в прежние драйверы
    pg, mo, s1, y1, s2, y2 = calls
    assert pg.startswith("-m fastbt.progon_morris_v7 --kub v7 --period polnaya") and "progrev_polnaya" in pg and "--r 1" in pg
    assert "-m fastbt.progon_morris_v7 --kub v7 --period polnaya" in mo and "ledger/polnaya_morris" in mo and "--prodolzhit" in mo
    assert "--procs 80" in mo                                                                    # polnaya = wf по процессам (Ж-136)
    for s, seed in ((s1, 1), (s2, 2)):
        assert s.startswith("-m fastbt.progon_v7 --kub v7 --period polnaya") and f"--seed {seed}" in s
        assert "--n1 200000" in s and "--prodolzhit" in s and "--zamorozka" in s and "ledger/polnaya_morris" in s
        assert "--fast" in s and "--procs 80" in s
    assert "ledger/polnaya_v7_s1" in s1 and "ledger/polnaya_v7_s2" in s2
    assert "svodka_ledger.py --yadro" in y1 and "svodka_ledger.py --yadro" in y2


def test_zapusk_v7_umolchanie_prohodov_s1_s2_bez_k3_i_spravki(tmp_path):
    r, calls = _ryad(_pakyet(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    pr = [c for c in calls if "fastbt.progon_v7" in c]
    assert [("--seed 1" in c, "--seed 2" in c) for c in pr] == [(True, False), (False, True)]
    assert not any("--k3" in c or "spravka" in c for c in calls)       # к3 v7 — ссылка на ledger v6, не пересчёт


def test_zapusk_v7_n1_i_strategii_iz_okruzheniya(tmp_path):
    _, calls = _ryad(_pakyet(tmp_path), PERENOS_N1="1000", PERENOS_STRATEGII="S8,S9", PERENOS_PROHODY="s1", PERENOS_FAST="0")
    s1 = [c for c in calls if "fastbt.progon_v7" in c][0]
    assert "--n1 1000" in s1 and "--strategii S8,S9" in s1 and "--fast" not in s1


def test_v7_bez_progon_v7_v_kode_ne_idyot_v_progon_server(tmp_path):
    r, calls = _ryad(_pakyet(tmp_path, progon_v7=False))
    assert r.returncode != 0 and "progon_v7" in r.stdout and not any("progon_server" in c for c in calls)


def test_v5_i_v6_ne_zatronuty_kubom_v7(tmp_path):
    r, calls = _ryad(_pakyet(tmp_path), per="train4", kub="v5", PERENOS_PROHODY="s1")
    assert r.returncode == 0 and not any("v7" in c for c in calls) and "--kub v5" in calls[0]


def _vhod(tmp: Path, per: str, kub: str):
    e = {**os.environ, "PERENOS_YADRA": "132"}
    if kub:
        e["PERENOS_KUB"] = kub
    return subprocess.run(["bash", "zapusk_sobol.sh", per], cwd=tmp, env=e, capture_output=True, text=True)


def test_period_polnaya_tolko_dlya_kuba_v7_i_v7_tolko_na_polnaya(tmp_path):
    t = _pakyet(tmp_path)
    for per, kub in (("polnaya", ""), ("polnaya", "v6"), ("train4", "v7"), ("wf", "v7"), ("v2", "v7")):
        r = _vhod(t, per, kub)
        assert r.returncode != 0 and "FAIL" in r.stdout and "polnaya" in r.stdout, (per, kub, r.stdout)


def test_polnaya_trebuet_otmetku_zamka_kak_wf():
    """polnaya читает данные ≥ 2022: в точке запуска проверяется отметка владельца (otmetka_wf), как у wf; процессы — как у wf."""
    t = (SH / "zapusk_sobol.sh").read_text()
    assert 'if [ "$per" = wf ] || [ "$per" = polnaya ]' in t and 'processov "$pk"' in t
