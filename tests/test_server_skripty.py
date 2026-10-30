"""Листы server-skripty (проверка цепочки X, 30.10): Д1 zabrat.sh берёт ledger 2.1 и 2.2 и файлы Морриса; Д2 pamyat.sh не ловит
чекпоинты прошлых проходов; Д3 e1_sektor не выдаёт FutureWarning (reindex(fill_value) вместо fillna(False)), результат побайтно прежний.
Мутации: условие только "2.1" — Д1 падает; ls любого checkpoint.json — Д2 падает; fillna(False) после reindex — Д3 падает."""
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import e1_sektor as es

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos"
rsync = pytest.mark.skipif(shutil.which("rsync") is None, reason="нет rsync")


def _server(src: Path, zabrat: Path = SH / "zabrat.sh"):
    for chain, v in (("train4_s1", "2.1"), ("train4_v6_s1", "2.2"), ("train4_v6_spravka", "2.2"), ("v2_s1", "2.0")):
        d = src / "ledger" / chain / "S1"
        (d / "p00" / "port").mkdir(parents=True)
        (d / "manifest.json").write_text('{\n "v": "%s"\n}\n' % v)
        (d / "checkpoint.json").write_text("{}")
        (d / "p00" / "port" / "00000.npy").write_text("x")
    m = src / "ledger" / "train4_morris" / "S1"
    m.mkdir(parents=True)
    for f in ("checkpoint.json", "zamorozka.yaml", "morris_y.jsonl", "morris_S1.csv", "lishnee.bin"):
        (m / f).write_text("x")
    for k in ("tables", "logs"):
        (src / k).mkdir()
        (src / k / "f.txt").write_text("x")


def _zabrat(script: Path, src: Path, dst: Path):
    return subprocess.run(["bash", str(script), str(src), str(dst)], capture_output=True, text=True)


@rsync
def test_d1_zabrat_beryot_21_22_i_morris(tmp_path):
    src, dst = tmp_path / "srv", tmp_path / "mak"
    _server(src)
    r = _zabrat(SH / "zabrat.sh", src, dst)
    assert r.returncode == 0, r.stdout + r.stderr
    for chain in ("train4_s1", "train4_v6_s1", "train4_v6_spravka"):
        assert (dst / "ledger" / chain / "S1/p00/port/00000.npy").exists(), chain
    assert not (dst / "ledger/v2_s1/S1/p00").exists()                            # 2.0 по-прежнему на сервере
    for f in ("checkpoint.json", "zamorozka.yaml", "morris_y.jsonl", "morris_S1.csv"):
        assert (dst / "ledger/train4_morris/S1" / f).exists(), f
    assert not (dst / "ledger/train4_morris/S1/lishnee.bin").exists()


@rsync
def test_d1_mutaciya_tolko_21_lovitsya(tmp_path):
    mut = tmp_path / "zabrat_mut.sh"
    mut.write_text((SH / "zabrat.sh").read_text().replace("grep -Eq '\"v\": \"2\\.(1|2)\"'", "grep -q '\"v\": \"2.1\"'"))
    assert mut.read_text() != (SH / "zabrat.sh").read_text()
    src, dst = tmp_path / "srv", tmp_path / "mak"
    _server(src)
    assert _zabrat(mut, src, dst).returncode == 0
    assert not (dst / "ledger/train4_v6_s1/S1/p00/port/00000.npy").exists()     # так выглядел дефект: тест Д1 на этом упал бы


def _zhdat(tmp: Path, pamyat: Path, pid_age: int) -> str:
    """Старый checkpoint (mtime 100 с назад), pid-файл старта pid_age с назад; ждём 2 с. Вернёт вывод pamyat_sledit-ожидания."""
    for f in ("obshchee.sh", "pamyat.sh"):
        (tmp / f).write_text((SH / f).read_text() if f != "pamyat.sh" else pamyat.read_text())
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    (tmp / "logs").mkdir()
    (tmp / "ledger/train4_s1/S1").mkdir(parents=True)
    ck, pf = tmp / "ledger/train4_s1/S1/checkpoint.json", tmp / "logs/sobol_train4.pid"
    ck.write_text("{}")
    pf.write_text("1")
    import time
    now = time.time()
    os.utime(ck, (now - 100, now - 100))
    os.utime(pf, (now - pid_age, now - pid_age))
    code = '. ./obshchee.sh; . ./pamyat.sh; pamyat_checkpoint_novyj train4 && echo NOVYJ || echo NET'
    return subprocess.run(["bash", "-c", code], cwd=tmp, capture_output=True, text=True).stdout.strip()


def test_d2_staryj_checkpoint_ignoriruetsya(tmp_path):
    assert _zhdat(tmp_path, SH / "pamyat.sh", pid_age=10) == "NET"               # checkpoint старше старта


def test_d2_novyj_checkpoint_schitaetsya(tmp_path):
    assert _zhdat(tmp_path, SH / "pamyat.sh", pid_age=200) == "NOVYJ"            # checkpoint новее старта


def test_d2_mutaciya_lyuboj_checkpoint_lovitsya(tmp_path):
    mut = tmp_path / "pamyat_mut.sh"
    mut.write_text((SH / "pamyat.sh").read_text().replace('[ ! -e "$pidf" ] || [ "$f" -nt "$pidf" ] && return 0', "return 0"))
    assert mut.read_text() != (SH / "pamyat.sh").read_text()
    d = tmp_path / "w"
    d.mkdir()
    assert _zhdat(d, mut, pid_age=10) == "NOVYJ"                                 # мутация принимает старый — тест выше упал бы


def test_d2_pamyat_sledit_ne_zhdyot_stary_chekpoint_na_linux(tmp_path):
    # ветка Linux: подставной free; наблюдатель с новым ожиданием не срабатывает на старый checkpoint за 2 с и уходит по таймауту
    (tmp_path / "bin").mkdir()
    fr = tmp_path / "bin/free"
    fr.write_text('#!/bin/sh\necho "Mem: 100 0 0 0 0 100"\n')
    fr.chmod(0o755)
    out = _zhdat(tmp_path, SH / "pamyat.sh", pid_age=10)
    assert out == "NET"
    r = subprocess.run(["bash", "-c", ". ./obshchee.sh; . ./pamyat.sh; pamyat_svobodno"], cwd=tmp_path, capture_output=True, text=True,
                       env={**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}"})
    assert r.stdout.strip() == "100"


def _ctx_i_daty():
    idx = pd.bdate_range("2018-01-01", periods=30)
    v = np.r_[np.arange(1.0, 16.0), np.arange(14.0, -1.0, -1.0)] + 10
    ctx = es.Ctx({"MOEXOG": pd.Series(v, index=idx)}, {}, {"LKOH": "MOEXOG"})
    return ctx, pd.bdate_range("2018-01-01", periods=40)       # даты шире индекса: reindex даёт пропуски (источник предупреждения)


def test_d3_net_future_warning_i_rezultat_pobaytno_prezhnij():
    ctx, dates = _ctx_i_daty()
    pt = dict(sektor_trend="po_trendu", shirina="net", ma_n=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        lo, sr = es.allow(pt, "LKOH", dates, ctx)
    dt = pd.DatetimeIndex(dates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        up, dn = (x.reindex(dt).fillna(False).to_numpy(bool) for x in ctx.trend("MOEXOG", 5))   # прежнее вычисление
    assert lo.tobytes() == up.tobytes() and sr.tobytes() == dn.tobytes()
    assert lo.dtype == bool and not lo[30:].any() and not sr[30:].any() and lo.any() and sr.any()


def test_d3_mutaciya_fillna_posle_reindex_lovitsya():
    ctx, dates = _ctx_i_daty()
    with pytest.warns(FutureWarning):
        for x in ctx.trend("MOEXOG", 5):
            x.reindex(pd.DatetimeIndex(dates)).fillna(False)
