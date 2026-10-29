"""Б1: гонка записи manifest.json (Pisatel.__init__) — процессы первых кусков одной стратегии стартуют разом.
Мутация: вернуть mp.write_text вместо _zapis_atomarno — test_gonka_n_processov_ne_chitaet_poluptoy роняется."""
import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import pytest

from fastbt import ledger_v2 as L

BOLSHOY = {"T": 6, "kod": "K1", "vhody": "V1", "nbars": list(range(20000))}   # большой файл — окно гонки шире


def _star(root, potok, bar, q):
    try:
        bar.wait(10)
        time.sleep(potok * 0.00015)        # разнос старта: поздний процесс застаёт файл посреди записи
        L.Pisatel(Path(root), "S1", potok, dict(BOLSHOY))
        q.put(None)
    except Exception as e:  # noqa: BLE001 — отчёт о типе ошибки родителю
        q.put(f"{type(e).__name__}: {e}")


def _povtor(root, n=6):
    ctx = mp.get_context("fork")
    bar, q = ctx.Barrier(n), ctx.Queue()
    ps = [ctx.Process(target=_star, args=(str(root), i, bar, q)) for i in range(n)]
    [p.start() for p in ps]
    oshibki = [q.get(timeout=60) for _ in ps]
    [p.join() for p in ps]
    return [o for o in oshibki if o]


def test_gonka_n_processov_ne_chitaet_poluptoy(tmp_path):
    bad = []
    for k in range(200):
        bad += _povtor(tmp_path / f"r{k}")
    assert not bad, f"{len(bad)} падений из 200 повторов: {bad[:2]}"


def test_chitatel_v_pauze_zapisi_determinirovanno(tmp_path, monkeypatch):
    """Запись в два приёма с паузой; в паузе читатель смотрит manifest.json: нет файла или целый JSON."""
    mpath = tmp_path / "S1" / "manifest.json"
    vidno, pauzy = [], []

    def dvuhpriemnaya(self, data, encoding=None, **kw):
        with open(self, "w", encoding=encoding) as f:
            f.write(data[: len(data) // 2])
            f.flush()
            pauzy.append(1)
            try:
                json.loads(mpath.read_text(encoding="utf-8"))
                vidno.append("celyy")
            except FileNotFoundError:
                vidno.append("net")
            except json.JSONDecodeError:
                vidno.append("obrezan")
            f.write(data[len(data) // 2:])

    monkeypatch.setattr(Path, "write_text", dvuhpriemnaya)
    L.Pisatel(tmp_path, "S1", 0, dict(BOLSHOY))
    assert pauzy and set(vidno) <= {"net", "celyy"}, vidno


def test_manifest_pobaytno_prezhniy_i_bez_tmp(tmp_path):
    L.Pisatel(tmp_path, "S1", 0, {"T": 6, "kod": "K1", "vhody": "V1"})
    ozhid = {"T": 6, "kod": "K1", "vhody": "V1", "v": L.V, "format": L.FORMAT, "strategiya": "S1"}
    p = tmp_path / "S1" / "manifest.json"
    assert p.read_text(encoding="utf-8") == json.dumps(ozhid, ensure_ascii=False, sort_keys=True, indent=1)
    assert "sozdan" not in p.read_text(encoding="utf-8")
    assert sorted(f.name for f in p.parent.iterdir() if f.is_file()) == ["manifest.json"]


def test_chitatel_vo_vremya_zapisi_vidit_celyy_ili_nichego(tmp_path):
    p = tmp_path / "m.json"
    tekst = json.dumps({"x": list(range(50000))})
    stop, bad = threading.Event(), []

    def chit():
        while not stop.is_set():
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass                       # файла ещё/уже нет — допустимо
            except json.JSONDecodeError:
                bad.append(1)

    t = threading.Thread(target=chit)
    t.start()
    for k in range(200):
        p.unlink(missing_ok=True)
        L._zapis_atomarno(p, tekst)
    stop.set()
    t.join()
    assert not bad
    assert os.listdir(tmp_path) == ["m.json"]
