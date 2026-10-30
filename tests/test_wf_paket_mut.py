"""Мутации пакета wf (проверка PR #732), не пойманные тестами автора: (г) `usech_2021.main` берёт CUT не из --do;
(д) поведение по умолчанию (без --do) — граница 2021-12-31, выход data/server_2021; (в) PERENOS_DO не доходит до VERSIYA
(сборщик не пишет `perenos_do`, `obshchee.sh` не читает). Синтетика во временном каталоге, данных репозитория не читает."""
import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRV = REPO / "scripts/server"
DO = "2026-02-28"


def _load():
    import sys
    sys.path.insert(0, str(SRV))
    s = importlib.util.spec_from_file_location("usech_2021_mut", SRV / "usech_2021.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


U = _load()
KAL = ("date,known_from,retrieved_at\n2021-05-03,2021-04-01,2026-10-30\n2021-12-31,2021-12-30,2026-10-30\n"
       "2022-01-03,2021-12-30,2026-10-30\n2026-02-13,2026-02-01,2026-10-30\n2026-02-20,2026-03-10,2026-10-30\n")


def _koren(tmp_path: Path) -> list[str]:
    (tmp_path / "data/calendar").mkdir(parents=True)
    (tmp_path / "data/calendar/cbr.csv").write_text(KAL)
    (tmp_path / "m.tsv").write_text("put\tsha256\tbajty\tmax_data\tchitatel\n")
    (tmp_path / "m_otvergnuto.tsv").write_text("put\tprichina\tchitatel\n")
    return ["--koren", str(tmp_path), "--tsv", str(tmp_path / "m.tsv"), "--tolko", "calendar/cbr"]


def _stroki(p: Path) -> list[str]:
    return [ln.split(",")[0] for ln in p.read_text().splitlines()[1:]]


def test_main_do_wf_rezhet_po_do_s_otmetkoj(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "CUT", U.CUT)                                    # глобал main восстанавливается после теста
    otm = tmp_path / "otm.json"
    otm.write_text('{"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "' + "a" * 64 + '", "data_snyatiya": "2026-10-30"}')
    assert U.main(_koren(tmp_path) + ["--do", DO, "--otmetka", str(otm)]) == 0
    assert _stroki(tmp_path / "data/server_wf/data/calendar/cbr.csv") == ["2021-05-03", "2021-12-31", "2022-01-03", "2026-02-13"]   # known_from 03-10 > DO — нет
    assert not (tmp_path / "data/server_2021").exists()                     # мутация (г): CUT=2021 оставила бы одну строку 2021


def test_main_po_umolchaniyu_rezhet_po_2021(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "CUT", U.CUT)
    assert U.main(_koren(tmp_path)) == 0
    assert _stroki(tmp_path / "data/server_2021/data/calendar/cbr.csv") == ["2021-05-03", "2021-12-31"]   # 2022+ не попадает без --do
    assert not (tmp_path / "data/server_wf").exists()


def test_perenos_do_v_versiyu_pishetsya_i_chitaetsya(tmp_path):
    s = (SRV / "sobrat_perenos.sh").read_text()
    pf = next(ln for ln in s.splitlines() if ln.lstrip().startswith("printf") and "tar_sha256" in ln)
    i = s.index(pf)
    blok = s[i:s.index("> \"$OUT/VERSIYA\"", i)]
    assert "perenos_do=%s" in blok and '"${PERENOS_DO:-}"' in blok           # мутация (в): ключ и значение выпали
    assert blok.count("%s") == len(re.findall(r'"[^"]*"', blok.split("\\n'", 1)[1]))      # число полей = число аргументов
    (tmp_path / "obshchee.sh").write_text((SRV / "perenos/obshchee.sh").read_text())
    (tmp_path / "VERSIYA").write_text(f"commit_short=abc\ntar_sha256=ff\nperenos_do={DO}\n")
    e = {k: v for k, v in os.environ.items() if not k.startswith("PERENOS_")}
    r = subprocess.run(["bash", "-c", '. ./obshchee.sh; echo "[$PERENOS_DO]"'], cwd=tmp_path, env=e, capture_output=True, text=True)
    assert r.stdout.strip().endswith(f"[{DO}]")                             # читатель: из VERSIYA, если переменная не задана
