"""Пакет пилота-1 (S1–S6, v7, polnaya): (1) кубы v7 и SHA256SUMS едут в пакете (белый список sobrat_perenos.sh), без data/;
(2) `PERENOS_KUB=v7 PERENOS_PILOT=1 zapusk_sobol.sh polnaya` запускает progon_v7 --pilot --n1 1000 без Морриса и без полного счёта.
Мутации: v7 не в списке пакета; пилот уходит в полный счёт (200 000, Моррис, --zamorozka). Синтетика, без сети."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_perenos_v7 import SH, _pakyet, _ryad  # noqa: E402

SOBRAT = (SH.parent / "sobrat_perenos.sh").read_text()
V7 = ["fastbt/designs/v7/kub_S1.yaml", "fastbt/designs/v7/kub_S6.yaml", "fastbt/designs/v7/SHA256SUMS"]


def _belyj(text: str):
    """Регулярка из строки белого списка designs/ в sobrat_perenos.sh."""
    m = re.search(r"'(\^fastbt/designs/.*\$)'", text)
    assert m, "нет белого списка designs/"
    return re.compile(m.group(1))


def test_g1_kuby_v7_i_sha256sums_v_pakete():
    rx = _belyj(SOBRAT)
    assert all(rx.match(p) for p in V7), [p for p in V7 if not rx.match(p)]
    for p in ("fastbt/designs/v6/kub_S1.yaml", "fastbt/designs/sweep/x.yaml", "fastbt/designs/progon/a.yaml"):
        assert rx.match(p), p                                          # прежнее не потеряно
    for p in ("fastbt/designs/v7/dannye.csv", "fastbt/designs/v7/x.parquet", "fastbt/designs/v8/kub_S1.yaml", "fastbt/designs/v7/sub/kub_S1.yaml"):
        assert not rx.match(p), p                                      # только yaml кубов и сумма


def test_g1_mutatsiya_v7_ne_v_pakete_lovitsya():
    mut = re.sub(r"\|v7/\([^)]*\)", "", SOBRAT)
    assert mut != SOBRAT and not all(_belyj(mut).match(p) for p in V7)


def test_g1_kuby_v7_kopiruyutsya_v_kuby_i_sha_ryadom():
    cikl = SOBRAT.split("for d in progon")[1].split("done")[0]
    assert cikl.startswith(" v5 v6 v7;") and "SHA256SUMS" in cikl


def _proverka_pilot(r, calls):
    assert r.returncode == 0, r.stdout + r.stderr
    pil = [c for c in calls if "fastbt.progon_v7" in c]
    assert len(pil) == 1, calls
    c = pil[0]
    assert c.startswith("-m fastbt.progon_v7 --kub v7 --period polnaya") and " --pilot" in c and "--n1 1000" in c
    assert "--semeystva S1,S2,S3,S4,S5,S6" in c and "--procs 64" in c and "ledger/polnaya_v7_pilot1" in c
    assert "--zamorozka" not in c and "200000" not in c and "--seed 2" not in c
    assert not any("progon_morris_v7" in x and "--r 1" not in x for x in calls)       # Морриса, кроме прогрева r=1, нет
    assert "ledger/polnaya_morris" not in "\n".join(calls) and not any("progon_server" in x for x in calls)
    assert "pilot_zamer.json" in r.stdout and "ledger/polnaya_v7_pilot1" in r.stdout


def _pilot(tmp, **env):
    return _ryad(_pakyet(tmp), PERENOS_PILOT="1", **env)


def test_g2_pilot_komandy_bez_morrisa_i_polnogo_scheta(tmp_path):
    _proverka_pilot(*_pilot(tmp_path))


def test_g2_mutatsiya_pilot_zapuskaet_polnyj_schet_lovitsya(tmp_path):
    """Без PERENOS_PILOT скрипт идёт полным счётом (Моррис + s1 s2, N1 200 000): проверка пилота обязана упасть."""
    with pytest.raises(AssertionError):
        _proverka_pilot(*_ryad(_pakyet(tmp_path)))


def test_g2_pilot_peremennye_n1_semejstva_procesy(tmp_path):
    _, calls = _pilot(tmp_path, PERENOS_N1="50", PERENOS_SEMEYSTVA="S1,S6", PERENOS_PROCS_PILOT="8")
    c = [x for x in calls if "fastbt.progon_v7" in x][0]
    assert "--n1 50" in c and "--semeystva S1,S6" in c and "--procs 8" in c


def test_g2_pilot_procesy_ne_bolshe_yader_minus_4(tmp_path):
    _, calls = _pilot(tmp_path, PERENOS_YADRA="40")
    assert "--procs 36" in [x for x in calls if "fastbt.progon_v7" in x][0]


def test_g2_bez_pilot_polnyj_schet_ne_izmenilsya(tmp_path):
    _, calls = _ryad(_pakyet(tmp_path), PERENOS_PROHODY="s1")
    assert not any("--pilot" in c for c in calls) and any("--n1 200000" in c for c in calls)


def test_g2_pilot_tolko_dlya_v7(tmp_path):
    r, calls = _ryad(_pakyet(tmp_path), per="train4", kub="v5", PERENOS_PILOT="1")
    assert r.returncode != 0 and "FAIL" in r.stdout and not any("--n1" in c for c in calls)


def test_g3_bash_n():
    for p in (SH / "zapusk_sobol.sh", SH.parent / "sobrat_perenos.sh"):
        assert subprocess.run(["bash", "-n", str(p)]).returncode == 0, p


def test_g3_zamok_polnaya_ne_trebuet_nichego_krome_paketa():
    """Отметка wf_zamok_snyat.json едет в пакете (корень + tar), polnaya проверяет её как wf; чтение < chtenie_konec — в коде драйвера."""
    t = (SH / "zapusk_sobol.sh").read_text()
    assert 'if [ "$per" = wf ] || [ "$per" = polnaya ]' in t and "otmetka_wf" in t and "wf_zamok_snyat.json" in SOBRAT
