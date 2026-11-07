"""P3-17, починка (а): проверка `k4567` — вместо устаревшего «used == 0» (после P2 used = 20) проверяется, что строк K4–K7 нет в `ALL_TRIALS.csv`.
Синтетический реестр; боевая функция `hp_ctrl.used_errors`, которую вызывает `run_hp.k4567_check`."""
from __future__ import annotations

import inspect

import pytest

from models_v2.s10 import hp_ctrl as CT, run_hp as RH

HEAD = "key,program,source,direction,config,kind\n"


def reg(tmp_path, keys) -> object:
    p = tmp_path / "ALL_TRIALS.csv"
    p.write_text(HEAD + "".join('%s,prog,src,dir,"{""a"": 1}",trial\n' % k for k in keys), encoding="utf-8")
    return p


def test_used_20_gibridy_ne_oshibka(tmp_path):
    """После P2: 20 строк программы (A1…D2 обоих модулей) и старые Ж-204 — ошибки нет (старое условие `used == 0` дало бы ошибку)."""
    keys = ["Ж-204:90a6f716373ba33b"] + ["Ж-217:%s:%s:abc" % (m, n) for m in ("xgb", "lgbm") for n in ("A1", "A2", "B1", "C1", "D2")]
    assert CT.used_errors(reg(tmp_path, keys)) == []


@pytest.mark.parametrize("name", ["K4", "k5", "K6", "K7"])
def test_stroka_kontrolya_v_reestre_lovitsya(tmp_path, name):
    """Мутация: контроль записан в реестр как trial (пошёл в N) — проверка обязана падать."""
    err = CT.used_errors(reg(tmp_path, ["Ж-217:xgb:A1:abc", "Ж-217:xgb:%s:def" % name]))
    assert len(err) == 1 and name in err[0]


def test_k4567_check_ne_trebuet_used_nol():
    """Боевая проверка не обращается к `used` из CHECKPOINT (источник правды — реестр)."""
    src = inspect.getsource(RH.k4567_check)
    assert "used" in src and "CT.used_errors" in src and '"used"' not in src and "CHECKPOINT used" not in src
