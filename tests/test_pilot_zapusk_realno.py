"""Прогрев Морриса v7 и пилот так, как их запускает сервер (scripts/server/perenos/zapusk_sobol.sh, PERENOS_KUB=v7 PERENOS_PILOT=1),
но на периоде train4 (данных 2022+ на Маке нет). Сухой режим скрипта этого не ловил: progon_morris_v7 подменяет canon_v6 на canon_v7,
а у того нет функции canon_v6 (AttributeError в плане Морриса). Сети нет."""
import pytest

import fastbt.progon_morris_v6 as pm
import fastbt.progon_morris_v7 as pm7
import fastbt.progon_v7 as p7

SIDS = ["S1", "S8", "S10"]


def _plan(sid):
    with pm7.kanon_v7(p7.canon7()):
        kub, ox = pm.setka(sid)
        return kub, ox, pm.plan(sid, kub, ox, 1, 10, int(kub["sobol"]["morris_seed"]))


@pytest.mark.parametrize("sid", SIDS)
def test_plan_morris_pod_kanonom_v7(sid):
    kub, ox, (nom, por, pts) = _plan(sid)
    assert pts and all(isinstance(p, dict) for p in pts)
    assert {n for n, _, _ in ox} >= {"k_okup"}                     # ось v7 открыта в Моррисе
