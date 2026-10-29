"""r1_v: период должен доходить до воркеров пула (на Маке старт — spawn, модульные глобалы не наследуются).
Причина падения np.stack в stroki_a: S1 в родителе на test1, S2–S6 в воркерах на умолчании train4."""
from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "revers"))
import r1_v as R  # noqa: E402

IMOEX = "imoex_fiktivnyj.csv"      # nastroit только запоминает путь, файл не читается


def test_bez_initializer_vorker_vidit_train4():
    """Воспроизведение причины: без initializer воркер spawn видит умолчание, а не период родителя."""
    R.nastroit("test1", IMOEX)
    with multiprocessing.get_context("spawn").Pool(1) as p:
        assert p.apply(R.period_vorkera) == "train4"


def test_s_initializer_vorker_vidit_test1():
    R.nastroit("test1", IMOEX)
    with R.sozdat_pul("test1", IMOEX, 2, multiprocessing.get_context("spawn")) as p:
        assert p.map(_vyzov, range(2)) == ["test1", "test1"]


def _vyzov(_):
    return R.period_vorkera()
