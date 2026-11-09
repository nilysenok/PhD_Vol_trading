"""V7 лист FX, G32: диагностика рассогласования координат z. Известный ответ на синтетике: переставленные и перевёрнутые компоненты — corr_same ≈ 0 / −1, перестановка находит соответствие;
одинаковые координаты — единичная диагональ. Мутация: корреляция без знака (|corr|) не отличает перевёрнутую компоненту — ловится. Артефакты Stage A подменены загрузчиком."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s14 import v7_fx_diag as DG

D = pd.bdate_range("2017-06-01", "2019-12-31")
Z = np.random.default_rng(0).normal(size=(len(D), 4))


def loader(transform):
    """Энкодер года Y+1 = transform(z энкодера Y); обе таблицы — на дни ≤ конца своего года."""
    def load(path):
        year = int(Path(path).name.split("_")[2])
        days = D[D < pd.Timestamp(year + 1, 1, 1)]
        z = Z[:len(days)]
        return dict(days=days.to_numpy("datetime64[ns]").astype(np.int64), z=transform(z) if (year % 2 == 1) else z)
    return load


def test_identical_coordinates_give_unit_diagonal_and_identity_permutation():
    df = DG.rows(loader(lambda z: z), kinds=("tcn",), seeds=(0,), years=(2018,))
    assert np.allclose(df["corr_same"], 1.0) and df["perm_is_identity"].all() and (df["n_dates"] > 200).all()


def test_flipped_and_permuted_components_are_found_with_sign():
    df = DG.rows(loader(lambda z: np.column_stack([-z[:, 0], z[:, 2], z[:, 1], z[:, 3]])), kinds=("tcn",), seeds=(0,), years=(2018,))
    c = df.set_index("comp")
    assert np.isclose(c.loc[1, "corr_same"], -1.0) and abs(c.loc[2, "corr_same"]) < 0.2 and abs(c.loc[3, "corr_same"]) < 0.2 and np.isclose(c.loc[4, "corr_same"], 1.0)
    assert not c["perm_is_identity"].any() and (c["perm_comp"].tolist() == [1, 3, 2, 4]) and np.isclose(c.loc[1, "corr_best_perm"], -1.0)
    assert not np.isclose(np.abs(c.loc[1, "corr_same"]), c.loc[1, "corr_same"])    # знак сохранён: мутант «|corr|» этого не покажет


def test_best_perm_matches_known_answer():
    p, tot = DG.best_perm(np.array([[0.0, -0.9], [0.8, 0.1]]))
    assert p == (1, 0) and np.isclose(tot, 1.7)
