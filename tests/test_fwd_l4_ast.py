"""Л4: изоляция и выходы. AST (класс № 141): модули cfg8/X1/ZB не обучают энкодер и не пересчитывают PCA, не вызывают пишущие стадии архива (`write`, `emit`, `merge_cp`);
выходы `data/fwd/models/{cfg8,x1,zb,zb0}.parquet` содержат только архивные строки (t⁺ <= 2026-02-27). Сканер ловит подложенные тексты."""
from __future__ import annotations

import ast

import pandas as pd
import pytest

from models_v2.s17 import fwd_src as S

R = S.ROOT
MODS = ["fwd_cfg8", "fwd_x1", "fwd_zb"]
BANNED = {"train_encoder", "fit_pca_past", "train_missing", "compute", "write", "emit", "merge_cp", "merge_checkpoint", "write_cfg8", "choose_i2", "loyo_scores"}
OUTS = ["cfg8", "x1", "zb", "zb0"]


def used_banned(text: str) -> set:
    """Имена из BANNED, встреченные как вызов, атрибут или импорт (определение собственной функции с таким именем — не вызов)."""
    found = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Attribute) and n.attr in BANNED:
            found.add(n.attr)
        elif isinstance(n, ast.Name) and n.id in BANNED:
            found.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            found |= {a.name.split(".")[-1] for a in n.names} & BANNED
    return found


@pytest.mark.parametrize("mod", MODS)
def test_ast_net_obucheniya_i_pishushchih_stadij(mod):
    text = (R / f"models_v2/s17/{mod}.py").read_text(encoding="utf-8")
    assert used_banned(text) == set(), mod
    assert len(text.splitlines()) <= 200


@pytest.mark.parametrize("text,name", [("from models_v2.s14.v7_1e_pre import train_encoder", "train_encoder"), ("STA.compute(x)", "compute"), ("x = fit_pca_past(a)", "fit_pca_past"),
                                       ("ST.write()", "write"), ("from models_v2.s15.de_e1_stage import emit", "emit"), ("R.merge_cp('k', {})", "merge_cp")])
def test_skaner_lovit_podlozhennye_teksty(text, name):
    assert used_banned(text) == {name}


@pytest.mark.parametrize("name", OUTS)
def test_vyhody_tolko_arhivnye_stroki(name):
    p = R / f"data/fwd/models/{name}.parquet"
    if not p.exists():
        pytest.skip("нет выхода")
    df = pd.read_parquet(p)
    assert df["t_plus"].max() <= S.ARCHIVE_END and df["d"].max() <= pd.Timestamp("2026-02-26") and len(df) == 2018
