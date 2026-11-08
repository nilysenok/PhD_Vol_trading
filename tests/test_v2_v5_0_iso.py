"""V5-0, G9: диагностика листа 0 (`v5_0_dx*`) не импортируется путём прогноза (AST); `--stage 0 --check` не меняет файлов (хэш до/после); мутации — импорт диагностики путём прогноза и «проверка пишет служебный ключ» ловятся.
G5–G7: мутация «+1 день» (доля/score дня d+1) ловится."""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s12 import run_v5, v5_0_dx_ols as O, v5_0_dx_score as DS, v5_0_io as IO, v5_0_stage as ST

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"v5_0_stage.py", "run_v5.py"}               # стадия подключает диагностику; сами dx-модули друг друга импортируют


def imports_dx(source: str) -> bool:
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import) and any("v5_0_dx" in a.name for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and ("v5_0_dx" in (n.module or "") or any(a.name.startswith("v5_0_dx") for a in n.names)):
            return True
    return False


def violators() -> list:
    bad = []
    for d in ("s10", "s11", "s12"):
        for p in sorted((ROOT / "models_v2" / d).glob("*.py")):
            if p.name in ALLOWED or p.name.startswith("v5_0_dx"):
                continue
            if imports_dx(p.read_text(encoding="utf-8")):
                bad.append("%s/%s" % (d, p.name))
    return bad


def test_dx_not_imported_by_forecast_paths():
    assert violators() == []


def test_mutation_dx_import_in_forecast_path_is_caught():
    assert imports_dx("from models_v2.s12 import v5_0_dx_spec\n") and imports_dx("import models_v2.s12.v5_0_dx_score as x\n") and imports_dx("from models_v2.s12.v5_0_dx_ols import nw_ols\n")
    assert not imports_dx("from models_v2.s12 import v5_0_io\n")


def _tmp_env(tmp_path, monkeypatch):
    for src, name in ((IO.REP, "rep"), (IO.OUT, "out")):
        shutil.copytree(src, tmp_path / name)
    monkeypatch.setattr(IO, "REP", tmp_path / "rep")
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")


def test_check_changes_no_file(tmp_path, monkeypatch, capsys):
    _tmp_env(tmp_path, monkeypatch)
    before = IO.outputs_digest()
    rc = run_v5.main(["--stage", "0", "--check"])
    assert IO.outputs_digest() == before
    assert rc == 0


def test_mutation_check_writes_service_key_is_caught(tmp_path, monkeypatch, capsys):
    _tmp_env(tmp_path, monkeypatch)
    real_check = ST.check

    def writing_check(r):
        (IO.REP / "CHECK_runtime_sec.txt").write_text("1.0")          # служебный ключ, как DX_runtime_sec в V4
        return real_check(r)

    monkeypatch.setattr(ST, "check", writing_check)
    assert run_v5.main(["--stage", "0", "--check"]) == 1
    assert "изменил файлы" in capsys.readouterr().out


def _toy():
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2019-01-01", periods=300)
    r = rng.lognormal(0, 0.4, len(d))
    return pd.DataFrame({"ticker": "A", "date": d, "t_plus": d + pd.offsets.BDay(1), "rv_tplus": r, "k1": 1.0}), r


def test_known_scores_uses_only_closed_pairs_and_plus_one_mutation_is_caught():
    k, r = _toy()
    ok = DS.known_scores(k, "k1", lead=0)
    assert np.allclose(ok["s_own"].to_numpy()[1:], r[:-1] - 1.0)          # на дне d известна пара с t⁺ = d, т. е. s предыдущей строки
    leak = DS.known_scores(k, "k1", lead=1)
    assert np.corrcoef(np.log(r[:-1]), leak["s_own"].to_numpy()[:-1])[0, 1] > 0.9      # мутация: score пары, закрытой завтра, — утечка
    assert abs(np.corrcoef(np.log(r[1:]), ok["s_own"].to_numpy()[1:])[0, 1]) < 0.2


def test_partial_catches_next_day_share():
    rng = np.random.default_rng(1)
    d = pd.bdate_range("2019-01-01", periods=400)
    x = rng.normal(size=len(d))
    df = pd.DataFrame({"ticker": "A", "date": d, "y": np.r_[x[1:], 0.0] + 0.1 * rng.normal(size=len(d)), "x": x, "c": rng.normal(size=len(d))})
    assert abs(O.partial(df, "y", "x", ["c"])["partial_corr"]) < 0.2            # боевая привязка: доля дня d, y зависит от доли дня d+1 -> связи нет
    df["x_next"] = df["x"].shift(-1)
    assert abs(O.partial(df.dropna(), "y", "x_next", ["c"])["partial_corr"]) > 0.9      # мутация «+1 день»: утечка видна
