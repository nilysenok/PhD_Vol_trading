"""W8-A, диагностика `x1_w8_diag` (W8_SPEC 1.0–1.5, гейты A.G1, A.G2): пересчёт = `W8_DECOMP.csv` побайтно (sha256 0e0032469f5178cf…, 510 строк); вердикт П1–П5 выведен ИЗ CSV (а не из строк E_verdict) и
совпадает с ними; тождество «Σ вкладов = Δ» (1e-12) пересчитано независимо из сохранённых рядов; мутации: знак Δ, база процентов (№ 125/140), подмена числа в таблице переносов, порядок групп, `--check`, который
пишет (№ 123). Реальные файлы только читаются (autouse-хэши, сравнение содержимого до и после)."""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w8_diag as D
from models_v2.s16 import x1_w8_diag_v as V
from models_v2.s16 import x1_w8_reg as R

HASH = "0e0032469f5178cf"
BLOCKS = {"A_cell": 288, "B_claim": 21, "C_group": 104, "D_transfer": 40, "E_verdict": 10, "F_noise": 18, "G_null": 29}


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (R.TRIALS, R.CKPT) if p.exists()}
        s.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(R.REP.glob("W8_*")) if p.is_file()})
        s["W8"] = sorted(str(p) for p in R.W8.rglob("*")) if R.W8.exists() else []
        return s
    before = st()
    yield
    assert st() == before


@pytest.fixture(scope="module")
def rendered():
    return D.render()


def mutant(old: str, new: str) -> types.ModuleType:
    """Копия боевого `x1_w8_diag` с одной подложенной ошибкой в исходнике; сам файл не меняется."""
    src = Path(D.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    m = types.ModuleType("x1_w8_diag_mutant")
    m.__file__ = D.__file__
    exec(compile(src.replace(old, new), D.__file__, "exec"), m.__dict__)
    return m


# ---------- A.G1: пересчёт = файл ----------

def test_recompute_equals_decomp_csv_bytewise_and_hash_and_blocks(rendered) -> None:
    text, df, _ = rendered
    old = D.OUT.read_bytes()
    assert text.encode("utf-8") == old and hashlib.sha256(old).hexdigest().startswith(HASH)
    assert len(df) == 510 and df["block"].value_counts().to_dict() == BLOCKS            # G2 спецификации
    assert not D.OUT.read_text(encoding="utf-8").startswith("block")


def test_check_reads_only_and_catches_changed_file(rendered, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(D, "render", lambda: rendered)                                  # тот же пересчёт, без повторного счёта
    good = tmp_path / "W8_DECOMP.csv"
    good.write_bytes(D.OUT.read_bytes())
    snap = good.read_bytes()
    assert D.check(good) == [] and good.read_bytes() == snap
    good.write_bytes(snap.replace(b"weekend", b"weekenD", 1))
    assert any("≠ пересчёту" in b for b in D.check(good))                               # подменён символ в файле
    good.write_bytes(snap[:-1])
    assert D.check(good)                                                                # обрезан последний байт


def test_check_that_writes_is_caught_by_internal_and_external_hash(rendered, tmp_path, monkeypatch) -> None:
    f = tmp_path / "W8_DECOMP.csv"
    f.write_bytes(D.OUT.read_bytes())
    before = hashlib.sha256(f.read_bytes()).hexdigest()
    monkeypatch.setattr(D, "render", lambda: (f.write_text(f.read_text(encoding="utf-8") + "#x\n", encoding="utf-8"), rendered)[1])          # мутация № 123: check пишет
    assert any("--check изменил файл" in b for b in D.check(f))
    assert hashlib.sha256(f.read_bytes()).hexdigest() != before                         # и внешний хэш это видит


def test_cli_check_and_default_dry_run_write_nothing(rendered, monkeypatch, capsys) -> None:
    monkeypatch.setattr(D, "render", lambda: rendered)
    monkeypatch.setattr(sys, "argv", ["x1_w8_diag", "--check"])
    with pytest.raises(SystemExit) as e:
        D.main()
    assert e.value.code == 0 and "ok" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x1_w8_diag"])
    D.main()                                                                            # без флага: сухой прогон, файлы не меняются (autouse)
    assert "сухой прогон" in capsys.readouterr().out


# ---------- тождество вкладов, независимо ----------

def test_contributions_sum_to_delta_and_delta_equals_independent_recount(rendered) -> None:
    _, df, x = rendered
    a = df[df["block"] == "A_cell"]
    for base in ("X1", "K2", "K3"):
        b = a[a["base"] == base]
        for p in ("h1", "h2", "M", *D.YEARS):
            r = b[b["period"] == p]
            tot = r[r["type"].isin(D.TYPES)]["contrib_pp_period"].sum()
            assert abs(tot - r[r["type"] == "all"]["delta_pct"].iloc[0]) < 1e-12, (base, p)       # Σ по типам = Δ периода
        yrs = b[(b["period"].isin(D.YEARS)) & (b["type"] == "all")]["contrib_pp_M"].sum()
        assert abs(yrs - b[(b["period"] == "M") & (b["type"] == "all")]["delta_pct"].iloc[0]) < 1e-12      # Σ по годам = Δ окна
    rv = x["rv"].to_numpy()
    lq = lambda f: rv / f - np.log(rv / f) - 1                                          # noqa: E731  (QLIKE дня, независимая запись)
    for base, col in (("X1", "F"), ("K2", "K2"), ("K3", "K3")):
        want = 100 * (lq(x["W5G"].to_numpy()).mean() - lq(x[col].to_numpy()).mean()) / lq(x[col].to_numpy()).mean()
        got = a[(a["base"] == base) & (a["period"] == "M") & (a["type"] == "all")]["delta_pct"].iloc[0]
        assert abs(got - want) < 1e-10, base
    assert abs(a[(a["base"] == "X1") & (a["period"] == "M") & (a["type"] == "all")]["delta_pct"].iloc[0] - (-0.4393)) < 1e-3     # разд. 1.2–1.3 спецификации


def tiny() -> pd.DataFrame:
    return pd.DataFrame(dict(l_X1=[1.0, 1.0, 2.0], l_K2=[2.0, 2.0, 2.0], l_W5G=[0.5, 1.5, 1.0], c_app=[0.1, 0.0, 0.2], u=[0.0, 0.1, 0.2]))


def assert_cell_known(m) -> None:
    x = tiny()
    c = m.cell(x, np.array([True, True, False]), "X1", 3, 1.0)
    assert c["n"] == 2 and c["Q_base"] == 1.0 and c["Q_W5G"] == 1.0 and c["delta_pct"] == 0.0 and c["sum_diff"] == 0.0
    c = m.cell(x, np.array([True, False, False]), "X1", 3, 1.0)                           # W5-G лучше X1: Δ = −50 %, база процентов — Q базы
    assert c["delta_pct"] == -50.0 and c["sum_diff"] == -0.5
    c = m.cell(x, np.array([False, False, True]), "K2", 3, 2.0)
    assert c["delta_pct"] == -50.0 and c["Q_base"] == 2.0
    assert np.isnan(m.cell(x, np.zeros(3, bool), "X1", 3, 1.0)["delta_pct"])             # пустая клетка не падает


def test_cell_known_answer_on_real_code() -> None:
    assert_cell_known(D)


@pytest.mark.parametrize("old,new", [
    ("float((lw.mean() - lb.mean()) / lb.mean() * 100)), sum_diff", "float((lb.mean() - lw.mean()) / lb.mean() * 100)), sum_diff"),       # знак Δ перевёрнут (№ 125)
    ("float((lw.mean() - lb.mean()) / lb.mean() * 100)), sum_diff", "float((lw.mean() - lb.mean()) / lw.mean() * 100)), sum_diff"),       # база процентов — W5-G вместо базы
    ("sum_diff=float((lw - lb).sum())", "sum_diff=float((lb - lw).sum())"),                                                              # знак вклада
])
def test_cell_mutations_are_caught(old, new) -> None:
    with pytest.raises(AssertionError):
        assert_cell_known(mutant(old, new))


def test_block_a_identity_breaks_when_a_type_is_dropped() -> None:
    x = pd.DataFrame(dict(yr=["2018"] * 4, half=["h1"] * 4, gap=[1, 1, 1, 0], type=["weekend", "hol1", "long", "none"],
                          l_X1=[1.0, 1.0, 1.0, 1.0], l_K2=[1.0] * 4, l_K3=[1.0] * 4, l_W5G=[0.5, 1.5, 2.0, 1.0], c_app=[0.0] * 4, u=[0.0] * 4))
    rows, ident = D.block_a(x)
    assert ident < 1e-12 and rows[0]["block"] == "A_cell"
    x2 = x.assign(type=["weekend", "hol1", "other_x", "none"])                            # тип вне списка TYPES: вклад выпал → тождество нарушено
    assert D.block_a(x2)[1] > 1e-3


# ---------- A.G2: вердикт из CSV ----------

def write_csv(path: Path, d_rows: list, c_rows: list, e_rows: list) -> Path:
    cols = ["block", "feature", "half", "direction", "group", "n", "beta_expost", "d_feat_vs_ctrl_pct", "d_feat_vs_X1_pct", "passed"]
    path.write_text("# шапка\n# вторая строка, с запятыми\n" + pd.DataFrame(d_rows + c_rows + e_rows, columns=cols).to_csv(index=False), encoding="utf-8")
    return path


DIRS = ("h1->h2", "h2->h1", "h1->h2no22", "h2no22->h1")


def feat(name, ctrl=(-1, -1, -1, -1), x1=(-1, -1, -1, -1), b1=(0.3, 0.1), b2=(0.3, 0.1), n1=(30, 30), n2=(30, 30), passed=1, na=None):
    d = [["D_transfer", name, None, dn, None, None, None, c, x, None] for dn, c, x in zip(DIRS, ctrl, x1)]
    c = [["C_group", name, h, None, g, nn, b, None, None, None] for h, bs, ns in (("h1", b1, n1), ("h2no22", b2, n2)) for g, nn, b in zip(("T1", "T2"), ns, bs)]
    if na:                                                                                # группа `na` с заданными β ex post в h1 и h2no22 (n = 40 в обеих)
        c += [["C_group", name, h, None, "na", 40, b, None, None, None] for h, b in (("h1", na[0]), ("h2no22", na[1]))]
    return d, c, [["E_verdict", name, None, None, None, None, None, None, None, passed]]


def build(tmp_path, feats) -> Path:
    d, c, e = [], [], []
    for f in feats:
        d += f[0]
        c += f[1]
        e += f[2]
    return write_csv(tmp_path / "W.csv", d, c, e)


def verdict_of(tmp_path, **kw) -> bool:
    return bool(V.verdict(build(tmp_path, [feat("f", **kw)]))[0]["PASS"].iloc[0])


def test_verdict_rule_known_answer_and_each_condition_is_caught(tmp_path) -> None:
    assert verdict_of(tmp_path) is True
    assert verdict_of(tmp_path, ctrl=(0, -1, -1, -1)) is False                           # П1: «< 0», ноль не проходит
    assert verdict_of(tmp_path, ctrl=(-1, 0.01, -1, -1)) is False
    assert verdict_of(tmp_path, x1=(-1, -1e-9, -1, -1)) is True and verdict_of(tmp_path, x1=(-1, 0, -1, -1)) is False        # П2: < 0 (к X1 в h1→h2, h2→h1)
    assert verdict_of(tmp_path, ctrl=(-1, -1, 0.0, -1)) is False and verdict_of(tmp_path, ctrl=(-1, -1, -1, 0.3)) is False     # П3
    assert verdict_of(tmp_path, b2=(0.1, 0.3)) is False                                  # П4: порядок групп разный
    assert verdict_of(tmp_path, n2=(30, 14)) is False                                    # П4: группа < 15 дней не в счёте → общих групп 1
    assert verdict_of(tmp_path, x1=(-1, -1, 0.0, 0.0)) is True                           # П5: «≤ 0», ноль проходит (П2 по первым двум направлениям)
    assert verdict_of(tmp_path, x1=(-1, -1, 1e-9, 0.0)) is False and verdict_of(tmp_path, x1=(-1, -1, 0.0, 1e-9)) is False


def test_verdict_comes_from_tables_not_from_e_verdict_row(tmp_path) -> None:
    r, passed, total = V.verdict(build(tmp_path, [feat("f", passed=0)]))                  # таблицы проходят, строка E_verdict говорит «нет»
    assert passed == ["f"] and total == 1 and bool(r["differs_from_E_verdict"].iloc[0])
    r, passed, _ = V.verdict(build(tmp_path, [feat("f", ctrl=(1, 1, 1, 1), passed=1)]))   # таблицы не проходят, строка говорит «да»
    assert passed == [] and bool(r["differs_from_E_verdict"].iloc[0])


def test_na_group_is_ignored_in_order(tmp_path) -> None:
    assert verdict_of(tmp_path, na=(0.9, -0.9)) is True                                   # `na` с обратным порядком не входит в П4: общих групп по-прежнему T1, T2
    assert verdict_of(tmp_path, n2=(30, 14), na=(0.9, 0.1)) is False                      # `na` не заменяет выпавшую группу < 15 дней


def test_main_returns_zero_only_for_ten_features_matching_e_verdict(tmp_path, capsys) -> None:
    ok = [feat("f%d" % i, passed=1 if i == 9 else 0, ctrl=(-1, -1, -1, -1) if i == 9 else (1, 1, 1, 1)) for i in range(10)]
    p = build(tmp_path, ok)
    assert V.main(p) == 0 and "прошли: f9; проверено 10" in capsys.readouterr().out
    assert V.main(build(tmp_path, ok[:9])) == 1                                          # проверено 9, а не 10
    bad = ok[:9] + [feat("f9", passed=0)]                                                # таблицы проходят, строка E_verdict — нет
    assert V.main(build(tmp_path, bad)) == 1


def test_real_verdict_from_csv_is_w8d8_only_and_agrees_with_e_verdict_rows(capsys) -> None:
    r, passed, total = V.verdict()
    assert (passed, total) == (["W8D8_imoex_rv"], 10) and not r["differs_from_E_verdict"].any()
    assert V.main() == 0 and "прошли: W8D8_imoex_rv; проверено 10" in capsys.readouterr().out
    assert r.set_index("feature").loc["W8D3_holiday", ["P1", "P2", "P3", "P4", "P5"]].tolist() == [False, True, True, True, True]      # разд. 1.5: W8D3 не прошёл П1


def test_real_verdict_follows_a_substituted_number_in_the_table(tmp_path) -> None:
    t = pd.read_csv(V.DECOMP, comment="#", dtype=str, keep_default_na=False)
    m = (t["block"] == "D_transfer") & (t["feature"] == "W8D8_imoex_rv") & (t["direction"] == "h2->h1")
    assert m.sum() == 1
    t.loc[m, "d_feat_vs_ctrl_pct"] = "0.5"                                               # подмена числа: П1 перестаёт выполняться
    p = tmp_path / "W8_DECOMP.csv"
    p.write_text(t.to_csv(index=False), encoding="utf-8")
    r, passed, total = V.verdict(p)
    assert passed == [] and total == 10 and bool(r.set_index("feature").at["W8D8_imoex_rv", "differs_from_E_verdict"])
    assert V.main(p) == 1                                                                 # строка E_verdict осталась старой → расхождение
