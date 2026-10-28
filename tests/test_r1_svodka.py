import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts/revers/r1_svodka.py"
_s = importlib.util.spec_from_file_location("r1_svodka", _p)
sv = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sv)

HEAD = f"# {sv.MARKER}\n# коммит: abc123\n# вход: sigma sha256=ff\n"


def make(tmp_path):
    (tmp_path / "docs/otchety/revers_r1").mkdir(parents=True)
    (tmp_path / "docs/tables/revers_r1").mkdir(parents=True)
    for k in sv.LISTY:
        (tmp_path / f"docs/otchety/revers_r1/{k}.md").write_text(f"сдвиг {k} ок\nтекст\n", encoding="utf-8")
        (tmp_path / f"docs/tables/revers_r1/{k}_t.csv").write_text(HEAD + "date,v\n2022-01-03,1\n", encoding="utf-8")
    return tmp_path


def run(root):
    out = root / "svodka.md"
    rc = sv.main(["--root", str(root), "--out", str(out)])
    return rc, out.read_text(encoding="utf-8")


def test_polnyj_nabor_ok(tmp_path):
    rc, txt = run(make(tmp_path))
    assert rc == 0
    assert txt.splitlines()[0].startswith("Реверс раунд 1, 2022–2024, справочно, без выбора; листов готово 7 из 7")
    assert "abc123" in txt and "sha256=" in txt and "ИТОГ: OK" in txt


def test_net_shapki_fail(tmp_path):
    root = make(tmp_path)
    (root / "docs/tables/revers_r1/a_t.csv").write_text("date,v\n2022-01-03,1\n", encoding="utf-8")
    rc, txt = run(root)
    assert rc == 1 and "нет шапки" in txt


def test_data_2025_fail(tmp_path):
    root = make(tmp_path)
    (root / "docs/tables/revers_r1/b_t.csv").write_text(HEAD + "date,v\n2025-01-09,1\n", encoding="utf-8")
    rc, txt = run(root)
    assert rc == 1 and "дата ≥ 2025" in txt


def test_bps_fail(tmp_path):
    root = make(tmp_path)
    (root / "docs/otchety/revers_r1/v.md").write_text("сдвиг\nиздержки 5 bps\n", encoding="utf-8")
    rc, txt = run(root)
    assert rc == 1 and "bps" in txt


def test_lishnyaya_tablica(tmp_path):
    root = make(tmp_path)
    (root / "docs/tables/revers_r1/x_lishnyaya.csv").write_text(HEAD + "date,v\n2022-01-03,1\n", encoding="utf-8")
    rc, txt = run(root)
    part = txt.split("Вне предрегистрации (не удалены):")[1].split("##")[0]
    assert rc == 0 and "x_lishnyaya.csv" in part


def test_zhdyot_v_pervoj_stroke(tmp_path):
    root = make(tmp_path)
    (root / "docs/otchety/revers_r1/g.md").write_text("сдвиг\nждёт R-P\n", encoding="utf-8")
    rc, txt = run(root)
    assert "листов готово 6 из 7" in txt.splitlines()[0] and "г: ABANDON/ждёт" in txt.splitlines()[0]
