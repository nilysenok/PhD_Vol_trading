import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts/revers/r1_svodka.py"


def load(src=None):
    s = importlib.util.spec_from_file_location("r1_svodka_x", _p)
    m = importlib.util.module_from_spec(s)
    if src is None:
        s.loader.exec_module(m)
    else:
        exec(compile(src, str(_p), "exec"), m.__dict__)
    return m


def heads(sv, net):
    h = f"# {sv.MARKER}\n# коммит: abc123\n# вход: sigma sha256=ff\n"
    return h + ("# дивиденды: НЕТ\n" if net else "")


def make(tmp_path, sv, net_head=True, main_net=False):
    tb = tmp_path / "docs/tables/revers_r1"
    (tb / "_net").mkdir(parents=True)
    (tmp_path / "docs/otchety/revers_r1").mkdir(parents=True)
    for k in sv.LISTY:
        (tmp_path / f"docs/otchety/revers_r1/{k}.md").write_text(f"сдвиг {k}\nтекст\n", encoding="utf-8")
        if k in ("a", "b"):
            continue
        (tb / f"{k}_t.csv").write_text(heads(sv, False) + "date,v\n2022-01-03,1\n", encoding="utf-8")
    cols = "объект,год,Sharpe,mSR,доходность\n"
    (tb / "a_x.csv").write_text(heads(sv, main_net) + cols + "A,2022,1.0,0.8,0.10\nA,2023,2.0,1.5,0.20\n",
                                encoding="utf-8")
    (tb / "_net/a_x.csv").write_text(heads(sv, net_head) + cols + "A,2022,0.4,0.3,0.04\nA,2023,1.5,1.0,0.15\n",
                                     encoding="utf-8")
    (tb / "b_razlozhenie.csv").write_text(heads(sv, False) + "срез,Sharpe\nлонг,1.0\n", encoding="utf-8")
    (tb / "_net/b_razlozhenie_net.csv").write_text(heads(sv, net_head) + "срез,Sharpe\nлонг,0.5\n", encoding="utf-8")
    return tmp_path


def run(sv, root):
    out = root / "svodka.md"
    rc = sv.main(["--root", str(root), "--out", str(out)])
    return rc, out.read_text(encoding="utf-8")


def test_pary_i_raznost(tmp_path):
    sv = load()
    rc, txt = run(sv, make(tmp_path, sv))
    assert rc == 0, txt
    sec = txt.split("## Цена дивидендной ошибки защищённой версии")[1].split("## Хэши")[0]
    assert "| A | 2022 | 0.6000 | 0.5000 | 6.0000 |" in sec
    assert "| A | 2023 | 0.5000 | 0.5000 | 5.0000 |" in sec
    assert "| лонг | 0.5000 |" in sec
    assert "как в версии предзащиты" in txt
    assert "_net" not in txt.split("Вне предрегистрации")[1].split("##")[0]


def test_net_est_bez_pary_fail(tmp_path):
    sv = load()
    root = make(tmp_path, sv)
    f = root / "docs/tables/revers_r1/_net/a_x.csv"
    f.write_text(f.read_text(encoding="utf-8").replace("A,2023,1.5,1.0,0.15\n", ""), encoding="utf-8")
    rc, txt = run(sv, root)
    assert rc == 1 and "без пары" in txt


def test_net_bez_pometki_fail(tmp_path):
    sv = load()
    rc, txt = run(sv, make(tmp_path, sv, net_head=False))
    assert rc == 1 and "нет «# дивиденды: НЕТ»" in txt


def test_osnovnaya_s_pometkoj_net_fail(tmp_path):
    sv = load()
    rc, txt = run(sv, make(tmp_path, sv, main_net=True))
    assert rc == 1 and "основная таблица помечена" in txt


def test_mutaciya_znak_raznosti_lovitsya(tmp_path):
    src = _p.read_text(encoding="utf-8")
    assert "(x - y) * kf" in src
    mut = load(src.replace("(x - y) * kf", "(y - x) * kf"))
    rc, txt = run(mut, make(tmp_path, mut))
    assert "| A | 2022 | 0.6000 | 0.5000 | 6.0000 |" not in txt
    assert "-0.6000" in txt
