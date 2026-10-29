from test_r1_svodka_div import _p, load

SHAPKA = "# реверс раунд 1, 2022–2024, справочно, без выбора\n# коммит: abc\n# вход: x sha256=ff\n"


def kornevoj(tmp_path, sv, imya, others=()):
    tb = tmp_path / "docs/tables/revers_r1"
    tb.mkdir(parents=True)
    (tmp_path / "docs/otchety/revers_r1").mkdir(parents=True)
    for k in sv.LISTY:
        (tmp_path / f"docs/otchety/revers_r1/{k}.md").write_text(f"сдвиг {k}\n", encoding="utf-8")
    for nm in (imya, *others):
        (tb / nm).write_text(SHAPKA + "k,v\na,1\n", encoding="utf-8")
    return tmp_path


def prognat(tmp_path, sv, imya, others=()):
    root = kornevoj(tmp_path, sv, imya, others)
    sv.sobrat(root, root / "out.md")
    return (root / "out.md").read_text(encoding="utf-8")


def test_diag_otnositsya_k_listu_d(tmp_path):
    sv = load()
    t = prognat(tmp_path, sv, "diag_tochki_S1.csv")
    assert "- д) " + sv.PREREG["d"] + ": есть" in t
    assert "Вне предрегистрации (не удалены):\n- нет" in t


def test_neizvestnyj_prefiks_vne_preregistracii(tmp_path):
    sv = load()
    t = prognat(tmp_path, sv, "diag_tochki_S1.csv", ("zzz_x.csv",))
    assert "Вне предрегистрации (не удалены):\n- `zzz_x.csv`" in t
    assert "diag_tochki_S1.csv`\n" not in t.split("Вне предрегистрации")[1].split("##")[0]


def test_sverka_train4_i_net_ne_tablicy_lista(tmp_path):
    sv = load()
    root = kornevoj(tmp_path, sv, "diag_kontrol_S2.csv")
    tb = root / "docs/tables/revers_r1"
    (tb / "_sverka_train4").mkdir()
    (tb / "_sverka_train4/diag_tochki_S1.csv").write_text(SHAPKA + "k,v\na,1\n", encoding="utf-8")
    sv.sobrat(root, root / "out.md")
    t = (root / "out.md").read_text(encoding="utf-8")
    assert "_sverka_train4" not in t


def test_mutaciya_diag_ne_priznan_lovitsya(tmp_path):
    src = _p.read_text(encoding="utf-8")
    bad = src.replace('("d_", "diag_")', '("d_",)')
    assert bad != src
    sv = load(bad)
    t = prognat(tmp_path, sv, "diag_tochki_S1.csv")
    assert "- д) " + sv.PREREG["d"] + ": нет" in t
    assert "- `diag_tochki_S1.csv`" in t
