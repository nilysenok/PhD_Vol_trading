"""Фабрика кандидатов validation A′ и заморозка (Ж-031 Э1-д, Ж-037). Таблицы итераций — реальные (train), данные не читаются;
git и журнал — во временном каталоге. Детекторы `ok_*` возвращают True, если свойство выполнено; мутации — `test_sweep_val_kand_mut.py`."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from fastbt import sweep_val as sv
from fastbt import sweep_val_kand as kd

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True, env=GIT_ENV)


@pytest.fixture(scope="module")
def spisok():
    return kd.build_spisok(".")


def repo_s_zamorozkoy(tmp, spisok_):
    """Временный репозиторий с закоммиченным файлом заморозки + журнал; (P, zamorozka-путь)."""
    git(tmp, "init", "-q")
    P = sv.Puti(koren=tmp)
    P.zamorozka.parent.mkdir(parents=True)
    sp = spisok_[0]
    P.zamorozka.write_text(json.dumps(dict(zamorozheno_v="2026-10-28 09:00", sha=sv.sha_kandidatov(sp), N_vsego={}, bazy=spisok_[1],
                                           spisok=sp), sort_keys=True), "utf-8")
    git(tmp, "add", "-A")
    git(tmp, "commit", "-qm", "z")
    return P


def ok_spisok_sostav(spisok_) -> bool:
    sp = spisok_[0]
    ids = [m["id"] for m in sp]
    return bool(len(sp) == 24 and len(set(ids)) == 24 and sum(m["kind"] == "S" for m in sp) == 18
                and sorted(spisok_[2]) == ["A", *kd.SIDS] and all(len(m["tochki"]) == (1 if m["kind"] == "S" else 6) for m in sp))


def ok_dubli_svedeny(sid="S5") -> bool:
    """Синтетика: в пул kand_a.csv вставлена копия ранга 1 с другим значением НЕАКТИВНОГО параметра (подпись результата та же).
    Дубль сведён: в списке три различные точки, копии нет, добран следующий различный ранг."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    t = pd.read_csv(Path(kd.TABLES) / "kand_a.csv", comment="#")
    pool = t[t["strategiya"] == sid].sort_values("rank").reset_index(drop=True)
    dup = pool.iloc[[0]].copy()
    dup["p.rm_sqrt_rv.k_sl"] = 99.0                                    # неактивен при rm_tip != sqrt_rv
    new = pd.concat([pool.iloc[[0]], dup, pool.iloc[1:]], ignore_index=True)
    new["rank"] = range(1, len(new) + 1)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / kd.TABLES).mkdir(parents=True)
        new.to_csv(Path(d) / kd.TABLES / "kand_a.csv", index=False)
        c = kd.kandidaty(sid, d)["cands"]
    keys = {kd.eff_key(sid, x["point"]) for x in c}
    return bool(len(c) == 3 and len(keys) == 3 and 2 not in [x["rank"] for x in c] and [x["rank"] for x in c] == [1, 3, 4])


def ok_otkaz_posle_kommita(tmp, spisok_) -> bool:
    """Файл правлен после коммита (sha пересчитан под правку!) — `load_zamorozka` отказывает."""
    P = repo_s_zamorozkoy(tmp, spisok_)
    kd.load_zamorozka(P)                                              # как закоммичено — читается
    z = json.loads(P.zamorozka.read_text("utf-8"))
    z["spisok"] = z["spisok"][:-1]
    z["sha"] = sv.sha_kandidatov(z["spisok"])
    P.zamorozka.write_text(json.dumps(z, sort_keys=True), "utf-8")
    try:
        kd.load_zamorozka(P)
    except sv.Stop:
        return True
    return False


def ok_otkaz_bez_da_do_zamka(tmp, monkeypatch, spisok_) -> bool:
    """`sweep_val` (реальный путь, `Puti.default`) без строки «да» в журнале: отказ до замка, реестра и чтения данных."""
    P = sv.Puti(koren=tmp, git_proverka=False)
    P.zamorozka.parent.mkdir(parents=True)
    sp = spisok_[0]
    P.zamorozka.write_text(json.dumps(dict(zamorozheno_v="2026-10-28 09:00", sha=sv.sha_kandidatov(sp), N_vsego={}, bazy=spisok_[1],
                                           spisok=sp)), "utf-8")
    P.zhurnal.parent.mkdir(parents=True)
    P.zhurnal.write_text("### Ж-777 · 28.10.2026, 11:00\nпросто запись, без слова-разрешения\n", "utf-8")
    called, built = [], []
    monkeypatch.setattr(sv.Puti, "default", classmethod(lambda cls, root=None: called.append(1) or P))
    monkeypatch.setattr(kd, "make_factory", lambda P_: built.append(1) or (lambda m, n: built.append("читает данные")))
    try:
        sv.main(["--zhurnal", "Ж-777", "--da", "просто запись, без слова-разрешения"])
    except sv.Stop:
        return bool(called and not P.lock.exists() and not P.reestr.exists() and not P.out.exists() and "читает данные" not in built)
    except Exception:                                                 # не Stop: прогон пошёл дальше проверки «да»
        return False
    return False


def ok_vremya_iz_fayla(tmp) -> bool:
    """Время заморозки позже «да» — только в файле; spec врёт, что заморозка давно: `sweep_val` всё равно отказывает."""
    from fastbt.sweep_val_dry import make_factory, make_meta, make_spec
    meta = make_meta()
    P, spec = make_spec(tmp, meta)
    z = json.loads(P.zamorozka.read_text("utf-8"))
    P.zamorozka.write_text(json.dumps(dict(z, zamorozheno_v="2026-10-28 10:00")), "utf-8")
    spec.update(zamorozheno_v="2000-01-01 00:00", zamorozheno_sha=sv.sha_kandidatov(meta))
    try:
        sv.run(P, spec, meta, make_factory(tmp), reps_lw=50, reps_k2=3)
    except sv.Stop:
        return not P.lock.exists()
    return False


def test_vremya_i_sha_zamorozki_iz_fayla_a_ne_iz_spec(tmp_path):
    assert ok_vremya_iz_fayla(tmp_path)


def ok_tablitsa_sborok(t) -> bool:
    """36 сборок + якорь + A на новых точках; top2 — ровно две лучшие по цели при плато ≥ 3."""
    s = t[t.kind == "sborka"]
    best = s[s.plateau_neighbors >= 3].sort_values("objective", ascending=False).head(2)
    return bool(len(s) == 36 and set(t.kind) == {"sborka", "S_centr_ravnye", "A_boevaya_s9", "A_novye_ravnye"} and (t.top2 > 0).sum() == 2
                and list(s[s.top2 > 0].sort_values("top2").index) == list(best.index))


TRIALS = [0.01 * ((i * 37) % 23) - 0.08 for i in range(40)]


def ok_dsr_ne_zavisit_ot_poryadka(f=None) -> bool:
    """Перестановка ledger не меняет DSR (N=100 > числа проб, как N_vsego > новых). f(sr, trials, n, T, skew, kurt) -> {dsr}."""
    from fastbt import sweep_val_stat as vs
    f = f or vs.dsr_n
    a = f(0.1, TRIALS, 100, 250, -0.2, 4.0)["dsr"]
    b = f(0.1, TRIALS[::-1], 100, 250, -0.2, 4.0)["dsr"]
    c = f(0.1, TRIALS[7:] + TRIALS[:7], 100, 250, -0.2, 4.0)["dsr"]
    return bool(a == b == c)


def test_dsr_ne_zavisit_ot_poryadka_ledger():
    assert ok_dsr_ne_zavisit_ot_poryadka()


def test_dsr_N_otdelno_var_po_novym():
    import statistics
    from fastbt import sweep_val_stat as vs
    r1, r2 = vs.dsr_n(0.1, TRIALS, 100, 250, 0.0, 3.0), vs.dsr_n(0.1, TRIALS, 400, 250, 0.0, 3.0)
    assert r1["n_trials"] == 100 and abs(r1["var_sr"] - statistics.variance(TRIALS)) < 1e-15 and r1["var_sr"] == r2["var_sr"]
    assert r2["dsr"] < r1["dsr"]                                          # N растёт при той же дисперсии — DSR падает


def test_N_dlya_A_i_Sk_iz_zamorozki(spisok):
    n = spisok[2]
    z = json.loads(Path("docs/otbor/zamorozka_A_prim.json").read_text("utf-8"))["N_vsego"]       # числа — из файла заморозки
    assert n == z and kd.n_a(n) == 36 + sum(z[s] for s in kd.SIDS)
    assert len(kd.ledger_sk("S1", ".")[0]) == 392 < n["S1"]               # новые пробы без дополнения; N — отдельно


def test_tablitsa_sborok_A():
    import pandas as pd
    t = pd.read_csv("docs/tables/sweep_e1v2/A_sborki.csv", comment="#")
    assert ok_tablitsa_sborok(t)


def test_realnyy_fayl_zamorozki_zakommichen_i_raven_fabrike(spisok):
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if "baza" not in json.loads((root / "docs/otbor/zamorozka_A_prim.json").read_text("utf-8"))["spisok"][0]:
        pytest.xfail("заморозка со старыми базами; перезаморозка отложена до сверки оценщика с s9 (Ж-040)")
    z = kd.load_zamorozka(sv.Puti(koren=root))                       # с git-проверкой: файл в коммите и не правлен
    assert z["spisok"] == json.loads(json.dumps(spisok[0])) and z["sha"] == sv.sha_kandidatov(spisok[0])
    assert z["N_vsego"] == spisok[2] and len(z["spisok"]) == 24


def test_sostav_spiska(spisok):
    assert ok_spisok_sostav(spisok)


def test_fabrika_determinirovana(spisok):
    again = kd.build_spisok(".")
    assert again == spisok and sv.sha_kandidatov(again[0]) == sv.sha_kandidatov(spisok[0])


def test_dubli_svedeny_S5():
    assert ok_dubli_svedeny()


def test_tochki_sovpadayut_s_tablitsami(spisok):
    top = kd.kandidaty("S1")["cands"][0]["point"]
    m = next(x for x in spisok[0] if x["id"] == "S1:1")
    assert m["tochki"]["S1"] == top and m["baza"]["id"] == "S1" and m["baza"]["konfig"]["semya"] == "S1"


def test_freeze_pishet_sha_vremya_N(tmp_path):
    z = kd.freeze(".", tmp_path / "z.json")
    f = json.loads((tmp_path / "z.json").read_text("utf-8"))
    assert f["sha"] == sv.sha_kandidatov(f["spisok"]) == z["sha"] and len(f["spisok"]) == 24
    assert f["zamorozheno_v"][:4] == "2026" and all(f["N_vsego"][s] > 100 for s in kd.SIDS) and f["N_vsego"]["A"] == 36
    with pytest.raises(FileExistsError):
        kd.freeze(".", tmp_path / "z.json")


def test_pravka_posle_kommita_otkaz(tmp_path, spisok):
    assert ok_otkaz_posle_kommita(tmp_path, spisok)


def test_pravka_spiska_bez_sha_otkaz(tmp_path, spisok):
    P = repo_s_zamorozkoy(tmp_path, spisok)
    z = json.loads(P.zamorozka.read_text("utf-8"))
    P.zamorozka.write_text(json.dumps(dict(z, spisok=z["spisok"][:-1])), "utf-8")
    with pytest.raises(sv.Stop):
        kd.load_zamorozka(P)


def test_nezakommichen_otkaz(tmp_path, spisok):
    P = repo_s_zamorozkoy(tmp_path, spisok)
    git(tmp_path, "rm", "-q", "--cached", str(P.zamorozka))
    with pytest.raises(sv.Stop):
        kd.load_zamorozka(P)


def test_sweep_val_bez_da_otkaz_do_zamka(tmp_path, monkeypatch, spisok):
    assert ok_otkaz_bez_da_do_zamka(tmp_path, monkeypatch, spisok)


def test_sweep_val_bez_fayla_zamorozki_otkaz(tmp_path):
    with pytest.raises(sv.Stop):
        kd.load_zamorozka(sv.Puti(koren=tmp_path, git_proverka=False))
