"""Эталоны пакета переноса считаются для периода пакета (лист wfП-4): perenos_do из VERSIYA -> train4 | wf.
train4 (пусто | 2021-12-31) — записи побайтно прежние, ключа period в meta нет; wf (2026-02-28) — якорь A_fix × 6 Sk и набор v6 на wf.
Оценщик — синтетика (даты 2022–2026 для wf, 2016 для train4); данных и сети нет, реальный счёт не запускается.
Мутации: «wf считает эталон train4» (оценщик wf отказывает на train4, как зона данных) и «период не передан на сервер» ловятся."""
import hashlib
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/server"))
import ekviv  # noqa: E402
import ekviv_io  # noqa: E402
import ekviv_v6 as e6  # noqa: E402
import etalony_mak  # noqa: E402
import proverka_ekviv  # noqa: E402
from fastbt import period_v2  # noqa: E402
from fastbt import progon_sobol as ps  # noqa: E402

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos/proverka_yakor.sh"
ARGS = ["--tochek", "6", "--tochek-v6", "12", "--procs", "1"]


class Fake:
    """Синтетика: ряд — функция (точка, период). Период, не равный ожидаемому, — PermissionError (как зона данных в fastbt.data)."""
    fdr = "t0d"
    vidennye: list = []

    def __init__(self, ozhid):
        self.n, self.ozhid = 0, ozhid

    def full(self, d):
        return dict(d, a=1)

    def evaluate_v2(self, pt, period):
        if period != self.ozhid:
            raise PermissionError(f"период {period}: данные ≥ 2022 закрыты")
        Fake.vidennye.append(period)
        self.n += 1
        h = zlib.crc32(repr((period, sorted((str(k), str(v)) for k, v in pt.items()))).encode())
        if self.n > 1 and h % 4 == 0:
            raise ValueError("синтетическая ошибка точки")
        rng = np.random.default_rng(h)
        p = rng.normal(0, 0.01, 6)
        return {"status": "ok", "sharpe": float(p.mean() / p.std()), "god": {"2016": float(p[0])}}, p, rng.normal(0, 0.1, 3)


def _sreda(monkeypatch, period):
    daty = np.array(["2022-01-03", "2026-02-27"], dtype="datetime64[D]") if period == "wf" else \
        np.arange("2016-01-04", "2016-01-10", dtype="datetime64[D]")
    ps._W.clear()
    Fake.vidennye = []
    monkeypatch.setattr(ps, "fabrika", lambda sid: ps.Sreda(Fake(period), "kod1", f"vh_{sid}", daty, ["AFLT"]))
    monkeypatch.setattr(ps, "fast_ok", lambda pt: False)
    monkeypatch.setattr(period_v2, "proverit_chtenie", lambda *a, **k: None)      # отметка владельца — вне теста


@pytest.fixture(autouse=True)
def _chistka():
    yield
    ps._W.clear()


def _sobrat(tmp_path, do, name="e.npz"):
    f = str(tmp_path / name)
    assert etalony_mak.main(["--out", f, "--do", do, *ARGS]) == 0
    return f


def _otpechatok(f):
    """sha256 по всем записям и meta без времени создания: «побайтно» для сравнения двух сборок."""
    rez, meta = ekviv_io.zagruzit(f)
    h = hashlib.sha256()
    for m in sorted(rez):
        r = rez[m]
        h.update(m.encode() + r["sig"].encode() + r["disk"].encode() + repr(sorted(r["hex"].items())).encode())
        [h.update(np.ascontiguousarray(r[k]).tobytes()) for k in ekviv_io.MASSIVY]
    h.update(repr(sorted((k, v) for k, v in meta.items() if k != "sozdan")).encode())
    return h.hexdigest()


def test_period_po_do():
    assert [ekviv.period_po_do(x) for x in ("", None, "2021-12-31", "2026-02-28")] == ["train4", "train4", "train4", "wf"]
    for plokho in ("2024-12-31", "2026-03-01", "2022-01-01"):
        with pytest.raises(ValueError):
            ekviv.period_po_do(plokho)


def test_train4_pobajtno_prezhniy(monkeypatch, tmp_path):
    _sreda(monkeypatch, "train4")
    s_do = _sobrat(tmp_path, "2021-12-31", "a.npz")
    ps._W.clear()
    pusto = _sobrat(tmp_path, "", "b.npz")
    ps._W.clear()
    rez, meta = ekviv.poschitat(ekviv.vybor(6), 1)               # прежний вызов без периода
    rez6, m6, kod6 = e6.poschitat(None, set(), 1, 12)
    e6.dobavit(rez, meta, rez6, m6, kod6)
    eta, meta_eta = ekviv_io.zagruzit(s_do)
    assert _otpechatok(s_do) == _otpechatok(pusto) and "period" not in meta_eta and "period" not in meta
    assert set(eta) == set(rez) and all(eta[m]["sig"] == rez[m]["sig"] and eta[m]["p"].tobytes() == rez[m]["p"].tobytes() for m in rez)
    assert {x for x in Fake.vidennye} == {"train4"}


def test_wf_schitaetsya_yakor_i_v6(monkeypatch, tmp_path):
    _sreda(monkeypatch, "wf")
    f = _sobrat(tmp_path, "2026-02-28")
    rez, meta = ekviv_io.zagruzit(f)
    assert meta["period"] == "wf" and set(Fake.vidennye) == {"wf"}
    assert sum(m.startswith("yakor__") for m in rez) == 6 and not any(m.startswith("tochka__") for m in rez)
    assert sum(m.startswith("v6yakor__") for m in rez) == 6 and sum(m.startswith("v6tochka__") for m in rez) == 12
    assert all('"period": "wf"' in r["sig"] for m, r in rez.items() if m.startswith("v6tochka__"))


def test_wf_etalon_sveryaetsya_na_servere(monkeypatch, tmp_path):
    _sreda(monkeypatch, "wf")
    f = _sobrat(tmp_path, "2026-02-28")
    ps._W.clear()
    args = ["--etalony", f, "--kod-sha", "k", "--dannye-sha", "d", "--procs", "1", "--otmetka", str(tmp_path / "pr/OK")]
    assert proverka_ekviv.main([*args, "--do", "2026-02-28"]) == 0 and (tmp_path / "pr/OK").exists()


def test_do_iz_okruzheniya(monkeypatch, tmp_path):
    _sreda(monkeypatch, "wf")
    monkeypatch.setenv("PERENOS_DO", "2026-02-28")
    f = str(tmp_path / "e.npz")
    assert etalony_mak.main(["--out", f, *ARGS]) == 0 and ekviv_io.zagruzit(f)[1]["period"] == "wf"


def test_mutaciya_wf_schitaet_etalon_train4_lovitsya(monkeypatch, tmp_path):
    _sreda(monkeypatch, "wf")
    monkeypatch.setattr(ekviv, "period_po_do", lambda do: "train4")      # подложено: период пакета проигнорирован
    with pytest.raises((PermissionError, ValueError)):                  # данные wf в зоне train4: отказ, как в fastbt.data
        etalony_mak.main(["--out", str(tmp_path / "e.npz"), "--do", "2026-02-28", *ARGS])


def test_mutaciya_period_ne_peredan_na_server_lovitsya(monkeypatch, tmp_path):
    _sreda(monkeypatch, "wf")
    f = _sobrat(tmp_path, "2026-02-28")
    ps._W.clear()
    otm = tmp_path / "pr/OK"
    args = ["--etalony", f, "--kod-sha", "k", "--dannye-sha", "d", "--procs", "1", "--otmetka", str(otm)]
    for do in ("", "2021-12-31"):                      # --do не передан / передан train4: FAIL дискретного, отметки нет
        assert proverka_ekviv.main([*args, "--do", do]) == 1 and not otm.exists()


def test_proverka_yakor_sh_peredayot_do():
    tekst = SH.read_text(encoding="utf-8")
    vyzov = tekst.split("proverka_ekviv.py", 1)[1].split('> "$log"')[0]          # команда пересчёта целиком
    assert '--do "${PERENOS_DO:-}"' in vyzov
