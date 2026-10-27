"""Скрипт открытия теста главы 4 (scripts/otkrytie_gl4.py) — на синтетике во временной папке.

Реальные данные ≥ 2022, test_locked/ и docs/final_gl4/spec_otkrytiya.json не читаются: корень кода, MAIN, журнал,
замок и выходы — tmp_path. Код открытия (daily_1850, s9_stats, sweep_drivers, сам скрипт) копируется в поддельный
корень побайтно, оценщик и проверка сдвига — синтетические (fastbt/sint.py).
    python3 -m pytest -q tests/test_otkrytie_gl4.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import otkrytie_gl4 as O  # noqa: E402

TIKERY = ["AAA", "BBB"]
OKNO = {"s": "2022-01-03", "po": "2022-01-31"}

SINT = '''"""Синтетический оценщик и проверка сдвига для тестов открытия (контракт sweep_evaluate v1.1, period = test)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STUP = {"-": 0, "konst": 0, "naivnaya": 1, "harj_b": 2, "gibrid": 3}


def _zamok(ctx):
    z = json.loads(Path(ctx["zamok"]).read_text())
    if z.get("status") != "начато":
        raise PermissionError("test без замка «начато»")


def evaluate(obj, params, period, commission):
    if period != "test":
        raise PermissionError(period)
    ctx = obj["test"]
    _zamok(ctx)
    d = pd.read_parquet(ctx["ryad_1850"])
    ok = ctx["okno"]
    d = d[(d["date"] >= ok["s"]) & (d["date"] <= ok["po"])]
    px = d.pivot(index="date", columns="ticker", values="close").sort_index()
    r = np.log(px).diff().mean(axis=1).fillna(0.0).values
    key = json.dumps([obj.get("kind"), obj.get("sigma"), str(obj.get("kontrol")), params, commission], sort_keys=True)
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    x = 0.5 * r + 0.0004 * STUP[obj.get("sigma") or "-"] - commission / 100 * 0.2 + rng.normal(0, 0.002, len(r))
    sd = x.std(ddof=1)
    m = x - x.mean()
    m2 = (m ** 2).mean()
    k = np.concatenate([[1.0], np.cumprod(1 + x)])
    mdd = float(np.max(1 - k / np.maximum.accumulate(k)))
    daty = [str(t.date()) for t in px.index]
    row = {"v": "1.1", "sharpe": float(x.mean() / sd * 252 ** 0.5), "by_year": {2022: float(x.mean() / sd * 252 ** 0.5)},
           "sr_d": float(x.mean() / sd), "T": len(x), "skew": float((m ** 3).mean() / m2 ** 1.5),
           "kurt": float((m ** 4).mean() / m2 ** 2), "msr": float("nan"), "sortino": 1.0, "mdd": mdd,
           "kalmar": float(x.mean() * 252 / mdd) if mdd > 0 else float("nan"), "oborot_y": 10.0,
           "dni_v_rynke": 0.5, "sdelok_y": 20.0, "period": "test", "okno_s": ok["s"], "okno_po": ok["po"],
           "chestnaya": "test", "comm_pct": commission, "excess_d": x.tolist(), "daty": daty}
    if obj.get("kind") == "E":
        row["dolya_ispolneniy"] = 0.7
    if isinstance(obj.get("kontrol"), dict):
        row["p_k2"], row["n_k2"] = 0.5, obj["kontrol"]["k2"]["n"]
    return row


def evaluate_sboy(obj, params, period, commission):
    raise RuntimeError("сбой оценщика")


def proverka(ctx):
    s = pd.read_parquet(ctx["vhody"]["sigma"])
    return [{"ryad": "sigma", "ok": bool(len(s) > 0), "argmax": 0}]


def proverka_plohaya(ctx):
    pd.read_parquet(ctx["vhody"]["sigma"])
    return [{"ryad": "sigma", "ok": False, "argmax": 1}]
'''


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def svechi(t: str, seed: int) -> pd.DataFrame:
    """10-минутки: на день бары 10:00, 18:40 и вечерний 19:00 (с ценой-ловушкой); дни 01.12.2021–28.02.2022."""
    rng = np.random.default_rng(seed)
    rows, p = [], 100.0
    for d in pd.bdate_range("2021-12-01", "2022-02-28"):
        p *= float(np.exp(rng.normal(0, 0.01)))
        for hh, mm, c in ((10, 0, p * 0.999), (18, 40, p), (19, 0, p * 5.0)):
            rows.append((d + pd.Timedelta(hours=hh, minutes=mm), c, c * 1.001, c * 0.999, c, 100))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


class Mir:
    """Поддельные корень кода и MAIN; spec собирается и замораживается заново после каждой правки."""

    def __init__(self, tmp: Path):
        self.root, self.main, self.out = tmp / "root", tmp / "main", tmp / "out"
        for p in ("fastbt/daily_1850.py", "fastbt/s9_stats.py", "fastbt/sweep_drivers.py"):
            (self.root / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / p, self.root / p)
        (self.root / "scripts").mkdir(parents=True)
        shutil.copyfile(ROOT / "scripts" / "otkrytie_gl4.py", self.root / "scripts" / "otkrytie_gl4.py")
        (self.root / "fastbt" / "sint.py").write_text(SINT)
        self.final_b = self.root / "docs" / "final_gl4" / "FINAL_B.md"
        self.final_b.parent.mkdir(parents=True)
        self.final_b.write_text("# FINAL-B (поддельный)\n")
        self.geit = self.root / "docs" / "kanon" / "v24" / "gates" / "E9_test.md"
        self.geit.parent.mkdir(parents=True)
        self.geit.write_text("# GATES\n\n## G1 — x\nCHECK: true\nEXPECT: 0\nEVIDENCE: ok (код 0)\n\n"
                             "## ABANDON (только с причиной; гейт не удалять и не переписывать)\n"
                             "ABANDON: <id> <причина>\n\n## Реестр\nзакрыто 1 из 1; ABANDON: 0; pending: 0\n")
        kat = self.main / "candles_10m"
        kat.mkdir(parents=True)
        for i, t in enumerate(TIKERY):
            svechi(t, i).to_parquet(kat / f"{t}.parquet", index=False)
        (self.main / "vhody").mkdir()
        pd.DataFrame({"date": pd.bdate_range("2022-01-03", "2022-01-31"), "sigma": 0.02}).to_parquet(
            self.main / "vhody" / "sigma.parquet", index=False)
        self.zhurnal = self.root / "docs" / "resheniya" / "resheniya.md"
        self.zhurnal.parent.mkdir(parents=True)
        self.da = "Владелец 28.10, 12:00: да — открывайте тест главы 4 по FINAL-B."
        self.zhurnal.write_text("# Журнал\n\n### Ж-998 · 28.10.2026, 11:00 · другое\n\n> «нет»\n\n"
                                f"### Ж-999 · 28.10.2026, 12:00 · «да» на открытие\n\n> «{self.da}»\n")
        self.P = O.Puti(root=self.root, main=self.main, spec=self.root / "docs" / "final_gl4" / "spec_otkrytiya.json",
                        zhurnal=self.zhurnal, schema=ROOT / "docs" / "final_gl4" / "chisla_gl4.schema.json",
                        lock=self.main / "test_locked" / "gl4_b" / "OPENED_GL4_B.json", out_data=self.out / "data",
                        out_tables=self.out / "tables", out_json=self.out / "chisla_gl4.json")
        fam = {"ocenshchik": "fastbt/sint.py:evaluate", "N": 50, "var_sr_d": 0.0004,
               "lestnica_sigma": list(O.LESTNICA),
               "kontroli": {"k2": {"n": 1000, "seed": 20261027, "sposob": "sdvig_po_krugu",
                                   "stupeni": ["naivnaya", "harj_b", "gibrid"]}, "k3": True, "k3_sigma": "gibrid"}}
        semi = {s: dict(fam, tochki=[{"id": f"{s}-1", "obj": {"w": 20}, "params": {"p": 0.6}}]) for s in O.SEMI}
        semi["C"]["tochki"].append({"id": "C-2", "obj": {"w": 60}, "params": {"p": 0.7}})
        self.spec = {
            "v": "1", "status": "zamorozheno", "zamorozheno_v": "2026-10-28 11:30",
            "final_b": {"put": "docs/final_gl4/FINAL_B.md", "sha256": ""},
            "kod": {}, "fastbt_sha_vse": "",
            "geity": {"docs/kanon/v24/gates/E9_test.md": ""},
            "vhody": {"sigma": {"put": "vhody/sigma.parquet", "sha256": ""}},
            "svechi_10m": {"katalog": "candles_10m", "tikery": TIKERY, "sha256": {t: "" for t in TIKERY}},
            "okno_testa": dict(OKNO), "lag_ispolneniya_dney": 1, "komissii_pct": [0.035, 0.04],
            "pometka": "после аудита", "lw": {"reps": 30, "block": 5, "seed": 42},
            "proverka_sdviga": "fastbt/sint.py:proverka",
            "baza": {"ocenshchik": "fastbt/sint.py:evaluate", "obj": {"kind": "A_sborka"}, "params": {}, "N": 36,
                     "var_sr_d": 0.0003},
            "semi": semi,
            "gipotezy": {
                "H2a": {"formulirovka": "гибрид лучше константной σ в C-1", "a": "C/C-1/gibrid/0.04/osnova",
                        "b": "C/C-1/konst/0.04/osnova", "metrika": "sharpe", "znak": "+", "alfa": 0.05},
                "H3": {"formulirovka": "E лучше C", "a": "E/E-1/gibrid/0.04/osnova", "b": "C/C-1/gibrid/0.04/osnova",
                       "metrika": "sharpe", "znak": "+", "alfa": 0.05}},
            "da_vladeltsa": {"stroka": self.da, "data": "2026-10-28 12:00", "zhurnal": "Ж-999", "sha_spec": ""},
        }
        self.zamorozit()

    def zamorozit(self, da=True):
        s = self.spec
        s["final_b"]["sha256"] = _sha(self.final_b)
        s["kod"] = {p: _sha(self.root / p) for p in sorted(O.moduli(s))}
        s["fastbt_sha_vse"] = O.fastbt_agg(self.root)
        s["geity"] = {p: _sha(self.root / p) for p in s["geity"]}
        for v in s["vhody"].values():
            v["sha256"] = O.sha_put(self.main / v["put"])
        s["svechi_10m"]["sha256"] = {t: _sha(self.main / "candles_10m" / f"{t}.parquet") for t in TIKERY}
        if da:
            s["da_vladeltsa"]["sha_spec"] = O.sha_spec_bez_da(s)
        self.P.spec.write_text(json.dumps(s, ensure_ascii=False, indent=1))

    def zapisat_bez_zamorozki(self):
        self.P.spec.write_text(json.dumps(self.spec, ensure_ascii=False, indent=1))


@pytest.fixture
def mir(tmp_path):
    return Mir(tmp_path)


def stop(P, msg):
    with pytest.raises(O.Stop) as e:
        O.run_test(P)
    assert e.value.code == 2
    assert msg in e.value.msg, e.value.msg
    return e.value.msg


def kolonki(p: Path) -> tuple:
    lines = p.read_text().splitlines()
    shapka = [x for x in lines if x.startswith("#")]
    dannye = [x for x in lines if x and not x.startswith("#")]
    return shapka, dannye[0].split(","), dannye[1:]


# ---------------------------------------------------------------- полный прогон, схема, без выбора

def test_shema_polnyy_progon_prohodit_shemu(mir):
    assert O.run_test(mir.P) == 0
    doc = json.loads(mir.P.out_json.read_text())
    shema = json.loads(mir.P.schema.read_text())
    assert O.proverit_shemu(doc, shema) == []
    assert doc["v"] == "1" and doc["status"] == "тест" and doc["meta"]["chestnaya"] == "test"
    assert doc["meta"]["okno_testa"] == OKNO and doc["meta"]["lag_ispolneniya_dney"] == 1
    z = json.loads(mir.P.lock.read_text())
    assert z["status"] == "завершено" and z["vyhod_sha256"]["json"] == _sha(mir.P.out_json)
    zh = [json.loads(x) for x in (mir.P.out_tables / "zhurnal_sha_gl4.jsonl").read_text().splitlines()]
    assert {x["etap"] for x in zh} == {"do_zamka", "posle_zamka", "vyhod"}
    assert any(x["chto"] == "код scripts/otkrytie_gl4.py" for x in zh)


def test_shema_lovit_lishnee_pole_i_tip(mir):
    O.run_test(mir.P)
    doc = json.loads(mir.P.out_json.read_text())
    shema = json.loads(mir.P.schema.read_text())
    doc["osnovnaya"][0]["vybor"] = True
    doc["meta"]["komissii_pct"] = [0.05]
    doc["v"] = "2"
    err = O.proverit_shemu(doc, shema)
    assert any("vybor" in e for e in err) and any("0.05" in e for e in err) and any("$.v" in e for e in err)


def test_vse_zadaniya_bez_vybora(mir):
    O.run_test(mir.P)
    doc = json.loads(mir.P.out_json.read_text())
    n_toch = sum(len(f["tochki"]) for f in mir.spec["semi"].values())          # 5
    assert len(doc["osnovnaya"]) == 2 + n_toch * 2 * 4                          # база × 2 комиссии + все ступени
    assert len(doc["lestnica"]) == n_toch * 2 * 3
    assert len(doc["kontroli"]) == n_toch * 2 * (3 + 1)
    assert {g["gipoteza"] for g in doc["gipotezy"]} == {"H2a", "H3"}
    ids = {r["id"] for r in doc["osnovnaya"]}
    assert {j[0] for j in O.plan_zadaniy(mir.spec) if j[8] == "osnova"} == ids
    assert {r["komissiya_pct"] for r in doc["osnovnaya"]} == {0.035, 0.04}
    baza = [r for r in doc["osnovnaya"] if r["semya"] == "A′"]
    assert len(baza) == 2 and all(r["delta_k_baze"] is None for r in baza)
    assert all(r["p_lw"] is not None for r in doc["osnovnaya"] if r["semya"] != "A′")
    k2 = [r for r in doc["kontroli"] if r["kontrol"] == "k2"]
    assert all(r["n_k2"] == 1000 and r["seed_k2"] == 20261027 for r in k2)
    assert [r["dolya_ispolneniy"] is not None for r in doc["osnovnaya"]] == [r["semya"] == "E" for r in doc["osnovnaya"]]


def test_komissii_tolko_0035_i_004(mir):
    mir.spec["komissii_pct"] = [0.04, 0.05]
    mir.zamorozit()
    stop(mir.P, "komissii_pct")
    assert not mir.P.lock.exists()


def test_komissii_odna_stavka_otkaz(mir):
    mir.spec["komissii_pct"] = [0.04]
    mir.zamorozit()
    stop(mir.P, "komissii_pct")
    assert not mir.P.lock.exists()


@pytest.mark.parametrize("pravka, msg", [
    (lambda s: s.update(semi={k: v for k, v in s["semi"].items() if k != "D"}), "semi"),
    (lambda s: s["semi"]["B"].update(lestnica_sigma=["konst", "harj_b", "naivnaya", "gibrid"]), "lestnica_sigma"),
    (lambda s: s["semi"]["E"]["kontroli"]["k2"].update(seed=None), "к2"),
    (lambda s: s["semi"]["E"]["kontroli"]["k2"].update(stupeni=["konst"]), "к2"),
    (lambda s: s["semi"]["C"]["kontroli"].update(k3_sigma="xx"), "к3"),
    (lambda s: s.update(status="chernovik"), "status"),
    (lambda s: s.update(lag_ispolneniya_dney=0), "lag"),
    (lambda s: s.update(okno_testa={"s": "2021-12-01", "po": "2022-01-31"}), "okno_testa"),
    (lambda s: s["gipotezy"]["H3"].update(a="E/E-9/gibrid/0.04/osnova"), "гипотеза H3"),
])
def test_komissii_i_struktura_spec_otkaz(mir, pravka, msg):
    pravka(mir.spec)
    mir.zamorozit()
    stop(mir.P, msg)
    assert not mir.P.lock.exists()


# ---------------------------------------------------------------- «да» владельца

@pytest.mark.parametrize("pravka, msg", [
    (lambda m: m.spec["da_vladeltsa"].update(stroka=""), "нет строки «да»"),
    (lambda m: m.spec["da_vladeltsa"].update(stroka="Владелец: подождём"), "нет строки «да»"),
    (lambda m: m.spec["da_vladeltsa"].update(data=""), "без даты"),
    (lambda m: m.spec["da_vladeltsa"].update(data="28.10 12:00"), "без даты"),
    (lambda m: m.spec["da_vladeltsa"].update(data="2026-10-28 11:00"), "раньше заморозки"),
    (lambda m: m.spec["da_vladeltsa"].update(zhurnal="Ж-998"), "нет в журнале"),
    (lambda m: m.spec["da_vladeltsa"].update(stroka=m.da + " (дописано)"), "нет в журнале"),
])
def test_bez_da_otkaz(mir, pravka, msg):
    pravka(mir)
    mir.zapisat_bez_zamorozki()
    stop(mir.P, msg)
    assert not mir.P.lock.exists() and not mir.P.out_json.exists()


def test_bez_da_spec_pravili_posle_da(mir):
    mir.spec["semi"]["C"]["tochki"][0]["params"]["p"] = 0.5     # правка после «да», SHA файлов не меняется
    mir.zapisat_bez_zamorozki()
    stop(mir.P, "менялась после «да»")
    assert not mir.P.lock.exists()


def test_bez_da_net_spec(mir):
    mir.P.spec.unlink()
    stop(mir.P, "нет спецификации")
    assert not mir.P.lock.exists()


# ---------------------------------------------------------------- подмена хэша

@pytest.mark.parametrize("chto", ["scripts/otkrytie_gl4.py", "fastbt/sint.py", "fastbt/daily_1850.py",
                                  "docs/final_gl4/FINAL_B.md", "docs/kanon/v24/gates/E9_test.md"])
def test_podmena_koda_final_b_geitov(mir, chto):
    p = mir.root / chto
    p.write_text(p.read_text() + "\n# подмена\n")
    stop(mir.P, "SHA не сходятся")
    assert not mir.P.lock.exists()


@pytest.mark.parametrize("chto", ["vhody/sigma.parquet", "candles_10m/BBB.parquet"])
def test_podmena_vhoda(mir, chto):
    p = mir.main / chto
    p.write_bytes(p.read_bytes() + b"\0")
    stop(mir.P, "SHA не сходятся")
    assert not mir.P.lock.exists()


def test_podmena_novyy_fayl_fastbt(mir):
    (mir.root / "fastbt" / "novyy.py").write_text("X = 1\n")
    stop(mir.P, "fastbt_sha_vse")
    assert not mir.P.lock.exists()


def test_podmena_ispolnyaemyy_kod_bez_sha(mir):
    mir.spec["kod"].pop("fastbt/s9_stats.py")
    mir.spec["da_vladeltsa"]["sha_spec"] = O.sha_spec_bez_da(mir.spec)
    mir.zapisat_bez_zamorozki()
    stop(mir.P, "исполняемый код без SHA")
    assert not mir.P.lock.exists()


# ---------------------------------------------------------------- гейты

@pytest.mark.parametrize("dop, msg", [("\n## G2 — y\nEVIDENCE: pending\n", "pending 1"),
                                      ("\nABANDON: G1 нет данных\n", "ABANDON 1")])
def test_geity_nezakrytye_otkaz(mir, dop, msg):
    mir.geit.write_text(mir.geit.read_text() + dop)
    mir.zamorozit()                         # SHA гейта сходится — отказ именно по содержанию
    stop(mir.P, msg)
    assert not mir.P.lock.exists()


def test_geity_net_fayla(mir):
    mir.geit.unlink()
    stop(mir.P, "нет файла")
    assert not mir.P.lock.exists()


# ---------------------------------------------------------------- замок до чтения, повтор

def test_zamok_do_chteniya(mir, monkeypatch):
    import pyarrow.parquet as pq
    zhurnal = []
    rt, rp = pq.read_table, pd.read_parquet

    def sostoyanie():
        lk = mir.P.lock
        return json.loads(lk.read_text()).get("status") if lk.exists() else None

    def rt2(src, *a, **k):
        zhurnal.append((str(src), sostoyanie()))
        return rt(src, *a, **k)

    def rp2(src, *a, **k):
        zhurnal.append((str(src), sostoyanie()))
        return rp(src, *a, **k)

    monkeypatch.setattr(pq, "read_table", rt2)
    monkeypatch.setattr(pd, "read_parquet", rp2)
    assert O.run_test(mir.P) == 0
    chteniya = [z for z in zhurnal if str(mir.main) in z[0] or str(mir.out) in z[0]]
    assert any("candles_10m" in z[0] for z in chteniya) and any("sigma.parquet" in z[0] for z in chteniya)
    assert all(s == "начато" for _, s in chteniya), chteniya


def test_povtor_zapreshchen(mir):
    assert O.run_test(mir.P) == 0
    s0 = _sha(mir.P.out_json)
    stop(mir.P, "уже есть")
    assert _sha(mir.P.out_json) == s0
    assert json.loads(mir.P.lock.read_text())["status"] == "завершено"


def test_povtor_posle_sboya_zapreshchen(mir):
    mir.spec["semi"]["D"]["ocenshchik"] = "fastbt/sint.py:evaluate_sboy"
    mir.zamorozit()
    with pytest.raises(RuntimeError, match="сбой оценщика"):
        O.run_test(mir.P)
    assert json.loads(mir.P.lock.read_text())["status"] == "прервано"
    mir.spec["semi"]["D"]["ocenshchik"] = "fastbt/sint.py:evaluate"
    mir.zamorozit()
    stop(mir.P, "уже есть")


def test_povtor_zamok_o_excl(mir):
    O.acquire_lock(mir.P, {"x": 1})
    with pytest.raises(O.Stop):
        O.acquire_lock(mir.P, {"x": 2})
    assert json.loads(mir.P.lock.read_text())["x"] == 1


def test_proverka_sdviga_otkaz_pod_zamkom(mir):
    mir.spec["proverka_sdviga"] = "fastbt/sint.py:proverka_plohaya"
    mir.zamorozit()
    stop(mir.P, "проверка сдвига")
    assert json.loads(mir.P.lock.read_text())["status"] == "отказ: проверка сдвига"
    assert not mir.P.out_json.exists()


# ---------------------------------------------------------------- ряд 18:50

def test_ryad_1850_bez_vechera_i_po_konec_okna(mir):
    O.run_test(mir.P)
    d = pd.read_parquet(mir.P.out_data / "ohlcv_daily_1850_test.parquet")
    assert d["date"].max() <= pd.Timestamp(OKNO["po"])
    assert d["date"].min() < pd.Timestamp("2022-01-01")           # прогрев из истории
    src = svechi("AAA", 0)
    bar = src[src["timestamp"] == pd.Timestamp("2022-01-10 18:40")]["close"].iloc[0]
    got = d[(d["ticker"] == "AAA") & (d["date"] == pd.Timestamp("2022-01-10"))]["close"].iloc[0]
    assert got == bar                                              # close = 18:40, вечерний 19:00 (×5) не вошёл
    doc = json.loads(mir.P.out_json.read_text())
    assert doc["ryad_1850"]["sha256"] == _sha(mir.P.out_data / "ohlcv_daily_1850_test.parquet")


# ---------------------------------------------------------------- заготовки, схема, шаблон

def test_zagotovki_shapki_bez_chisel_i_kolonki_skripta():
    kat = ROOT / "docs" / "final_gl4" / "zagotovki"
    for name, cols in O.COLS.items():
        shapka, kol, dannye = kolonki(kat / f"t_gl4_test_{name}.csv")
        assert "# статус: тест" in shapka, name
        assert any(x.startswith("# окно данных:") and "d+1" in x and "честная половина" in x for x in shapka), name
        assert any(x.startswith("# код:") and "хэш входов" in x for x in shapka), name
        assert kol == cols, name
        assert dannye == [], name


def test_zagotovki_kolonki_sovpadayut_s_vyhodom(mir):
    O.run_test(mir.P)
    for name, cols in O.COLS.items():
        shapka, kol, dannye = kolonki(mir.P.out_tables / f"t_gl4_test_{name}.csv")
        z_shapka, z_kol, _ = kolonki(ROOT / "docs" / "final_gl4" / "zagotovki" / f"t_gl4_test_{name}.csv")
        assert kol == z_kol and len(shapka) == len(z_shapka) and dannye
        assert shapka[0] == z_shapka[0] and shapka[1] == "# статус: тест"


def test_shema_v_i_kolonki():
    shema = json.loads((ROOT / "docs" / "final_gl4" / "chisla_gl4.schema.json").read_text())
    assert shema["properties"]["v"]["const"] == O.V
    d = shema["$defs"]
    for name, ref in (("osnovnaya", "stroka_osnovnaya"), ("lestnica", "stroka_lestnica"),
                      ("kontroli", "stroka_kontrol"), ("gipotezy", "stroka_gipoteza"), ("po_godam", "stroka_god")):
        assert d[ref]["required"] == O.COLS[name], name


def test_shablon_spec_otkazyvaet():
    s = json.loads((ROOT / "docs" / "final_gl4" / "zagotovki" / "spec_otkrytiya.shablon.json").read_text())
    with pytest.raises(O.Stop) as e:
        O.check_spec(s)
    assert "status" in e.value.msg


def test_proverki_bez_zamka_i_bez_dannyh(mir, capsys):
    mir.spec["da_vladeltsa"].update(stroka="", data="")
    mir.zapisat_bez_zamorozki()
    assert O.run_proverki(mir.P) == 0
    out = capsys.readouterr().out
    assert "«да»: нет" in out and O.sha_spec_bez_da(mir.spec) in out
    assert not mir.P.lock.exists() and not mir.P.out_data.exists()


def test_holm():
    assert O.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_main_bez_scope_otkaz():
    with pytest.raises(O.Stop):
        O.main([])
