#!/usr/bin/env python3
"""Одно открытие теста главы 4 по FINAL-B: семьи C, E, B, D против базы A′ (Ж-031 Э4, Ж-036 T4).

Требования — `.claude/rules/otbor_statistika.md` п. 6; образцы — `scripts/analysis/final_test_r2.py` (глава 3),
`scripts/analysis/final_test_ch4_v2.py` (второе открытие главы 4). Ряд 2022+ до 18:50 собирается только здесь,
под замком, функциями `fastbt/daily_1850.py` (толкование Ж-029).

Запуск (только дирижёр, один раз, после «да» владельца):
    python3 scripts/otkrytie_gl4.py --scope test
    python3 scripts/otkrytie_gl4.py --proverki      # шаги 1–5 без замка и без данных: что ещё не готово

Порядок; отказ на шагах 1–5 открытие не тратит (код 2, замок не создаётся):
  1) спецификация `docs/final_gl4/spec_otkrytiya.json` (v = 1): status = zamorozheno, все SHA заполнены, семьи ровно
     C/E/B/D, комиссии ровно {0,035; 0,04} % на сторону, лестница σ ровно konst → naivnaya → harj_b → gibrid, к2 и к3
     объявлены, гипотезы ссылаются на объявленные задания;
  2) «да» владельца: строка со словом «да», дата-время не раньше заморозки, строка дословно стоит в журнале
     `docs/resheniya/resheniya.md` под своим Ж-номером, SHA спецификации без поля «да» совпадает с записанным при «да»;
  3) замок `test_locked/gl4_b/OPENED_GL4_B.json` отсутствует (одно открытие);
  4) гейты из спецификации: файл есть, SHA совпал, нет `EVIDENCE: pending` и строк `ABANDON: <id>`;
  5) SHA-256: FINAL-B, код (каждый файл spec.kod и все *.py/*.yaml fastbt одним агрегатом), входы (файл или каталог),
     10-минутки ряда 18:50. Хэш читает байты файла, значения не разбирает — как в главе 3 и во втором открытии;
     каждый модуль, который скрипт исполняет, обязан стоять в spec.kod;
  6) замок пишется (O_CREAT | O_EXCL, 0444) ДО чтения рядов теста;
  7) ряд 18:50 по spec.okno_testa.po — `daily_1850.read_bars` + `daily_1850.aggregate` (история до теста — прогрев);
  8) проверка сдвига каждого ряда (spec.proverka_sdviga); не прошла — отказ, замок «отказ: проверка сдвига»;
  9) все задания одним прогоном, без выбора: база A′ × комиссии; семья × точка × ступень σ × комиссия (основа);
     к2 (объявленные ступени) и к3 × комиссии; Δ к A′, лестница, к3 — Ледуа–Вольф `fastbt/s9_stats.lw_sharpe_diff`;
     PSR/DSR — `fastbt/sweep_drivers.psr` / `expected_max_sharpe` с N и V из FINAL-B; гипотезы — Холм по всем;
 10) таблицы (шапки = `docs/final_gl4/zagotovki/`), JSON чисел (схема `chisla_gl4.schema.json` проверяется до записи),
     журнал SHA; замок дописывается: «завершено» и SHA выходов. Исключение после замка — замок «прервано: …»,
     повтор запрещён (решает владелец).

Оценщик семьи — `<файл>.py:<функция>` из спецификации, контракт `docs/kanon/v24/kontrakty/sweep_evaluate.md`:
`f(obj, params, period, commission) -> row` с period = "test"; obj несёт `kind`, `sigma`, `kontrol` и `test` —
контекст открытия (замок, путь ряда 18:50, окно, пути входов).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

V = "1"                                                  # версия спецификации и JSON чисел
ROW_V = "1.1"                                            # версия строки оценщика (sweep_evaluate.md)
SEMI = ("C", "E", "B", "D")
KOMISSII = (0.035, 0.04)                                 # % на сторону (Ж-031)
LESTNICA = ("konst", "naivnaya", "harj_b", "gibrid")     # константная → наивная √RV → HAR-J (б) → гибрид
BAZA = "A′"
STATUS = "тест"
ROW_KEYS = ("v", "sharpe", "by_year", "sr_d", "T", "skew", "kurt", "msr", "sortino", "mdd", "kalmar",
            "oborot_y", "dni_v_rynke", "sdelok_y", "period", "okno_s", "okno_po", "chestnaya", "comm_pct", "excess_d")

COLS = {
    "osnovnaya": ["id", "semya", "tochka", "sigma", "komissiya_pct", "sharpe", "msr", "sortino", "kalmar", "mdd", "psr",
                  "dsr", "N", "oborot_y", "dni_v_rynke", "sdelok_y", "dolya_ispolneniy", "delta_k_baze", "se_delta",
                  "p_lw", "T", "okno_s", "okno_po", "chestnaya", "status", "pometka"],
    "lestnica": ["semya", "tochka", "komissiya_pct", "ot", "k", "sharpe_ot", "sharpe_k", "delta", "se_delta", "p_lw",
                 "status"],
    "kontroli": ["semya", "tochka", "sigma", "komissiya_pct", "kontrol", "sharpe_osnova", "sharpe_kontrol", "delta",
                 "se_delta", "p", "sposob_p", "n_k2", "seed_k2", "status"],
    "gipotezy": ["gipoteza", "formulirovka", "a", "b", "metrika", "znak", "delta", "se_delta", "p", "p_holm", "alfa",
                 "vyvod", "status"],
    "po_godam": ["id", "god", "sharpe", "status"],
}
OPIS = {
    "osnovnaya": "основная: база A′ и семьи C/E/B/D, все точки FINAL-B, все ступени σ, обе комиссии",
    "lestnica": "лестница σ: соседние ступени konst → naivnaya → harj_b → gibrid, Δ Шарпа и p Ледуа–Вольфа",
    "kontroli": "контроли без модели: к2 (сдвиг σ̂ по кругу) и к3 (A′ без того, что глушит затвор)",
    "gipotezy": "гипотезы FINAL-B: Δ, p Ледуа–Вольфа, поправка Холма по всем гипотезам",
    "po_godam": "Шарп по годам теста для каждой строки основной таблицы",
}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))


class Stop(SystemExit):
    """Отказ: код 2, сообщение в stderr."""

    def __init__(self, msg):
        super().__init__(2)
        self.msg = msg
        print(f"СТОП: {msg}", file=sys.stderr, flush=True)


@dataclasses.dataclass
class Puti:
    """Все пути одним объектом — тесты подменяют их временной папкой."""
    root: Path = ROOT
    main: Path = MAIN
    spec: Path = ROOT / "docs" / "final_gl4" / "spec_otkrytiya.json"
    zhurnal: Path = ROOT / "docs" / "resheniya" / "resheniya.md"
    schema: Path = ROOT / "docs" / "final_gl4" / "chisla_gl4.schema.json"
    lock: Path = MAIN / "data" / "regen" / "test_locked" / "gl4_b" / "OPENED_GL4_B.json"
    out_data: Path = MAIN / "data" / "fastbt" / "test_gl4_b"
    out_tables: Path = ROOT / "docs" / "tables" / "gl4_test_b"
    out_json: Path = ROOT / "docs" / "final_gl4" / "chisla_gl4.json"


# ======================= общие =======================

def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_put(p: Path) -> str:
    """Файл — SHA-256 байтов; каталог — SHA-256 строк «путь\\0SHA\\n» по всем файлам, отсортированным по пути."""
    p = Path(p)
    if p.is_file():
        return sha(p)
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        h.update(f"{f.relative_to(p).as_posix()}\0{sha(f)}\n".encode())
    return h.hexdigest()


def fastbt_agg(root: Path) -> str:
    """SHA всего кода fastbt: *.py и *.yaml, без __pycache__."""
    base = Path(root) / "fastbt"
    h = hashlib.sha256()
    for f in sorted(x for x in base.rglob("*") if x.suffix in (".py", ".yaml") and "__pycache__" not in x.parts):
        h.update(f"{f.relative_to(base).as_posix()}\0{sha(f)}\n".encode())
    return h.hexdigest()


def sha_spec_bez_da(spec: dict) -> str:
    d = {k: v for k, v in spec.items() if k != "da_vladeltsa"}
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def git_head(root) -> str:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
    return r.stdout.strip() if r.returncode == 0 else ""


def kom(c: float) -> str:
    return f"{c:g}"


def job_id(semya, tochka, sigma, c, kontrol="osnova") -> str:
    return f"{semya}/{tochka}/{sigma}/{kom(c)}/{kontrol}"


def parse_dt(s: str):
    return time.strptime(str(s).strip(), "%Y-%m-%d %H:%M")


# ======================= 1. спецификация =======================

def load_spec(P: Puti) -> dict:
    p = Path(P.spec)
    if not p.exists():
        raise Stop(f"нет спецификации {p} — открытие только после заморозки FINAL-B")
    return json.loads(p.read_text())


def plan_zadaniy(spec: dict) -> list:
    """Все задания одного прогона: (id, ocenshchik, obj, params, c, semya, tochka, sigma, kontrol). Без выбора."""
    out = []
    b = spec["baza"]
    for c in spec["komissii_pct"]:
        out.append((job_id(BAZA, "-", "-", c), b["ocenshchik"], dict(b["obj"]), dict(b.get("params") or {}), c,
                    BAZA, "-", "-", "osnova"))
    for s in SEMI:
        f = spec["semi"][s]
        k2 = f["kontroli"]["k2"]
        for t in f["tochki"]:
            for c in spec["komissii_pct"]:
                for g in f["lestnica_sigma"]:
                    out.append((job_id(s, t["id"], g, c), f["ocenshchik"], dict(t["obj"], kind=s, sigma=g, kontrol=None),
                                dict(t.get("params") or {}), c, s, t["id"], g, "osnova"))
                for g in k2["stupeni"]:
                    out.append((job_id(s, t["id"], g, c, "k2"), f["ocenshchik"],
                                dict(t["obj"], kind=s, sigma=g, kontrol={"k2": {"n": k2["n"], "seed": k2["seed"],
                                                                                "sposob": k2["sposob"]}}),
                                dict(t.get("params") or {}), c, s, t["id"], g, "k2"))
                if f["kontroli"]["k3"]:
                    g = f["kontroli"]["k3_sigma"]
                    out.append((job_id(s, t["id"], g, c, "k3"), f["ocenshchik"],
                                dict(t["obj"], kind=s, sigma=g, kontrol="k3"),
                                dict(t.get("params") or {}), c, s, t["id"], g, "k3"))
    return out


def check_spec(spec: dict) -> None:
    """Шаг 1: заморозка и полнота. Пустой SHA, лишняя или пропущенная семья, иная комиссия — отказ до данных."""
    bad = []
    if str(spec.get("v")) != V:
        bad.append(f"v = {spec.get('v')!r}, нужно {V!r}")
    if spec.get("status") != "zamorozheno":
        bad.append(f"status = {spec.get('status')!r}, нужно 'zamorozheno'")
    try:
        parse_dt(spec.get("zamorozheno_v", ""))
    except ValueError:
        bad.append("zamorozheno_v не в формате ГГГГ-ММ-ДД ЧЧ:ММ")
    if not (spec.get("final_b") or {}).get("sha256"):
        bad.append("нет SHA FINAL-B")
    for grp in ("kod", "geity"):
        if not spec.get(grp):
            bad.append(f"пусто: {grp}")
        for p, s in (spec.get(grp) or {}).items():
            if not s:
                bad.append(f"нет SHA {grp}: {p}")
    if not spec.get("fastbt_sha_vse"):
        bad.append("нет fastbt_sha_vse")
    for k, v in (spec.get("vhody") or {}).items():
        if not (v or {}).get("sha256") or not (v or {}).get("put"):
            bad.append(f"вход {k}: нет put или sha256")
    sv = spec.get("svechi_10m") or {}
    if not sv.get("tikery") or set(sv.get("sha256") or {}) != set(sv.get("tikery") or ()) \
            or not all((sv.get("sha256") or {}).values()):
        bad.append("svechi_10m: тикеры и SHA 10-минуток не совпадают или пусты")
    if [float(x) for x in spec.get("komissii_pct") or ()] != list(KOMISSII):
        bad.append(f"komissii_pct = {spec.get('komissii_pct')}, нужно {list(KOMISSII)} (Ж-031)")
    ok = spec.get("okno_testa") or {}
    try:
        if not (time.strptime(ok["s"], "%Y-%m-%d") <= time.strptime(ok["po"], "%Y-%m-%d")) or ok["s"] < "2022-01-01":
            bad.append("okno_testa: s ≤ po и s ≥ 2022-01-01")
    except (KeyError, ValueError, TypeError):
        bad.append("okno_testa: нужны s и po в формате ГГГГ-ММ-ДД")
    if spec.get("lag_ispolneniya_dney") != 1:
        bad.append("lag_ispolneniya_dney ≠ 1 (Д2: решение d исполняется d+1)")
    b = spec.get("baza") or {}
    if not b.get("ocenshchik") or not isinstance(b.get("obj"), dict) or not isinstance(b.get("N"), int) \
            or not isinstance(b.get("var_sr_d"), (int, float)):
        bad.append("baza: нужны ocenshchik, obj, N, var_sr_d")
    if not re.fullmatch(r"[\w/.-]+\.py:\w+", str(spec.get("proverka_sdviga") or "")):
        bad.append("proverka_sdviga: '<файл>.py:<функция>' (Д1, до счёта)")
    semi = spec.get("semi") or {}
    if set(semi) != set(SEMI):
        bad.append(f"semi = {sorted(semi)}, нужно ровно {list(SEMI)}")
    for s in SEMI:
        f = semi.get(s) or {}
        if not f.get("ocenshchik") or not isinstance(f.get("N"), int) or f["N"] < 1:
            bad.append(f"{s}: нужны ocenshchik и N ≥ 1")
        if not isinstance(f.get("var_sr_d"), (int, float)):
            bad.append(f"{s}: нет var_sr_d (дисперсия дневных SR проб на разработке)")
        ids = [t.get("id") for t in f.get("tochki") or ()]
        if not ids or len(set(ids)) != len(ids) or not all(ids) or \
                not all(isinstance(t.get("obj"), dict) for t in f.get("tochki") or ()):
            bad.append(f"{s}: точки — непустой список с уникальными id и obj")
        if tuple(f.get("lestnica_sigma") or ()) != LESTNICA:
            bad.append(f"{s}: lestnica_sigma ≠ {list(LESTNICA)}")
        kt = f.get("kontroli") or {}
        k2 = kt.get("k2") or {}
        if not (isinstance(k2.get("n"), int) and k2["n"] >= 1 and isinstance(k2.get("seed"), int) and k2.get("sposob")
                and k2.get("stupeni") and set(k2["stupeni"]) <= set(LESTNICA) - {"konst"}):
            bad.append(f"{s}: к2 — n, seed, sposob, stupeni ⊂ лестница без konst")
        if not isinstance(kt.get("k3"), bool) or (kt.get("k3") and kt.get("k3_sigma") not in LESTNICA):
            bad.append(f"{s}: к3 — k3 (bool) и k3_sigma из лестницы")
    lw = spec.get("lw") or {}
    if not all(isinstance(lw.get(k), int) for k in ("reps", "block", "seed")):
        bad.append("lw: reps, block, seed — целые")
    if bad:
        raise Stop("спецификация не заморожена — тест закрыт: " + "; ".join(bad))
    ids = {j[0] for j in plan_zadaniy(spec)}
    for h, g in (spec.get("gipotezy") or {}).items():
        if g.get("a") not in ids or g.get("b") not in ids or g.get("znak") not in ("+", "-") \
                or not isinstance(g.get("alfa"), float) or g.get("metrika") != "sharpe":
            bad.append(f"гипотеза {h}: a, b — id заданий, znak +/-, alfa, metrika = sharpe")
    if not spec.get("gipotezy"):
        bad.append("нет гипотез")
    if bad:
        raise Stop("спецификация не заморожена — тест закрыт: " + "; ".join(bad))


# ======================= 2. «да» владельца =======================

def check_da(spec: dict, P: Puti) -> None:
    d = spec.get("da_vladeltsa") or {}
    stroka, data, zh = str(d.get("stroka") or "").strip(), str(d.get("data") or ""), str(d.get("zhurnal") or "")
    if not stroka or not re.search(r"(?<![А-Яа-яЁё])да(?![А-Яа-яЁё])", stroka, re.I):
        raise Stop("нет строки «да» владельца (da_vladeltsa.stroka)")
    try:
        t_da = parse_dt(data)
    except ValueError:
        raise Stop("«да» без даты: da_vladeltsa.data — ГГГГ-ММ-ДД ЧЧ:ММ") from None
    if t_da < parse_dt(spec["zamorozheno_v"]):
        raise Stop("«да» раньше заморозки спецификации — «да» относится к другой версии")
    if not re.fullmatch(r"Ж-\d{3,}", zh):
        raise Stop("da_vladeltsa.zhurnal — номер вида Ж-NNN")
    txt = Path(P.zhurnal).read_text() if Path(P.zhurnal).exists() else ""
    m = re.search(rf"^### {re.escape(zh)}\b.*?(?=^### |\Z)", txt, re.M | re.S)
    if not m or stroka not in m.group(0):
        raise Stop(f"строки «да» нет в журнале {P.zhurnal} под {zh}")
    if d.get("sha_spec") != sha_spec_bez_da(spec):
        raise Stop("спецификация менялась после «да» (sha_spec ≠ SHA спецификации без поля «да»)")


# ======================= 3–5. замок, гейты, SHA =======================

def check_lock_free(P: Puti) -> None:
    if Path(P.lock).exists():
        raise Stop(f"{P.lock} уже есть — открытие было: {Path(P.lock).read_text()[:300]}")


def check_geity(spec: dict, P: Puti) -> None:
    bad = []
    for p in spec["geity"]:
        f = Path(P.root) / p
        if not f.exists():
            bad.append(f"{p}: нет файла")
            continue
        t = f.read_text()
        n_p = len(re.findall(r"^EVIDENCE:\s*pending\b", t, re.M))
        n_a = len(re.findall(r"^ABANDON:\s*(?!<id>)\S", t, re.M))
        if n_p or n_a:
            bad.append(f"{p}: pending {n_p}, ABANDON {n_a}")
    if bad:
        raise Stop("незакрытые гейты: " + "; ".join(bad))


def moduli(spec: dict) -> set:
    """Файлы кода, которые скрипт исполняет (каждый обязан стоять в spec.kod)."""
    m = {"scripts/otkrytie_gl4.py", "fastbt/daily_1850.py", "fastbt/s9_stats.py", "fastbt/sweep_drivers.py",
         spec["baza"]["ocenshchik"].split(":")[0], spec["proverka_sdviga"].split(":")[0]}
    m |= {f["ocenshchik"].split(":")[0] for f in spec["semi"].values()}
    return m


def verify_sha(spec: dict, P: Puti) -> dict:
    """Шаг 5: SHA всего, что входит в открытие. Возвращает {что: SHA} для журнала."""
    got, bad = {}, []

    def one(path, want, what):
        path = Path(path)
        if not path.exists():
            bad.append(f"{what}: нет {path}")
            return
        g = sha_put(path)
        got[what] = g
        if g != want:
            bad.append(f"{what}: SHA {g[:16]}… ≠ {str(want)[:16]}…")

    one(Path(P.root) / spec["final_b"]["put"], spec["final_b"]["sha256"], "FINAL-B")
    for p, s in spec["kod"].items():
        one(Path(P.root) / p, s, f"код {p}")
    net = sorted(moduli(spec) - set(spec["kod"]))
    if net:
        bad.append(f"исполняемый код без SHA в spec.kod: {net}")
    agg = fastbt_agg(P.root)
    got["fastbt/**"] = agg
    if agg != spec["fastbt_sha_vse"]:
        bad.append(f"код fastbt (все *.py, *.yaml): {agg[:16]}… ≠ fastbt_sha_vse {str(spec['fastbt_sha_vse'])[:16]}…")
    for p, s in spec["geity"].items():
        one(Path(P.root) / p, s, f"гейты {p}")
    for k, v in spec["vhody"].items():
        one(Path(P.main) / v["put"], v["sha256"], f"вход {k}")
    sv = spec["svechi_10m"]
    for t in sv["tikery"]:
        one(Path(P.main) / sv["katalog"] / f"{t}.parquet", sv["sha256"][t], f"10м {t}")
    if bad:
        raise Stop("SHA не сходятся — открытие не начато: " + "; ".join(bad))
    return got


def precheck(P: Puti) -> tuple:
    """Шаги 1–5 без чтения данных теста; возвращает (spec, SHA)."""
    spec = load_spec(P)
    check_spec(spec)
    check_da(spec, P)
    check_lock_free(P)
    check_geity(spec, P)
    return spec, verify_sha(spec, P)


def acquire_lock(P: Puti, info: dict) -> None:
    """Шаг 6: замок до чтения рядов; O_EXCL — два одновременных запуска не пройдут оба."""
    Path(P.lock).parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(P.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise Stop(f"{P.lock} уже есть — открытие было") from None
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(dict(info, status="начато", vremya=now()), ensure_ascii=False, indent=1))


def finish_lock(P: Puti, extra: dict, status: str) -> None:
    lk = Path(P.lock)
    d = json.loads(lk.read_text())
    d.update(extra, status=status, vremya_konets=now())
    os.chmod(lk, 0o644)
    lk.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    os.chmod(lk, 0o444)


# ======================= 6+. счёт под замком =======================

def zagruzit(P: Puti, ref: str):
    """Модуль по пути от корня (тот самый файл, чей SHA сверен); ref — 'путь.py' или 'путь.py:функция'."""
    put, _, fn = ref.partition(":")
    if str(P.root) not in sys.path:
        sys.path.insert(0, str(P.root))
    f = (Path(P.root) / put).resolve()
    name = "otkr_" + hashlib.sha1(str(f).encode()).hexdigest()[:16]     # другой корень — другой модуль
    if name not in sys.modules:
        sp = importlib.util.spec_from_file_location(name, f)
        mod = importlib.util.module_from_spec(sp)
        sys.modules[name] = mod
        sp.loader.exec_module(mod)
    mod = sys.modules[name]
    return getattr(mod, fn) if fn else mod


def ryad_1850(spec: dict, P: Puti, zhurnal: list) -> Path:
    """Шаг 7: дневной ряд до 18:50 по конец окна теста — функциями daily_1850 (read_bars, aggregate)."""
    import pandas as pd
    d1850 = zagruzit(P, "fastbt/daily_1850.py")
    sv, po = spec["svechi_10m"], pd.Timestamp(spec["okno_testa"]["po"])
    end = po + pd.Timedelta(days=1)
    parts = [d1850.aggregate(d1850.read_bars(Path(P.main) / sv["katalog"] / f"{t}.parquet", end=end), t)
             for t in sv["tikery"]]
    d = pd.concat(parts, ignore_index=True)
    d = d[d["date"] <= po].sort_values(["ticker", "date"]).reset_index(drop=True)
    out = Path(P.out_data) / "ohlcv_daily_1850_test.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out, index=False)
    zhurnal.append(dict(vremya=now(), etap="posle_zamka", chto="ряд 18:50", put=str(out), sha256=sha(out),
                        strok=int(len(d)), daty_po=str(d["date"].max().date()) if len(d) else None))
    return out


def kontekst(spec: dict, P: Puti, ryad: Path) -> dict:
    return {"zamok": str(P.lock), "ryad_1850": str(ryad), "okno": dict(spec["okno_testa"]),
            "vhody": {k: str(Path(P.main) / v["put"]) for k, v in spec["vhody"].items()}, "main": str(P.main)}


def check_row(row: dict, spec: dict, c: float, jid: str, kontrol) -> None:
    bad = [k for k in ROW_KEYS if k not in row]
    if bad:
        raise RuntimeError(f"{jid}: в строке оценщика нет {bad}")
    ok = spec["okno_testa"]
    if str(row["v"]) != ROW_V or row["period"] != "test" or row["chestnaya"] != "test" \
            or str(row["okno_s"]) != ok["s"] or str(row["okno_po"]) != ok["po"] \
            or abs(float(row["comm_pct"]) - c) > 1e-12 or len(row["excess_d"]) != int(row["T"]):
        raise RuntimeError(f"{jid}: строка не по контракту v{ROW_V} (v, period/chestnaya = test, окно, комиссия, T)")
    if kontrol == "k2" and not {"p_k2", "n_k2"} <= set(row):
        raise RuntimeError(f"{jid}: к2 без p_k2 / n_k2")


def fnum(x):
    """Число для JSON: nan/inf → None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def holm(ps: list) -> list:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out, run = [0.0] * m, 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        out[i] = run
    return out


def schitat(spec: dict, P: Puti, ctx: dict) -> dict:
    """Шаг 9: все задания одним прогоном; затем производные таблицы. Выбора нет."""
    lwf = zagruzit(P, "fastbt/s9_stats.py:lw_sharpe_diff")
    sd = zagruzit(P, "fastbt/sweep_drivers.py")
    lw = spec["lw"]
    pometka = spec.get("pometka", "после аудита")
    rows = {}
    for jid, ref, obj, params, c, semya, tochka, sigma, kontrol in plan_zadaniy(spec):
        obj["test"] = ctx
        r = zagruzit(P, ref)(obj, params, "test", c)
        check_row(r, spec, c, jid, kontrol)
        rows[jid] = dict(r, _semya=semya, _tochka=tochka, _sigma=sigma, _c=c, _kontrol=kontrol)

    def lwd(a, b):
        xa, xb = rows[a]["excess_d"], rows[b]["excess_d"]
        if len(xa) != len(xb) or rows[a].get("daty", None) != rows[b].get("daty", None):
            raise RuntimeError(f"{a} и {b}: разные дни — Ледуа–Вольф невозможен")
        d, se, p = lwf(xa, xb, reps=lw["reps"], block=lw["block"], seed=lw["seed"])
        return fnum(d), fnum(se), fnum(p)

    osn, lest, kontr, pg = [], [], [], []
    for jid, r in rows.items():
        if r["_kontrol"] != "osnova":
            continue
        fam = spec["baza"] if r["_semya"] == BAZA else spec["semi"][r["_semya"]]
        N, var = fam["N"], float(fam["var_sr_d"])
        sr_star = sd.expected_max_sharpe(N, var)
        d = se = p = None
        if r["_semya"] != BAZA:
            d, se, p = lwd(jid, job_id(BAZA, "-", "-", r["_c"]))
        osn.append(dict(id=jid, semya=r["_semya"], tochka=r["_tochka"], sigma=r["_sigma"], komissiya_pct=r["_c"],
                        sharpe=fnum(r["sharpe"]), msr=fnum(r["msr"]), sortino=fnum(r["sortino"]),
                        kalmar=fnum(r["kalmar"]), mdd=fnum(r["mdd"]),
                        psr=fnum(sd.psr(r["sr_d"], 0.0, int(r["T"]), r["skew"], r["kurt"])),
                        dsr=fnum(sd.psr(r["sr_d"], sr_star, int(r["T"]), r["skew"], r["kurt"])), N=N,
                        oborot_y=fnum(r["oborot_y"]), dni_v_rynke=fnum(r["dni_v_rynke"]), sdelok_y=fnum(r["sdelok_y"]),
                        dolya_ispolneniy=fnum(r.get("dolya_ispolneniy")), delta_k_baze=d, se_delta=se, p_lw=p,
                        T=int(r["T"]), okno_s=r["okno_s"], okno_po=r["okno_po"], chestnaya="test", status=STATUS,
                        pometka=pometka))
        for god, s in sorted(r["by_year"].items()):
            pg.append(dict(id=jid, god=int(god), sharpe=fnum(s), status=STATUS))
    for s in SEMI:
        f = spec["semi"][s]
        for t in f["tochki"]:
            for c in spec["komissii_pct"]:
                for a, b in zip(LESTNICA[:-1], LESTNICA[1:]):
                    ja, jb = job_id(s, t["id"], a, c), job_id(s, t["id"], b, c)
                    d, se, p = lwd(jb, ja)
                    lest.append(dict(semya=s, tochka=t["id"], komissiya_pct=c, ot=a, k=b,
                                     sharpe_ot=fnum(rows[ja]["sharpe"]), sharpe_k=fnum(rows[jb]["sharpe"]), delta=d,
                                     se_delta=se, p_lw=p, status=STATUS))
                k2 = f["kontroli"]["k2"]
                for g in k2["stupeni"]:
                    jo, jk = job_id(s, t["id"], g, c), job_id(s, t["id"], g, c, "k2")
                    kontr.append(dict(semya=s, tochka=t["id"], sigma=g, komissiya_pct=c, kontrol="k2",
                                      sharpe_osnova=fnum(rows[jo]["sharpe"]), sharpe_kontrol=fnum(rows[jk]["sharpe"]),
                                      delta=fnum(rows[jo]["sharpe"] - rows[jk]["sharpe"]), se_delta=None,
                                      p=fnum(rows[jk]["p_k2"]), sposob_p="перестановки (оценщик)",
                                      n_k2=int(rows[jk]["n_k2"]), seed_k2=k2["seed"], status=STATUS))
                if f["kontroli"]["k3"]:
                    g = f["kontroli"]["k3_sigma"]
                    jo, jk = job_id(s, t["id"], g, c), job_id(s, t["id"], g, c, "k3")
                    d, se, p = lwd(jo, jk)
                    kontr.append(dict(semya=s, tochka=t["id"], sigma=g, komissiya_pct=c, kontrol="k3",
                                      sharpe_osnova=fnum(rows[jo]["sharpe"]), sharpe_kontrol=fnum(rows[jk]["sharpe"]),
                                      delta=d, se_delta=se, p=p, sposob_p="Ледуа–Вольф", n_k2=None, seed_k2=None,
                                      status=STATUS))
    gip = []
    for h, g in sorted(spec["gipotezy"].items()):
        d, se, p = lwd(g["a"], g["b"])
        gip.append(dict(gipoteza=h, formulirovka=g.get("formulirovka", ""), a=g["a"], b=g["b"], metrika="sharpe",
                        znak=g["znak"], delta=d, se_delta=se, p=p, alfa=g["alfa"], status=STATUS))
    for x, ph in zip(gip, holm([x["p"] if x["p"] is not None else 1.0 for x in gip])):
        x["p_holm"] = ph
        napr = x["delta"] is not None and (x["delta"] > 0) == (x["znak"] == "+")
        x["vyvod"] = "подтверждена" if (ph < x["alfa"] and napr) else "не подтверждена"
    ryady = {jid: [fnum(v) for v in r["excess_d"]] for jid, r in rows.items()}
    return dict(osnovnaya=osn, lestnica=lest, kontroli=kontr, gipotezy=gip, po_godam=pg, _ryady=ryady)


# ======================= 10. выход =======================

def shapka(name: str, spec: dict, meta: dict) -> str:
    ok = spec["okno_testa"]
    return (f"# таблица: t_gl4_test_{name} — {OPIS[name]}\n"
            f"# статус: {STATUS}\n"
            f"# окно данных: {ok['s']} — {ok['po']}; исполнение d+1; честная половина: тест 2022+, одно открытие\n"
            f"# код: {meta['git_head']}; спецификация: {meta['spec_sha']}; хэш входов: {meta['vhody_agg']}\n"
            f"# источник: scripts/otkrytie_gl4.py (v{V}); числа — {meta['json']}\n")


def to_csv(name: str, rows: list, spec: dict, meta: dict, out: Path) -> Path:
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS[name], lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS[name]})
    p = out / f"t_gl4_test_{name}.csv"
    p.write_text(shapka(name, spec, meta) + buf.getvalue())
    return p


def proverit_shemu(doc, shema: dict, put="$", root=None) -> list:
    """Минимальная проверка JSON Schema (type, enum, const, required, properties, additionalProperties, items, $ref,
    minimum, maximum) — jsonschema в окружении нет, новой зависимости не добавляем."""
    root = root or shema
    if "$ref" in shema:
        node = root
        for k in shema["$ref"].lstrip("#/").split("/"):
            node = node[k]
        return proverit_shemu(doc, node, put, root)
    err = []
    tipy = {"string": str, "integer": int, "number": (int, float), "object": dict, "array": list, "boolean": bool,
            "null": type(None)}
    t = shema.get("type")
    if t:
        ts = t if isinstance(t, list) else [t]
        okt = any(isinstance(doc, tipy[x]) and not (x in ("integer", "number") and isinstance(doc, bool)) for x in ts)
        if not okt:
            return [f"{put}: тип {type(doc).__name__} ∉ {ts}"]
    if "const" in shema and doc != shema["const"]:
        err.append(f"{put}: ≠ {shema['const']!r}")
    if "enum" in shema and doc not in shema["enum"]:
        err.append(f"{put}: {doc!r} ∉ {shema['enum']}")
    if isinstance(doc, (int, float)) and not isinstance(doc, bool):
        if "minimum" in shema and doc < shema["minimum"]:
            err.append(f"{put}: < {shema['minimum']}")
        if "maximum" in shema and doc > shema["maximum"]:
            err.append(f"{put}: > {shema['maximum']}")
    if isinstance(doc, dict):
        for k in shema.get("required", ()):
            if k not in doc:
                err.append(f"{put}: нет {k}")
        pr = shema.get("properties", {})
        for k, v in doc.items():
            if k in pr:
                err += proverit_shemu(v, pr[k], f"{put}.{k}", root)
            elif shema.get("additionalProperties") is False:
                err.append(f"{put}: лишнее поле {k}")
            elif isinstance(shema.get("additionalProperties"), dict):
                err += proverit_shemu(v, shema["additionalProperties"], f"{put}.{k}", root)
    if isinstance(doc, list) and "items" in shema:
        for i, x in enumerate(doc):
            err += proverit_shemu(x, shema["items"], f"{put}[{i}]", root)
    return err


def zapisat(spec: dict, P: Puti, res: dict, sha_do: dict, zhurnal: list) -> dict:
    import pandas as pd
    out = Path(P.out_tables)
    out.mkdir(parents=True, exist_ok=True)
    vhody_agg = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in sorted(sha_do.items())).encode()).hexdigest()
    meta = dict(git_head=git_head(P.root), spec_sha=sha(P.spec), vhody_agg=vhody_agg, json=str(P.out_json))
    tabl = {}
    for name in COLS:
        p = to_csv(name, res[name], spec, meta, out)
        tabl[name] = {"put": str(p), "sha256": sha(p)}
    ry = Path(P.out_data) / "ryady_excess_test.parquet"
    pd.DataFrame([(j, i, x) for j, xs in res["_ryady"].items() for i, x in enumerate(xs)],
                 columns=["id", "t", "excess_d"]).to_parquet(ry, index=False)
    tabl["ryady_excess"] = {"put": str(ry), "sha256": sha(ry)}
    doc = {
        "v": V, "status": STATUS,
        "meta": {"sozdan": now(), "git_head": meta["git_head"], "spec_sha256": meta["spec_sha"],
                 "final_b_sha256": spec["final_b"]["sha256"], "zamok": str(P.lock),
                 "okno_testa": dict(spec["okno_testa"]), "lag_ispolneniya_dney": spec["lag_ispolneniya_dney"],
                 "chestnaya": "test", "komissii_pct": list(spec["komissii_pct"]), "lw": dict(spec["lw"]),
                 "pometka": spec.get("pometka", "после аудита"), "da_vladeltsa": dict(spec["da_vladeltsa"]),
                 "vhody_agg_sha256": vhody_agg},
        "sha_do_zamka": sha_do,
        "ryad_1850": [z for z in zhurnal if z["chto"] == "ряд 18:50"][0],
        "osnovnaya": res["osnovnaya"], "lestnica": res["lestnica"], "kontroli": res["kontroli"],
        "gipotezy": res["gipotezy"], "po_godam": res["po_godam"], "tablicy": tabl,
    }
    err = proverit_shemu(doc, json.loads(Path(P.schema).read_text()))
    if err:
        raise RuntimeError("JSON чисел не проходит схему: " + "; ".join(err[:10]))
    Path(P.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(P.out_json).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    for k, v in tabl.items():
        zhurnal.append(dict(vremya=now(), etap="vyhod", chto=k, put=v["put"], sha256=v["sha256"]))
    zhurnal.append(dict(vremya=now(), etap="vyhod", chto="JSON чисел", put=str(P.out_json), sha256=sha(P.out_json)))
    zp = out / "zhurnal_sha_gl4.jsonl"
    zp.write_text("".join(json.dumps(z, ensure_ascii=False) + "\n" for z in zhurnal))
    return {"json": sha(P.out_json), "zhurnal_sha": sha(zp), **{k: v["sha256"] for k, v in tabl.items()}}


def run_test(P: Puti) -> int:
    spec, sha_do = precheck(P)
    zhurnal = [dict(vremya=now(), etap="do_zamka", chto=k, sha256=v) for k, v in sorted(sha_do.items())]
    acquire_lock(P, dict(v=V, spec=str(P.spec), spec_sha256=sha(P.spec), git_head=git_head(P.root), sha_do_zamka=sha_do))
    try:
        ryad = ryad_1850(spec, P, zhurnal)
        ctx = kontekst(spec, P, ryad)
        sdv = zagruzit(P, spec["proverka_sdviga"])(ctx)
        plohie = [x for x in sdv if not x.get("ok")]
        zhurnal.append(dict(vremya=now(), etap="posle_zamka", chto="проверка сдвига",
                            ryady=[x.get("ryad") for x in sdv], ne_proshli=[x.get("ryad") for x in plohie]))
        if not sdv or plohie:
            finish_lock(P, {"proverka_sdviga": sdv}, "отказ: проверка сдвига")
            raise Stop(f"проверка сдвига не прошла: {[x.get('ryad') for x in plohie] or 'пусто'}")
        res = schitat(spec, P, ctx)
        vyhod = zapisat(spec, P, res, sha_do, zhurnal)
    except Stop:
        raise
    except BaseException as e:      # noqa: BLE001 — замок фиксирует сбой; повтор только по решению владельца
        finish_lock(P, {"oshibka": repr(e)[:500]}, "прервано")
        raise
    finish_lock(P, {"vyhod_sha256": vyhod}, "завершено")
    print(f"открытие завершено: {P.out_json}", flush=True)
    return 0


def run_proverki(P: Puti) -> int:
    """Шаги 1, 3–5 и состояние «да» — без замка и без данных. Для заморозки FINAL-B до «да»."""
    spec = load_spec(P)
    check_spec(spec)
    try:
        check_da(spec, P)
        print("«да»: есть")
    except Stop as e:
        print(f"«да»: нет ({e.msg})")
    check_lock_free(P)
    check_geity(spec, P)
    verify_sha(spec, P)
    print(f"проверки до замка пройдены; sha_spec без «да» = {sha_spec_bez_da(spec)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scope", choices=["test"])
    ap.add_argument("--proverki", action="store_true")
    a = ap.parse_args(argv)
    if a.proverki:
        return run_proverki(Puti())
    if a.scope != "test":
        raise Stop("нужен --scope test (открытие) или --proverki")
    return run_test(Puti())


if __name__ == "__main__":
    sys.exit(main())
