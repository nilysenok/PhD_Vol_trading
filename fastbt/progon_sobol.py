"""Л-6: драйвер Соболя прогона этапа 1 (docs/otchety/2026-10-28_progon_etap1.md §1, §3 IG-5/6/8, §4; Ж-082, Ж-083).
Куб — fastbt/designs/progon/kub_S{k}.yaml; точка i — i-я точка Соболя (scramble, seed), потом canon_v2 (N — один раз).
Проход 1 — индексы 0…4999, проход 2 — 5000…29999 той же последовательности (fast_forward). 10 процессов, куски по 100 точек,
у процесса свой p{NN}/. Резюм: klyuch_sha, уже закрытые в ledger, не считаются. Путь: медленный (EvaluatorV2, эталон)
или быстрый (--fast: fast_etap1.patched() для покрытых точек). Метрик не печатает.
Л10 (Ж-127, Ж-129 п. 8, Ж-130): --kub v5 — fastbt/designs/v5/kub_S{k}.yaml, свёртки canon_v5, якоря yakorya_progon в точке;
период train4 | v2 (свой ключ ledger и свой каталог), seed | seed2 куба, проход k3 (полная решётка раздела k3), fdr из куба
(оценщик обязан считать его же — иначе отказ). Путь designs/progon (по умолчанию) — побайтно прежний (Rezhim R0)."""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.util
import itertools
import json
import multiprocessing as mp
import random
import resource
import shutil
import sys
import time
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

from fastbt import canon_v5 as cv5, canon_v6, fdr as fdr_mod, ledger_v2 as lm, period_v2, progon_v6 as v6

ROOT = Path(__file__).resolve().parent.parent
KUB = ROOT / "fastbt/designs/progon/kub_{sid}.yaml"
ZAMER = ROOT / "docs/tables/progon_e1"
LEDGER = ZAMER / "_ledger_v2"
SIDS = [f"S{k}" for k in range(1, 7)]
PERIOD, KOMISSIYA, KUSOK, GB, BAIT_NA_TOCHKU, MAX_PROC = "train4", 0.0004, 100, 1e9, 80_000, 10
BAIT_V21 = 20_000       # суммы сегментов (17 × 16 × 9 × 8 Б ≈ 20 КБ) вместо рядов ≈ 80 КБ
BAZA = dict(okno_s="2016-01-01", okno_po="2019-12-31", progrev_s="2015-01-01", zaderzhka="d+1", chestnaya="net",
            status="ok", prichina=None)
SVERKA_PROPUSK = {"t_sek", "kod", "kod_slit", "klyuch_sha", "excess", "sloy"}
Sreda = namedtuple("Sreda", "evr kod vhody daty tikery")
KUB5, LEDGER5 = ROOT / "fastbt/designs/v5/kub_{sid}.yaml", ZAMER / "_ledger_v5"
KUB6, LEDGER6 = ROOT / "fastbt/designs/v6/kub_{sid}.yaml", ZAMER / "_ledger_v6"      # Ж-137: куб v6, ledger 2.2
OCENSHCHIK_V5 = ("fastbt.eval_v5", "EvaluatorV5")        # лист Л9; контракт выхода = EvaluatorV2 (+ poz, fdr)
OKNO = {n: (f"{p.y0}-01-01", str(p.progrev.date())) for n, p in period_v2.PERIODY.items()}   # (okno_s, progrev_s); train4, v2, wf
Rezhim = namedtuple("Rezhim", "setka period seed k3 fdr rol t_vyhod zam", defaults=(None, None, ()))   # setka: progon | v5 | v6; seed: ключ sobol куба
# rol = "spravka" (Ж-137 п. 6): тот же куб, t_vyhod точек подменён на rz.t_vyhod, ledger 2.2 с rol = spravka; вне N (Obedinenie)
# v6: rol всегда (sobol | kontrol | spravka); zam = ((sid, sha zamorozka.yaml, оси), …) — progon_v6.zamorozka
R0 = Rezhim("progon", PERIOD, "seed", False, None)            # прежний путь designs/progon
fabrika = None      # sid -> Sreda; None — реальная (fabrika_real); тесты подставляют синтетику
_W: dict = {}


def kub_put(sid: str, rz: Rezhim = R0) -> Path:
    return {"progon": KUB, "v5": KUB5, "v6": KUB6}[rz.setka].with_name(f"kub_{sid}.yaml")


def load_kub(sid: str, rz: Rezhim = R0) -> dict:
    if rz.setka == "v6":                                   # куб v6 + основа v5 (ключ "v5")
        return canon_v6.load_kub(sid, kub_put(sid, rz).parent)
    return yaml.safe_load(kub_put(sid, rz).read_text(encoding="utf-8"))


def baza(rz: Rezhim) -> dict:
    return {**BAZA, "okno_s": OKNO[rz.period][0], "progrev_s": OKNO[rz.period][1]}


def proverit_v5(c: dict, rz: Rezhim) -> None:
    """Куб v5 (canon_v5.proverit) + период объявлен в кубе, комиссия 0,04 %, fdr куба = fdr прохода и известен fdr.py."""
    (canon_v6 if rz.setka == "v6" else cv5).proverit(c)
    period_v2.proverit_chtenie(rz.period)                  # wf (дни ≥ 2022): отказ до отметки владельца
    if rz.period not in OKNO or rz.period not in c.get("periody", [c.get("period")]):
        raise ValueError(f"{c['sid']}: период {rz.period!r} не объявлен в кубе (periody) — отказ")
    if c.get("komissiya") != KOMISSIYA:
        raise ValueError("куб: komissiya = 0.0004")
    if c.get("fdr") != rz.fdr or (rz.fdr is not None and rz.fdr not in fdr_mod.REZHIMY):
        raise ValueError(f"{c['sid']}: fdr куба {c.get('fdr')!r}, прохода {rz.fdr!r}; известны None и {fdr_mod.REZHIMY} — отказ")


def proverit_ledger(rz: Rezhim, v: str) -> None:
    """ledger 2.1 режет кварталы по годам периода (ledger_v2.gody_perioda: train4 2016–2019, v2 2015–2019);
    окно периода шире годов ledger — отказ до старта, а не в процессе."""
    gody = lm.gody_perioda(rz.period)
    if v in lm.SUMMY_V and not set(period_v2.gody(rz.period)) <= set(gody):
        raise SystemExit(f"ledger 2.1 считает кварталы {gody[0]}–{gody[-1]}, период {rz.period} шире — "
                         "--ledger 2.0 или расширение ledger_v2 — отказ")


def proverit_kub(c: dict) -> None:
    if c.get("v", 2) not in (2, 3):
        raise ValueError(f"куб v = {c.get('v')}: драйвер читает v: 2 и v: 3")
    c = {k: v for k, v in c.items() if k not in ("faza3", "msp")}   # v: 3 — разделы фазы 3 и слоя МСП не оси Соболя
    if c.get("period") != PERIOD:
        raise ValueError(f"период {c.get('period')!r}: прогон считает только {PERIOD} (Ж-082)")
    if c.get("komissiya") != KOMISSIYA or c.get("fdr") is not None:
        raise ValueError("куб: komissiya = 0.0004 и fdr = null")
    from fastbt.signals.interface import get_family
    if set(c["signal"]) != set(get_family(c["sid"]).grid):    # оценщик молча игнорирует чужие оси
        raise ValueError(f"{c['sid']}: оси signal {sorted(c['signal'])} ≠ оси семьи {sorted(get_family(c['sid']).grid)}")


def tochki_v5(kub: dict, n0: int, n1: int, rz: Rezhim) -> list:
    """v5: [(i, якоря ∪ canon_v5)]; несовместимая точка выброшена (индекс i пропущен). k3 — i-я точка полной решётки
    раздела k3 (объём ≤ 12 800 — Соболь давал бы повторы), иначе i-я точка Соболя с kub.sobol[rz.seed]."""
    proverit_v5(kub, rz)
    if rz.setka == "v6":
        return v6.tochki(kub, n0, n1, rz, tochki_v5)
    if rz.k3:
        k = cv5.k3_kub(kub)
        cv5.proverit(k)
        imena, setki = zip(*cv5.osi(k))
        kombi = enumerate(itertools.islice(itertools.product(*setki), n0, n1), n0)
        out = [(i, c) for i, c in ((i, cv5.canon_v5(kub["sid"], dict(zip(imena, x)))) for i, x in kombi) if c is not None]
    else:
        out = cv5.tochki_v5(kub, n0, n1, rz.seed)[0]
    ya = kub.get("yakorya_progon", {})
    return [(i, {**ya, **c}) for i, c in out]


def obem_k3(kub: dict, rz: Rezhim = R0) -> int:
    return v6.obem_k3(kub) if rz.setka == "v6" else cv5.obem(cv5.k3_kub(kub))


def tochki(kub: dict, n0: int, n1: int, rz: Rezhim = R0) -> list:
    """[(i, pt)], i ∈ [n0, n1): индекс значения оси = floor(u·len(ось)), затем canon_v2."""
    if rz.rol == "spravka":
        return [(i, {**pt, "t_vyhod": rz.t_vyhod}) for i, pt in tochki(kub, n0, n1, rz._replace(rol=None))]
    if rz.setka != "progon":
        return tochki_v5(kub, n0, n1, rz)
    proverit_kub(kub)
    sekcii = ("signal", "obshchie", "novye") if kub.get("v", 2) == 2 else ("signal", "filtr", "vhod", "vyhod")
    osi = [(n, v["setka"] if isinstance(v, dict) else v) for sec in sekcii for n, v in kub.get(sec, {}).items()]
    s = kub["sobol"]
    sob = qmc.Sobol(len(osi), scramble=s["scramble"], seed=s["seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         # «баланс» Соболя: n не степень двойки — продолжение, так задумано
        if n0:                                  # scipy: fast_forward(0) — OverflowError
            sob.fast_forward(n0)
        u =sob.random(n1 - n0)
    return [(n0 + j, lm.canon_v2(kub["sid"], {n: g[min(int(x * len(g)), len(g) - 1)] for (n, g), x in zip(osi, row)}))
            for j, row in enumerate(u)]


def proverit_disk(n_tochek: int, root: Path = LEDGER, bait: int = BAIT_NA_TOCHKU) -> None:
    """IG-6 (Л7, класс C): свободно ≥ 1,5 × оценки прохода + 5 ГБ; жёсткого порога 40 ГБ нет."""
    p = Path(root)
    while not p.exists():
        p = p.parent
    free, need = shutil.disk_usage(p).free, 1.5 * n_tochek * bait + 5 * GB
    if free < need:
        raise SystemExit(f"IG-6: свободно {free / GB:.1f} ГБ, нужно ≥ {need / GB:.1f} ГБ — отказ")


def ocenshchik(sid: str, inp, rz: Rezhim):
    from fastbt import etap1
    from fastbt.etap1_eval_v2 import EvaluatorV2
    if rz.setka == "progon":
        return EvaluatorV2(sid, inp, etap1.base_point(sid), KOMISSIYA)
    if rz.setka == "v6":
        return v6.ocenshchik(sid, inp, etap1.base_point(sid), KOMISSIYA, rz)
    mod, cls = OCENSHCHIK_V5
    if importlib.util.find_spec(mod) is None:
        raise SystemExit(f"{mod} (лист Л9) не слит — куб v5 считать нечем — отказ")
    kw = {"rol": rz.rol} if rz.rol == "spravka" else {}
    if mod == "fastbt.eval_v6":              # оценщик v6 ряды не читает: боевые источники подаёт istochniki_v6 (Ист-v6)
        from fastbt import istochniki_v6
        kw["ist"] = istochniki_v6.zagruzit(rz.period, list(inp.sk.tickers))
    return getattr(importlib.import_module(mod), cls)(sid, inp, etap1.base_point(sid), KOMISSIYA, fdr=rz.fdr, **kw)


def fabrika_real(sid: str, rz: Rezhim = R0) -> Sreda:
    from fastbt import cache, etap1
    from fastbt.etap1_eval import years_of
    inp = etap1.build_inputs(sid, period=rz.period)
    evr = ocenshchik(sid, inp, rz)
    dt = evr._engine(rz.period)[0].dates
    daty = dt[np.isin(dt.year, years_of(rz.period))].to_numpy().astype("datetime64[D]")
    vh = v6.vhody(inp.digest(), evr) if rz.setka == "v6" else inp.digest()
    return Sreda(evr, cache.code_hash(), vh, daty, list(inp.sk.tickers))


def sreda(sid: str, rz: Rezhim = R0) -> Sreda:
    """Один оценщик на стратегию за раз; даты ≥ 2020 в рядах — отказ; v5: fdr оценщика ≠ fdr куба — отказ."""
    if _W.get("sid") != (sid, rz):
        _W.pop("sreda", None)
        sr = fabrika(sid) if fabrika else fabrika_real(sid, rz)
        if sr.daty.size == 0 or sr.daty.max() >= np.datetime64("2020-01-01"):
            raise ValueError(f"{sid}: в рядах есть даты ≥ 2020 — отказ (Ж-082)")
        if rz.setka != "progon" and getattr(sr.evr, "fdr", "нет атрибута fdr") != rz.fdr:
            raise ValueError(f"{sid}: оценщик {type(sr.evr).__name__} не считает fdr = {rz.fdr!r} куба "
                             f"(у него {getattr(sr.evr, 'fdr', 'нет атрибута fdr')!r}) — отказ, не молчаливый null")
        _W["sid"], _W["sreda"] = (sid, rz), sr
    return _W["sreda"]


def manifest(sid: str, sr: Sreda, c: dict, kod_slit: bool, rz: Rezhim = R0) -> dict:
    b = baza(rz)
    m = {"period": rz.period, "okno_s": b["okno_s"], "okno_po": b["okno_po"], "progrev_s": b["progrev_s"],
         "komissiya": KOMISSIYA, "fdr": rz.fdr, "kod": sr.kod, "vhody": sr.vhody, "kod_slit": kod_slit,
         "daty_sha": hashlib.sha256(sr.daty.tobytes()).hexdigest(), "tikery": sr.tikery, "T": int(sr.daty.size),
         "kub_sha": hashlib.sha256(kub_put(sid, rz).read_bytes()).hexdigest(),
         "sobol_seed": None if rz.k3 else c["sobol"][rz.seed], "sozdan": time.strftime("%Y-%m-%d")}
    m = m if rz.setka == "progon" else {**m, "setka": c["setka"], "k3": rz.k3, "sobol_klyuch": rz.seed}
    if rz.setka == "v6":                                   # README v6: setka_data, zamorozka_sha (k3 — без заморозки)
        m = {**m, "setka_data": str(c["data"]), "zamorozka_sha": None if rz.k3 else v6.zam_sid(rz, sid)[0]}
    return {**m, "rol": rz.rol, "t_vyhod": rz.t_vyhod} if rz.rol else m


def sha_tochki(sid: str, sr: Sreda, pt: dict, rz: Rezhim = R0) -> str:
    return lm.klyuch_sha(lm.klyuch(sid, pt, rz.period, KOMISSIYA, rz.fdr), sr.kod, sr.vhody)


def fast_ok(pt: dict) -> bool:
    return importlib.util.find_spec("fastbt.fast_etap1") is not None and \
        importlib.import_module("fastbt.fast_etap1").pokryto(pt)


def schitat_tochku(sr: Sreda, sid: str, i: int, pt: dict, ist: str, kod_slit: bool, fast: bool, v: str = lm.V,
                   rz: Rezhim = R0) -> tuple:
    """(строка §1.2, port float64, tik): ошибка оценщика — строка `oshibka`, а не обрыв прогона."""
    t0, sloy, r, ex_p, ex_t = time.perf_counter(), "slow", {}, None, None
    try:
        if fast and fast_ok(pt):
            with importlib.import_module("fastbt.fast_etap1").patched():
                r, ex_p, ex_t = sr.evr.evaluate_v2(pt, rz.period)
            sloy = "fast"
        else:
            r, ex_p, ex_t = sr.evr.evaluate_v2(pt, rz.period)
    except Exception as e:                      # noqa: BLE001 — контракт §1.2: status = oshibka, причина в строке
        r = {"status": "oshibka", "prichina": f"{type(e).__name__}: {e}"[:200]}
    kl = lm.klyuch(sid, pt, rz.period, KOMISSIYA, rz.fdr)
    r = {**baza(rz), **r, "v": v, "format": lm.FORMAT, "strategiya": sid, "klyuch": kl, "kod": sr.kod,
         "klyuch_sha": lm.klyuch_sha(kl, sr.kod, sr.vhody), "vhody": sr.vhody, "kod_slit": kod_slit,
         "period": rz.period, "komissiya": KOMISSIYA, "fdr": rz.fdr, "istochnik": ist, "sobol_i": i, "kontrol": None,
         "vozm": None, "sloy": sloy, "t_sek": time.perf_counter() - t0}
    return r, ex_p, ex_t


def _init(ctx: dict, schetchik) -> None:
    _W.clear()
    _W.update(ctx)
    with schetchik.get_lock():
        _W["nn"], schetchik.value = schetchik.value, schetchik.value + 1


def schitat_kusok(zad: tuple) -> tuple:
    sid, n0, n1 = zad
    rz = _W.get("rz", R0)
    c, sr = _W["kubs"][sid], sreda(sid, rz)
    d = Path(_W["root"]) / sid
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "daty.npy").exists():
        np.save(d / f"daty.npy.{_W['nn']}.tmp.npy", sr.daty)
        (d / f"daty.npy.{_W['nn']}.tmp.npy").replace(d / "daty.npy")
    v = _W.get("v", lm.V)
    mf = manifest(sid, sr, c, _W["kod_slit"], rz)
    poz_ok = hasattr(sr.evr, "poz")                                    # Л7b: настоящий оценщик отдаёт |pos| (EvaluatorV2.poz)
    pis = {lm.V22: lm.Pisatel22, lm.V21: lm.Pisatel21}.get(v)
    w = pis(_W["root"], sid, _W["nn"], mf, daty=sr.daty, poz=poz_ok) if pis else lm.Pisatel(_W["root"], sid, _W["nn"], mf)
    zak, t, oshibok = _W["zakryt"][sid] | w.done(), [], 0
    for i, pt in tochki(c, n0, n1, rz):
        if sha_tochki(sid, sr, pt, rz) in zak:
            continue
        r, ex_p, ex_t = schitat_tochku(sr, sid, i, pt, _W["ist"], _W["kod_slit"], _W["fast"], v, rz)
        if v == lm.V22:
            r.update(setka=mf["setka"], rol=rz.rol, klyuch_n=lm.klyuch_n(sid, r["klyuch"]))
        importlib.import_module("fastbt.zaem_v2").pisat(w, r, ex_p, ex_t, getattr(sr.evr, "zaem", None),
                                                           sr.evr.poz if poz_ok and v in lm.SUMMY_V and r["status"] == "ok" else None)   # Ж-089: ряд платы за заём; Л7b: |pos|
        importlib.import_module("fastbt.tejk_signal").pisat_diag(w.pd, r, getattr(sr.evr, "diag", None) if r["status"] == "ok" else None)   # Ж-087
        zak = zak | {r["klyuch_sha"]}
        t.append(r["t_sek"])
        oshibok += r["status"] == "oshibka"
    importlib.import_module("fastbt.zaem_v2").zakryt(w)
    w.close()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    return sid, len(t), oshibok, t, rss


def zakrytye(root: Path, sid: str) -> set:
    """Резюм: klyuch_sha закрытых шардов всех p{NN}/ (Pisatel сам отбрасывает хвост без SHA). Вызывать до старта процессов."""
    mp_ = Path(root) / sid / "manifest.json"
    if not mp_.exists():
        return set()
    m = json.loads(mp_.read_text(encoding="utf-8"))
    return set().union(*[lm.Pisatel(root, sid, int(p.name[1:]), m).done() for p in (Path(root) / sid).glob("p[0-9]*")])


def schet(sids, n0: int, n1, ist: str, root: Path, processov: int, kod_slit: bool, fast: bool, v: str = lm.V,
          rz: Rezhim = R0) -> dict:
    """Все sids × [n0, n1) куски по 100 в очередь на `processov` процессов; n1 — число или {sid: n1}. {sid: {n, oshibok, t, rss}}."""
    if not 1 <= processov <= MAX_PROC:
        raise ValueError(f"процессов 1…{MAX_PROC}")
    kubs = {s: load_kub(s, rz) for s in sids}
    ctx = dict(kubs=kubs, root=Path(root), ist=ist, kod_slit=kod_slit, fast=fast, v=v, rz=rz,
               zakryt={s: zakrytye(root, s) for s in sids})
    n1s = n1 if isinstance(n1, dict) else dict.fromkeys(sids, n1)
    zad = [(s, a, min(a + KUSOK, n1s[s])) for s in sids for a in range(n0, n1s[s], KUSOK)]
    fk, itog = mp.get_context("fork"), {s: dict(n=0, oshibok=0, t=[], rss=0.0) for s in sids}
    with cf.ProcessPoolExecutor(processov, mp_context=fk, initializer=_init, initargs=(ctx, fk.Value("i", 0))) as ex:
        for sid, n, ob, t, rss in ex.map(schitat_kusok, zad):
            a = itog[sid]
            a["n"], a["oshibok"], a["t"], a["rss"] = a["n"] + n, a["oshibok"] + ob, a["t"] + t, max(a["rss"], rss)
    return itog


def zapisat_zamer(itog: dict, processov: int) -> Path:
    """IG-5: только время и RSS, метрик нет."""
    p = ZAMER / f"zamer_prohod0_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategiya", "tochek", "s_na_tochku_p50", "s_na_tochku_p90", "rss_mb_max", "processov"])
        for s, a in itog.items():
            q = np.percentile(a["t"], [50, 90]) if a["t"] else [float("nan")] * 2
            w.writerow([s, a["n"], f"{q[0]:.3f}", f"{q[1]:.3f}", f"{a['rss']:.0f}", processov])
    return p


def sverka(n: int, root: Path, fast: bool, rz: Rezhim = R0) -> int:
    """IG-8: n случайных строк (seed 20261028) пересчитываются; порт, tik и поля §1.2 (кроме t_sek, kod, kod_slit) — побайтно."""
    chs = {s: lm.Chitatel(root, s, sverka_kod=True) for s in SIDS if (Path(root) / s / "manifest.json").exists()}
    pool = [(s, r) for s, ch in chs.items() for r in ch.rows() if r.get("sobol_i") is not None]
    vybor, bad = random.Random(20261028).sample(pool, min(n, len(pool))), []
    for sid, r in vybor:
        sr, c = sreda(sid, rz), load_kub(sid, rz)
        i, pt = tochki(c, r["sobol_i"], r["sobol_i"] + 1, rz)[0]
        r2, ex_p, ex_t = schitat_tochku(sr, sid, i, pt, r["istochnik"], r["kod_slit"], fast, chs[sid].v, rz)
        sig = lambda x: json.dumps({k: v for k, v in x.items() if k not in SVERKA_PROPUSK}, sort_keys=True)  # noqa: E731
        if chs[sid].v == lm.V21:        # суммы сегментов, float64, побайтно
            pz = sr.evr.poz if chs[sid].manifest.get("poz_konec") else None
            pp, pt_, sp, st = pz or (None,) * 4
            ok = sig(r) == sig(r2) and chs[sid].summy([r])[0].tobytes() == lm.summy(sr.daty, ex_p, pp, sp).tobytes() \
                and chs[sid].summy([r], "tik")[0].tobytes() == lm.summy(sr.daty, np.asarray(ex_t, float), pt_, st).tobytes()
        else:
            ok = sig(r) == sig(r2) and np.asarray(chs[sid].matrica([r])[:, 0], float).tobytes() == np.asarray(ex_p, float).tobytes() \
                and chs[sid].tik([r])[0].tobytes() == np.asarray(ex_t).astype(np.float32).tobytes()
        if not ok:
            bad.append((sid, r["sobol_i"]))
    print(f"IG-8: сверено {len(vybor)}, расхождений {len(bad)}" + (f": {bad}" if bad else ""))
    return 1 if bad or len(vybor) < n else 0


def rezhim(a, sids: list, ap) -> tuple:
    """(Rezhim, корень ledger): progon — прежний; v5 — fdr из куба (у всех стратегий один), свой каталог на период × seed | k3."""
    if a.kub == "progon":
        if a.period != PERIOD or a.seed != 1 or a.k3 or a.n1:
            ap.error("--period/--seed/--k3/--n1 — только для --kub v5")
        return R0, a.root or LEDGER
    fdrs = {load_kub(s, R0._replace(setka=a.kub)).get("fdr") for s in sids}
    if len(fdrs) != 1:
        ap.error(f"fdr в кубах разный: {fdrs}")
    if a.k3 and (a.seed != 1 or a.n1):
        ap.error("k3 — полная решётка: без --seed 2 и --n1")
    rz = Rezhim(a.kub, a.period, "seed2" if a.seed == 2 else "seed", a.k3, fdrs.pop())
    if a.kub == "v6":              # v6: заморозка Морриса всем проходам, кроме k3 (контроль на кубе v5: новые оси «выкл.»)
        zk = getattr(a, "zamorozka", None)
        if a.k3 == bool(zk):
            ap.error("v6: --zamorozka <каталог Морриса> нужна всем проходам, кроме k3 (у k3 её нет)")
        rz = rz._replace(rol="kontrol" if a.k3 else "sobol", zam=() if a.k3 else v6.zamorozka(zk, sids, lambda s: kub_put(s, rz)))
    return rz, (a.root or {"v5": LEDGER5, "v6": LEDGER6}[a.kub]) / f"{a.period}_{'k3' if a.k3 else f's{a.seed}'}"


def n1_po_sid(kubs: dict, arg, rz: Rezhim) -> dict:
    """{sid: N1}: k3 — объём решётки; иначе --n1 (число или S1=…,S2=…), где не задан — sobol.N1 куба."""
    if rz.k3:
        return {s: obem_k3(c, rz) for s, c in kubs.items()}
    over = {} if not arg else {"*": int(arg)} if arg.isdigit() else {k: int(v) for k, v in (x.split("=") for x in arg.split(","))}
    if set(over) - set(kubs) - {"*"}:
        raise ValueError(f"--n1: стратегии {sorted(set(over) - set(kubs) - {'*'})} не в прогоне")
    return {s: over.get(s, over.get("*", c["sobol"]["N" if rz.setka == "v6" else "N1"])) for s, c in kubs.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Соболь-драйвер прогона этапа 1 (счёт — только по команде дирижёра)")
    ap.add_argument("--prohod", type=int, choices=[0, 1, 2], help="0 — пробный (IG-5), 1 — точки 0…4999, 2 — 5000…29999")
    ap.add_argument("--tochek", type=int, default=200, help="для --prohod 0: точек на стратегию")
    ap.add_argument("--processov", type=int, default=MAX_PROC, help=f"процессов, не больше {MAX_PROC}")
    ap.add_argument("--sverka", type=int, help="IG-8: пересчитать N случайных строк ledger и сверить побайтно")
    ap.add_argument("--kod-slit", action="store_true", help="kod_slit = true (по умолчанию false: код ветки не слит)")
    ap.add_argument("--fast", action="store_true", help="быстрый слой (Ж-083) для покрытых точек; по умолчанию медленный путь")
    ap.add_argument("--ledger", choices=[lm.V, lm.V21], default=lm.V21, help="формат ledger: 2.1 — суммы сегментов (по умолчанию), 2.0 — ряды")
    ap.add_argument("--strategii", default=",".join(SIDS))
    ap.add_argument("--root", type=Path, help=f"по умолчанию {LEDGER} (progon) или {LEDGER5}/<период>_<s1|s2|k3> (v5)")
    ap.add_argument("--kub", choices=["progon", "v5", "v6"], default="progon", help="progon — прежний путь; v5 — куб Ж-127; v6 — куб Ж-132, ledger 2.2")
    ap.add_argument("--zamorozka", type=Path, help="v6: каталог Морриса (<Sk>/zamorozka.yaml); всем проходам, кроме k3")
    ap.add_argument("--period", choices=sorted(OKNO), default=PERIOD, help="v5: train4 (В1) или v2 (Ж-113)")
    ap.add_argument("--seed", type=int, choices=[1, 2], default=1, help="v5: 1 — sobol.seed, 2 — sobol.seed2 (Ж-130)")
    ap.add_argument("--k3", action="store_true", help="v5: проход к3 — полная решётка раздела k3 (оси v4)")
    ap.add_argument("--n1", help="v5: N1 вместо куба — число для всех или S1=65536,S2=65536")
    a = ap.parse_args(argv)
    sids = a.strategii.split(",")
    rz, root = rezhim(a, sids, ap)
    a.ledger = lm.V22 if rz.setka == "v6" else a.ledger        # v6 пишет только ledger 2.2 (setka, rol, klyuch_n)
    if a.sverka:
        return sverka(a.sverka, root, a.fast, rz)
    if a.prohod is None:
        ap.error("нужен --prohod или --sverka")
    kubs = {s: load_kub(s, rz) for s in sids}
    if rz.setka == "progon":
        [proverit_kub(c) for c in kubs.values()]
        sb = kubs[sids[0]]["sobol"]
        p1 = sb.get("prohod1", sb.get("N1"))        # куб v: 3 объявляет N1
        p2 = sb.get("prohod2", p1)
        n0, n1 = {0: (0, a.tochek), 1: (0, p1), 2: (p1, p2)}[a.prohod]
    else:
        if a.prohod == 2:
            ap.error("куб v5: один проход [0, N1) — --prohod 1")
        [proverit_v5(c, rz) for c in kubs.values()]
        proverit_ledger(rz, a.ledger)
        n0, n1 = 0, (dict.fromkeys(sids, a.tochek) if a.prohod == 0 else n1_po_sid(kubs, a.n1, rz))
    root = root / "_probe" if a.prohod == 0 else root
    n1s = n1 if isinstance(n1, dict) else dict.fromkeys(sids, n1)
    proverit_disk(sum(v - n0 for v in n1s.values()), root, BAIT_V21 if a.ledger in lm.SUMMY_V else BAIT_NA_TOCHKU)
    ist = ("k3" if rz.k3 else f"sobol1_seed{a.seed}") if rz.setka != "progon" else "sobol2" if a.prohod == 2 else "sobol1"
    itog = schet(sids, n0, n1, ist, root, a.processov, a.kod_slit, a.fast, a.ledger, rz)
    for s, v in itog.items():
        print(f"{s}: посчитано {v['n']}, ошибок {v['oshibok']}, RSS до {v['rss']:.0f} МБ")
    if a.prohod == 0:
        print("замер:", zapisat_zamer(itog, a.processov))
    return 2 if any(v["oshibok"] for v in itog.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
