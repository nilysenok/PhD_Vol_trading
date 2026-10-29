"""Л-6: драйвер Соболя прогона этапа 1 (docs/otchety/2026-10-28_progon_etap1.md §1, §3 IG-5/6/8, §4; Ж-082, Ж-083).
Куб — fastbt/designs/progon/kub_S{k}.yaml; точка i — i-я точка Соболя (scramble, seed), потом canon_v2 (N — один раз).
Проход 1 — индексы 0…4999, проход 2 — 5000…29999 той же последовательности (fast_forward). 10 процессов, куски по 100 точек,
у процесса свой p{NN}/. Резюм: klyuch_sha, уже закрытые в ledger, не считаются. Путь: медленный (EvaluatorV2, эталон)
или быстрый (--fast: fast_etap1.patched() для покрытых точек). Метрик не печатает."""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.util
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

from fastbt import ledger_v2 as lm

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
fabrika = None      # sid -> Sreda; None — реальная (fabrika_real); тесты подставляют синтетику
_W: dict = {}


def load_kub(sid: str) -> dict:
    return yaml.safe_load(KUB.with_name(f"kub_{sid}.yaml").read_text(encoding="utf-8"))


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


def tochki(kub: dict, n0: int, n1: int) -> list:
    """[(i, pt)], i ∈ [n0, n1): индекс значения оси = floor(u·len(ось)), затем canon_v2."""
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


def fabrika_real(sid: str) -> Sreda:
    from fastbt import cache, etap1
    from fastbt.etap1_eval import years_of
    from fastbt.etap1_eval_v2 import EvaluatorV2
    inp = etap1.build_inputs(sid, period=PERIOD)
    evr = EvaluatorV2(sid, inp, etap1.base_point(sid), KOMISSIYA)
    dt = evr._engine(PERIOD)[0].dates
    daty = dt[np.isin(dt.year, years_of(PERIOD))].to_numpy().astype("datetime64[D]")
    return Sreda(evr, cache.code_hash(), inp.digest(), daty, list(inp.sk.tickers))


def sreda(sid: str) -> Sreda:
    """Один оценщик на стратегию за раз; даты ≥ 2020 в рядах — отказ."""
    if _W.get("sid") != sid:
        _W.pop("sreda", None)
        sr = (fabrika or fabrika_real)(sid)
        if sr.daty.size == 0 or sr.daty.max() >= np.datetime64("2020-01-01"):
            raise ValueError(f"{sid}: в рядах есть даты ≥ 2020 — отказ (Ж-082)")
        _W["sid"], _W["sreda"] = sid, sr
    return _W["sreda"]


def manifest(sid: str, sr: Sreda, c: dict, kod_slit: bool) -> dict:
    return {"period": PERIOD, "okno_s": BAZA["okno_s"], "okno_po": BAZA["okno_po"], "progrev_s": BAZA["progrev_s"],
            "komissiya": KOMISSIYA, "fdr": None, "kod": sr.kod, "vhody": sr.vhody, "kod_slit": kod_slit,
            "daty_sha": hashlib.sha256(sr.daty.tobytes()).hexdigest(), "tikery": sr.tikery, "T": int(sr.daty.size),
            "kub_sha": hashlib.sha256(KUB.with_name(f"kub_{sid}.yaml").read_bytes()).hexdigest(),
            "sobol_seed": c["sobol"]["seed"], "sozdan": time.strftime("%Y-%m-%d")}


def sha_tochki(sid: str, sr: Sreda, pt: dict) -> str:
    return lm.klyuch_sha(lm.klyuch(sid, pt, PERIOD, KOMISSIYA, None), sr.kod, sr.vhody)


def fast_ok(pt: dict) -> bool:
    return importlib.util.find_spec("fastbt.fast_etap1") is not None and \
        importlib.import_module("fastbt.fast_etap1").pokryto(pt)


def schitat_tochku(sr: Sreda, sid: str, i: int, pt: dict, ist: str, kod_slit: bool, fast: bool, v: str = lm.V) -> tuple:
    """(строка §1.2, port float64, tik): ошибка оценщика — строка `oshibka`, а не обрыв прогона."""
    t0, sloy, r, ex_p, ex_t = time.perf_counter(), "slow", {}, None, None
    try:
        if fast and fast_ok(pt):
            with importlib.import_module("fastbt.fast_etap1").patched():
                r, ex_p, ex_t = sr.evr.evaluate_v2(pt, PERIOD)
            sloy = "fast"
        else:
            r, ex_p, ex_t = sr.evr.evaluate_v2(pt, PERIOD)
    except Exception as e:                      # noqa: BLE001 — контракт §1.2: status = oshibka, причина в строке
        r = {"status": "oshibka", "prichina": f"{type(e).__name__}: {e}"[:200]}
    kl = lm.klyuch(sid, pt, PERIOD, KOMISSIYA, None)
    r = {**BAZA, **r, "v": v, "format": lm.FORMAT, "strategiya": sid, "klyuch": kl, "kod": sr.kod,
         "klyuch_sha": lm.klyuch_sha(kl, sr.kod, sr.vhody), "vhody": sr.vhody, "kod_slit": kod_slit,
         "period": PERIOD, "komissiya": KOMISSIYA, "fdr": None, "istochnik": ist, "sobol_i": i, "kontrol": None,
         "vozm": None, "sloy": sloy, "t_sek": time.perf_counter() - t0}
    return r, ex_p, ex_t


def _init(ctx: dict, schetchik) -> None:
    _W.clear()
    _W.update(ctx)
    with schetchik.get_lock():
        _W["nn"], schetchik.value = schetchik.value, schetchik.value + 1


def schitat_kusok(zad: tuple) -> tuple:
    sid, n0, n1 = zad
    c, sr = _W["kubs"][sid], sreda(sid)
    d = Path(_W["root"]) / sid
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "daty.npy").exists():
        np.save(d / f"daty.npy.{_W['nn']}.tmp.npy", sr.daty)
        (d / f"daty.npy.{_W['nn']}.tmp.npy").replace(d / "daty.npy")
    v = _W.get("v", lm.V)
    mf = manifest(sid, sr, c, _W["kod_slit"])
    poz_ok = hasattr(sr.evr, "poz")                                    # Л7b: настоящий оценщик отдаёт |pos| (EvaluatorV2.poz)
    w = lm.Pisatel21(_W["root"], sid, _W["nn"], mf, daty=sr.daty, poz=poz_ok) if v == lm.V21 else lm.Pisatel(_W["root"], sid, _W["nn"], mf)
    zak, t, oshibok = _W["zakryt"][sid] | w.done(), [], 0
    for i, pt in tochki(c, n0, n1):
        if sha_tochki(sid, sr, pt) in zak:
            continue
        r, ex_p, ex_t = schitat_tochku(sr, sid, i, pt, _W["ist"], _W["kod_slit"], _W["fast"], v)
        importlib.import_module("fastbt.zaem_v2").pisat(w, r, ex_p, ex_t, getattr(sr.evr, "zaem", None),
                                                           sr.evr.poz if poz_ok and v == lm.V21 and r["status"] == "ok" else None)   # Ж-089: ряд платы за заём; Л7b: |pos|
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


def schet(sids, n0: int, n1: int, ist: str, root: Path, processov: int, kod_slit: bool, fast: bool, v: str = lm.V) -> dict:
    """Все sids × [n0, n1) куски по 100 в очередь на `processov` процессов. {sid: {n, oshibok, t, rss}}."""
    if not 1 <= processov <= MAX_PROC:
        raise ValueError(f"процессов 1…{MAX_PROC}")
    kubs = {s: load_kub(s) for s in sids}
    ctx = dict(kubs=kubs, root=Path(root), ist=ist, kod_slit=kod_slit, fast=fast, v=v,
               zakryt={s: zakrytye(root, s) for s in sids})
    zad = [(s, a, min(a + KUSOK, n1)) for s in sids for a in range(n0, n1, KUSOK)]
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


def sverka(n: int, root: Path, fast: bool) -> int:
    """IG-8: n случайных строк (seed 20261028) пересчитываются; порт, tik и поля §1.2 (кроме t_sek, kod, kod_slit) — побайтно."""
    chs = {s: lm.Chitatel(root, s, sverka_kod=True) for s in SIDS if (Path(root) / s / "manifest.json").exists()}
    pool = [(s, r) for s, ch in chs.items() for r in ch.rows() if r.get("sobol_i") is not None]
    vybor, bad = random.Random(20261028).sample(pool, min(n, len(pool))), []
    for sid, r in vybor:
        sr, c = sreda(sid), load_kub(sid)
        i, pt = tochki(c, r["sobol_i"], r["sobol_i"] + 1)[0]
        r2, ex_p, ex_t = schitat_tochku(sr, sid, i, pt, r["istochnik"], r["kod_slit"], fast, chs[sid].v)
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
    ap.add_argument("--root", type=Path, default=LEDGER)
    a = ap.parse_args(argv)
    sids = a.strategii.split(",")
    if a.sverka:
        return sverka(a.sverka, a.root, a.fast)
    if a.prohod is None:
        ap.error("нужен --prohod или --sverka")
    kubs = [load_kub(s) for s in sids]
    [proverit_kub(c) for c in kubs]
    sb = kubs[0]["sobol"]
    p1 = sb.get("prohod1", sb.get("N1"))        # куб v: 3 объявляет N1
    p2 = sb.get("prohod2", p1)
    n0, n1 = {0: (0, a.tochek), 1: (0, p1), 2: (p1, p2)}[a.prohod]
    root = a.root / "_probe" if a.prohod == 0 else a.root
    proverit_disk(len(sids) * (n1 - n0), root, BAIT_V21 if a.ledger == lm.V21 else BAIT_NA_TOCHKU)
    itog = schet(sids, n0, n1, "sobol2" if a.prohod == 2 else "sobol1", root, a.processov, a.kod_slit, a.fast, a.ledger)
    for s, v in itog.items():
        print(f"{s}: посчитано {v['n']}, ошибок {v['oshibok']}, RSS до {v['rss']:.0f} МБ")
    if a.prohod == 0:
        print("замер:", zapisat_zamer(itog, a.processov))
    return 2 if any(v["oshibok"] for v in itog.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
