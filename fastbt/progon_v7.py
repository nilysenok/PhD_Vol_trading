"""Серверный Соболь по кубу v7 (Ж-155 п. 7, Ж-156 пп. 2–4; лист V7-8a): надстройка над progon_server (счёт, flock, чекпоинт, --prodolzhit).
Данных не читает. Что добавляет v7: setka v7 (ledger 2.2, manifest.setka = c["setka"] куба), период `polnaya` (прогрев 26.08.2014, оценка 2017–27.02.2026,
зона данных wf), чекпоинт каждые SHAG = 10 000 точек (progon_server.CHEKPOINT = 5 000 — на время прохода подменяется, v5/v6 не затронуты),
пилот `--n1 1000` по каждому семейству с замером в pilot_zamer.json, к3 v7 = ссылка на ledger v6 (k3_ssylka.json), не пересчёт.
Куб и точки — fastbt.canon_v7 (S1–S6) и canon_v7b (S8–S12): load_kub, proverit, один сэмплер tochki_v7 (для S8–S12 через tochki_v7b). Запуск: python3 -m fastbt.progon_v7 --kub v7 --period polnaya --seed 1 --n1 1000 --out <каталог> --pilot."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from fastbt import ledger_v2 as lm, period_v2, progon_server as srv, progon_sobol as ps

SETKA, PERIOD, SHAG, PILOT_N, ZAMER = "v7", "polnaya", 10_000, 1000, "pilot_zamer.json"
KUB7 = ps.ROOT / "fastbt/designs/v7/kub_{sid}.yaml"
LEDGER7 = ps.ZAMER / "_ledger_v7"
SEMEJSTVA = ("S1", "S2", "S3", "S4", "S5", "S6", "S8", "S9", "S10", "S11", "S12")      # S7 в v7 не входит (Ж-153 п. 2)
SEMEJSTVA_IST7 = ("S8", "S9", "S10", "S11", "S12")      # семьям нужны Ist7 (10-минутки, панель сечения); S1–S6 — ряды v6
CANON = "fastbt.canon_v7"
PILOT1 = "пилот-1"
NULI = ("min_uderzhanie", "min_hold")            # константы, у которых «выкл.» = 0; остальные оси — "net" (Ж-160 п. 3)


def fiksirovat(tf, osi):
    """Ось из --bez-osej подменяется на «выкл.» (0 / "net") в точке до оценки; ключ точки — по подменённой. Оси нет в точке — не добавляется."""
    osi = tuple(osi)
    if not osi:
        return tf
    return lambda kub, n0, n1, rz: [(i, {**pt, **{o: (0 if o in NULI else "net") for o in osi if o in pt}}) for i, pt in tf(kub, n0, n1, rz)]


def min_uder(sid: str, bez_osej=()) -> int:
    """Константа `min_uderzhanie.dney` из kub_S{k}.yaml (нет в кубе — 0); `--bez-osej min_uderzhanie` — 0. Ж-153 §1.1: EvaluatorV7 сам обнуляет не-контртренд."""
    import yaml
    m = yaml.safe_load(kub_put7(sid).read_text(encoding="utf-8")).get("min_uderzhanie")
    return 0 if "min_uderzhanie" in bez_osej or not m else int(m["dney"])


def ocenshchik7(staryj, bez_osej=()):
    """ps.ocenshchik для setka v7: EvaluatorV7 с рядами v6 и константой min_uderzhanie из куба (по умолчанию в классе 0 — константа иначе не доходит)."""
    def f(sid, inp, rz):
        if rz.setka != SETKA:
            return staryj(sid, inp, rz)
        from fastbt import eval_v7_ist, etap1, istochniki_v6
        from fastbt.eval_v7 import PROXY, EvaluatorV7
        from fastbt.eval_v7_otn import SEMYI_OTN
        tik = list(inp.sk.tickers)
        kw = {"ist7": eval_v7_ist.zagruzit(rz.period, tik, dev=rz.period == "train4").ist7()} if sid in SEMEJSTVA_IST7 + SEMYI_OTN else {}     # S8–S12: источники v7; S1–S4: IMOEX и карта секторов оси otnositelno
        return EvaluatorV7(sid, inp, etap1.base_point(PROXY.get(sid, sid)), ps.KOMISSIYA, fdr=rz.fdr, ist=istochniki_v6.zagruzit(rz.period, tik),
                           min_uderzhanie=min_uder(sid, bez_osej), **kw)
    return f


def kvantili(t) -> tuple:
    """(p50, p90) секунд на точку; пусто — (None, None)."""
    return (float(np.percentile(t, 50)), float(np.percentile(t, 90))) if len(t) else (None, None)


def canon7():
    try:
        return importlib.import_module(CANON)
    except ModuleNotFoundError as e:
        if e.name != CANON:
            raise
        raise SystemExit(f"{CANON} (лист V7-1b) не слит — куб и точки v7 читать нечем — отказ") from e


def semya(sid: str):
    """Модуль канона семьи: S1–S6 — canon_v7, S8–S12 — canon_v7b (оба с load_kub, proverit)."""
    cn = canon7()
    return importlib.import_module(CANON + "b") if sid in importlib.import_module(CANON + "b").SIDS_B else cn


def tochki_semya(kub: dict, n0: int, n1: int, rz) -> list:
    """[(i, точка)] — единственный сэмплер Соболя v7 (canon_v7.tochki_v7; для S8–S12 через canon_v7b.tochki_v7b), оси v7 открыты (GOTOVY_V7)."""
    m = semya(kub["sid"])
    if kub["sid"] in getattr(m, "RESHETKA", ()):          # S8/S9: вся решётка по разу, срез [n0, n1) для кусков
        return [x for x in m.reshetka(kub, canon7().GOTOVY_V7) if n0 <= x[0] < n1]
    f = m.tochki_v7b if hasattr(m, "tochki_v7b") else m.tochki_v7
    return f(kub, n0, n1, rz.seed, canon7().GOTOVY_V7)[0]


def kub_put7(sid: str) -> Path:
    return KUB7.with_name(f"kub_{sid}.yaml")


def zona_ctx(period: str):
    """wf — зона wf (данные по 2026-02-28); polnaya — своя зона polnaya: чтение строго < period_v2.chtenie_konec (до акта 2025-01-01).
    Замок — отметка владельца, иначе ZamokVladeltsa."""
    from fastbt import zona
    return zona.zone_ctx(period) if period in ("wf", PERIOD) else contextlib.nullcontext()


def rezhim7(seed: int, fdr, rol: str = "sobol", period: str = PERIOD) -> ps.Rezhim:
    return ps.Rezhim(SETKA, period, "seed2" if seed == 2 else "seed", False, fdr, rol)


def zagruzit_kubi(sids, seed: int) -> dict:
    kubs = {}
    for s in sids:
        cn = semya(s)
        c = cn.load_kub(s, kub_put7(s).parent)
        cn.proverit(c)
        if c.get("setka") != SETKA:
            raise SystemExit(f"{kub_put7(s)}: setka = {c.get('setka')!r}, ожидалась {SETKA!r} — отказ")
        kubs[s] = (kub_put7(s), c)
    return kubs


def n_po_sid(kubs: dict, arg) -> dict:
    """{sid: N}: --n1 (число или S1=…,S2=…), где не задан — sobol.N куба (v7, как v6)."""
    over = {} if not arg else {"*": int(arg)} if str(arg).isdigit() else {k: int(v) for k, v in (x.split("=") for x in arg.split(","))}
    if set(over) - set(kubs) - {"*"}:
        raise ValueError(f"--n1: стратегии {sorted(set(over) - set(kubs) - {'*'})} не в прогоне")
    n = {s: over.get(s, over.get("*", c["sobol"]["N"])) for s, (_, c) in kubs.items()}
    for s, (_, c) in kubs.items():                       # S8/S9: фактическое число = вся решётка, не --n1 (Ж-161)
        if s in getattr(semya(s), "RESHETKA", ()):
            n[s] = len(semya(s).reshetka(c, canon7().GOTOVY_V7))
    return n


def fabrika7(sid: str, rz: ps.Rezhim):
    """ps.fabrika_real для v7: входы (SkEvaluator, SIG_GRIDS) у S8–S12 строятся под прокси S3 — своих сигнальных сеток у семей v7 в sigprov нет;
    оценщик — под настоящим sid. Вход в manifest ('vhody', Д5) для S8–S12 включает sha источников v7."""
    from fastbt import cache, etap1
    from fastbt.eval_v7 import PROXY
    from fastbt.etap1_eval import years_of
    inp = etap1.build_inputs(PROXY.get(sid, sid), period=rz.period)
    evr = ps.ocenshchik(sid, inp, rz)
    dt = evr._engine(rz.period)[0].dates
    daty = dt[np.isin(dt.year, years_of(rz.period))].to_numpy().astype("datetime64[D]")
    vh = inp.digest() if sid not in SEMEJSTVA_IST7 else f"{inp.digest()}:{evr.ist7.sha}"
    return ps.Sreda(evr, cache.code_hash(), vh, daty, list(inp.sk.tickers))


def sreda7(sid: str, rz: ps.Rezhim = ps.R0):
    """ps.sreda для polnaya: верхняя граница дат — конец периода + 1 (а не 2020, как у прежних), только с отметкой владельца."""
    if ps._W.get("sid") != (sid, rz):
        ps._W.pop("sreda", None)
        sr = ps.fabrika(sid) if ps.fabrika else fabrika7(sid, rz)
        if rz.period != "train4":                          # train4 (сухой прогон до листа «зона») — разработочный, отметка не нужна
            period_v2.proverit_chtenie("wf")
        verh = str(period_v2.chtenie_konec(rz.period).date())          # train4/wf: konec + 1 день; polnaya: граница чтения до акта
        if sr.daty.size == 0 or sr.daty.max() >= np.datetime64(verh):
            raise ValueError(f"{sid}: в рядах есть даты ≥ {verh} (максимум {sr.daty.max() if sr.daty.size else 'нет дат'}) — отказ")
        if getattr(sr.evr, "fdr", "нет атрибута fdr") != rz.fdr:
            raise ValueError(f"{sid}: fdr оценщика ≠ fdr куба {rz.fdr!r} — отказ")
        ps._W["sid"], ps._W["sreda"] = (sid, rz), sr
    return ps._W["sreda"]


@contextlib.contextmanager
def podmeny(zam: dict, shag: int, tochki_fn=None, bez_osej=(), meta=None):
    """На время прохода: путь куба, точки, manifest (+ setka_data, zamorozka_sha), среда, шаг чекпоинта. Дети (fork) наследуют; выход — всё назад."""
    staryj, man, kp = (ps.kub_put, ps.tochki, ps.manifest, ps.sreda, srv.CHEKPOINT), ps.manifest, ps.kub_put
    tf = tochki_fn or tochki_semya                 # zam (заморозка Морриса) в сэмплер Соболя не входит — только в manifest
    tf = fiksirovat(tf, bez_osej)
    ps.kub_put = lambda sid, rz=ps.R0: kub_put7(sid) if rz.setka == SETKA else kp(sid, rz)
    ps.tochki = lambda kub, n0, n1, rz=ps.R0: tf(kub, n0, n1, rz) if rz.setka == SETKA else staryj[1](kub, n0, n1, rz)
    ps.manifest = lambda sid, sr, c, ks, rz=ps.R0: ({**man(sid, sr, c, ks, rz), "setka_data": str(c.get("data")), "zamorozka_sha": zam.get(sid), **(meta or {})}
                                                    if rz.setka == SETKA else man(sid, sr, c, ks, rz))
    ps.sreda, srv.CHEKPOINT = (lambda sid, rz=ps.R0: sreda7(sid, rz) if rz.setka == SETKA else staryj[3](sid, rz)), shag
    oc, ps.ocenshchik = ps.ocenshchik, ocenshchik7(ps.ocenshchik, bez_osej)
    try:
        yield
    finally:
        ps.kub_put, ps.tochki, ps.manifest, ps.sreda, srv.CHEKPOINT = staryj
        ps.ocenshchik = oc


def progon7(kubs: dict, out, procs: int, n: dict, seed: int = 1, prodolzhit=False, shag: int = SHAG, fast=False, kod_slit=False,
            zam=None, tochki_fn=None, rezhim=srv.REZHIM, bez_osej=(), meta=None, period: str = PERIOD) -> dict:
    """{sid: n, oshibok, sboy, posledniy_i, deti} — srv.progon на rz setka v7, rol sobol, ledger 2.2, чекпоинт каждые shag точек."""
    fdrs = {c.get("fdr") for _, c in kubs.values()}
    if len(fdrs) != 1:
        raise SystemExit(f"fdr в кубах разный: {fdrs}")
    rz = rezhim7(seed, fdrs.pop(), period=period)
    with podmeny(zam or {}, shag, tochki_fn, bez_osej, meta):
        return srv.progon(kubs, out, procs, n, prodolzhit, kod_slit, fast, rezhim=rezhim, rz=rz, ledger=lm.V22)


def razmer(d: Path) -> int:
    return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())


def zamer_semejstva(root: Path, sid: str, n: int, sek: float) -> dict:
    """Секунды на точку (медиана и среднее t_sek строк ledger; стенка/n — с накладными), размер ledger на n точек и на 1 000, пик RSS."""
    t = [json.loads(x)["t_sek"] for p in sorted((Path(root) / sid).glob("p[0-9]*/rows.jsonl")) for x in p.read_text(encoding="utf-8").splitlines()]
    mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / (1e6 if sys.platform == "darwin" else 1e3)
    try:
        import psutil
        uss = psutil.Process().memory_full_info().uss / 1e6
    except Exception:  # noqa: BLE001 — psutil не обязателен; пустое поле честнее выдуманного
        uss = None
    b = razmer(Path(root) / sid)
    p50, p90 = kvantili(t)
    return dict(n=n, strok=len(t), sek_na_tochku_p50=p50, sek_na_tochku_p90=p90, sek_na_tochku_sredn=float(np.mean(t)) if t else None, sek_na_tochku_mediana=float(np.median(t)) if t else None,
                sek_stenka_na_tochku=sek / max(n, 1), rss_pik_mb=round(mb, 1), uss_roditelya_mb=None if uss is None else round(uss, 1),
                ledger_bajt=b, ledger_bajt_na_1000=int(b * 1000 / max(n, 1)),
                ledger_gb_na_1000=int(b * 1000 / max(n, 1)) / 1e9)


def pilot(kubs: dict, out: Path, procs: int, n: dict, bez_osej=(), semeystva=None, **kw) -> dict:
    """Ж-156 п. 2а: каждое семейство отдельным проходом (замер по семействам) → <out>/pilot_zamer.json. RSS — накопленный максимум процессов.
    Пилот-1 (Ж-160 п. 3): bez_osej фиксируются в «выкл.»; пометка pilot/osi_otsutstvuyut/semeystva — в замере и manifest каждого семейства."""
    res, bez = {}, list(bez_osej)
    pom = dict(pilot=PILOT1, osi_otsutstvuyut=bez, semeystva=list(semeystva or kubs))
    for s in kubs:
        t0 = time.perf_counter()
        itog = progon7({s: kubs[s]}, out, procs, {s: n[s]}, bez_osej=bez, meta=pom, **kw)
        res[s] = {**zamer_semejstva(out, s, itog[s]["n"], time.perf_counter() - t0), "sboy": itog[s]["sboy"], **pom}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / ZAMER).write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return res


def k3_ssylka(out: Path, v6_root: Path, sids) -> Path:
    """к3 v7 = ссылка на ledger v6 (план §2): путь и sha manifest.json каждой стратегии; нет manifest v6 — отказ."""
    s = {}
    for sid in sids:
        mf = Path(v6_root) / sid / "manifest.json"
        if not mf.exists():
            raise SystemExit(f"{mf}: нет — к3 v6 не посчитан, ссылаться не на что — отказ")
        s[sid] = dict(put=str(Path(v6_root) / sid), manifest_sha=hashlib.sha256(mf.read_bytes()).hexdigest())
    p = Path(out) / "k3_ssylka.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return p


def razbor(argv=None):
    ap = argparse.ArgumentParser(description="Соболь по кубу v7 (период polnaya, ledger 2.2, чекпоинт 10 000)")
    ap.add_argument("--kub", required=True, choices=[SETKA])
    ap.add_argument("--period", required=True, choices=[PERIOD, "train4"], help="train4 — только сухой прогон до листа «зона»")
    ap.add_argument("--seed", type=int, choices=[1, 2], default=1)
    ap.add_argument("--n1", help="N вместо куба — число для всех или S1=1000,S2=1000")
    ap.add_argument("--strategii", "--semeystva", default=",".join(SEMEJSTVA), help="какие семейства считать (S1,S2,…)")
    ap.add_argument("--bez-osej", default="", help="оси/константы в «выкл./net/0» (пилот-1): min_uderzhanie,k_okup,otnositelno")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--procs", type=int, default=srv.procs_po_umolchaniyu())
    ap.add_argument("--chekpoint", type=int, default=SHAG, help="точек между чекпоинтами (v7: 10 000)")
    ap.add_argument("--prodolzhit", action="store_true", help="досчитать недостающие; отказ при чужом kod/vhody/кубе/seed")
    ap.add_argument("--pilot", action="store_true", help=f"пилот: по семейству отдельно, замер в {ZAMER} (обычно с --n1 {PILOT_N})")
    ap.add_argument("--k3-v6", type=Path, help="каталог ledger v6 к3: записать ссылку k3_ssylka.json и выйти")
    ap.add_argument("--zamorozka", type=Path, help="каталог Морриса v7 (<Sk>/zamorozka.yaml)")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--kod-slit", action="store_true")
    return ap.parse_args(argv)


def sids_iz(a) -> list:
    sids = [x for x in a.strategii.split(",") if x]
    if set(sids) - set(SEMEJSTVA):
        raise SystemExit(f"семейства {sorted(set(sids) - set(SEMEJSTVA))} не в v7 ({','.join(SEMEJSTVA)}) — отказ")
    return sids


def main(argv=None) -> int:
    a = razbor(argv)
    sids = sids_iz(a)
    if a.k3_v6:
        print("к3 v7 = ссылка на v6:", k3_ssylka(a.out, a.k3_v6, sids))
        return 0
    from fastbt import sobytiya_e1
    sobytiya_e1.trebovat_svod()
    zam = {s: ps.v6.sha_fajla(a.zamorozka / s / "zamorozka.yaml") for s in sids} if a.zamorozka else {}
    with zona_ctx(a.period):
        kubs = zagruzit_kubi(sids, a.seed)
        n = n_po_sid(kubs, a.n1)
        kw = dict(seed=a.seed, prodolzhit=a.prodolzhit, shag=a.chekpoint, fast=a.fast, kod_slit=a.kod_slit, zam=zam, period=a.period)
        if a.pilot:
            print(json.dumps(pilot(kubs, a.out, a.procs, n, bez_osej=[x for x in a.bez_osej.split(",") if x], semeystva=sids, **kw), ensure_ascii=False, indent=1))
            return 0
        itog = progon7(kubs, a.out, a.procs, n, **kw)
    for s, v in itog.items():
        print(f"{s}: посчитано {v['n']}, ошибок {v['oshibok']}, чекпоинт {v['posledniy_i']}, сбой {v['sboy']}")
    return 2 if any(v["sboy"] or v["oshibok"] for v in itog.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
