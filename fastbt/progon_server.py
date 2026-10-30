"""Ж-131 п. 3–4: серверный запуск Соболя — обёртка над fastbt.progon_sobol (драйвер и ledger_v2 не правятся: Л7b).
obshchie (Linux): оценщик со всеми входами строится ОДИН раз в родителе, gc.freeze, fork — дети читают те же объекты без копий.
prezhniy (macOS): fork без предзагрузки. Стратегия — каталог <out>/<Sk>/ (ledger 2.1; period v2 — 2.0), flock, checkpoint.json
каждые CHEKPOINT точек; сбой стратегии отменяет только её задачи. --prodolzhit: отказ при чужом kod/vhody/кубе/seed/периоде/k3.
Куб v5 (Л10): точки, выброс canon_v5, якоря, k3 — код драйвера (ps.tochki, ps.rezhim, ps.manifest), не копия. Запуск:
python3 -m fastbt.progon_server --kub v5 --period train4|v2 --seed 1|2 [--k3] [--n1 ..] --out <каталог> [--strategii S1,..]
Куб v6 (С-v6, Ж-137): то же с --kub v6 --zamorozka <каталог Морриса> (кроме --k3), ledger 2.2 (setka = v6, rol), см. fastbt/progon_v6.py.
python3 -m fastbt.progon_server --kub <kub_Sk.yaml>.. --period train4 [--seed <sobol.seed>] --out <каталог> (designs/progon)."""
import argparse
import concurrent.futures as cf
import fcntl
import gc
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import uuid
from pathlib import Path

import yaml

from fastbt import canon_v6, ledger_v2 as lm
from fastbt import progon_sobol as ps

CHEKPOINT, CHP, ZAMOK = 5000, "checkpoint.json", ".zamok"
POLYA = ("kod", "vhody", "kub_sha", "sobol_seed", "period", "setka", "k3", "sobol_klyuch", "rol", "t_vyhod", "zamorozka_sha")     # отпечаток: сверяется с чекпоинтом и manifest.json
REZHIM = "obshchie" if sys.platform.startswith("linux") else "prezhniy"


def procs_po_umolchaniyu() -> int:
    return max(1, (os.cpu_count() or 5) - 4)          # Ж-131 п. 4: ядра − 4


def zagruzit_kub(path, period: str, seed=None, rz=None) -> dict:
    """Куб из kub_<sid>.yaml (ps.manifest берёт kub_sha по этому имени); seed — переопределение sobol.seed (путь progon);
    rz (v5) — проверка драйвера proverit_v5: период объявлен в кубе, fdr куба = fdr прохода."""
    path = Path(path)
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.name != f"kub_{c['sid']}.yaml":
        raise SystemExit(f"{path}: файл куба обязан называться kub_{c['sid']}.yaml")
    if rz is not None:
        c = canon_v6.load_kub(c["sid"], path.parent) if rz.setka == "v6" else c      # v6: + основа v5 (ключ "v5")
        ps.proverit_v5(c, rz)
        return c
    if period != ps.PERIOD or c.get("period") != period:
        raise SystemExit(f"период {period!r} (в кубе {c.get('period')!r}): путь designs/progon считает только {ps.PERIOD}; "
                         "период v2 — только с кубом v5 (--kub v5) — отказ")
    if seed is not None:
        c = {**c, "sobol": {**c["sobol"], "seed": int(seed)}}
    ps.proverit_kub(c)
    return c


def zamok(d: Path):
    """flock на каталог стратегии: второй запуск в тот же каталог — отказ; снимается ядром при смерти процесса."""
    d.mkdir(parents=True, exist_ok=True)
    f = open(d / ZAMOK, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit(f"{d}: каталог занят другим запуском ({ZAMOK})")
    return f


def nachalo(d: Path, otp: dict, prodolzhit: bool) -> int:
    """Первая несчитанная точка: 0 или posledniy_i + 1 чекпоинта; отказ при чужом отпечатке или начатом каталоге."""
    chp, mf = d / CHP, d / "manifest.json"
    est = [p for p in (chp, mf) if p.exists()]
    if est and not prodolzhit:
        raise SystemExit(f"{d}: каталог уже начат — нужен --prodolzhit")
    for p in est:
        ex = json.loads(p.read_text(encoding="utf-8"))
        raz = [k for k in POLYA if k in ex and ex[k] != otp.get(k)]
        if raz:
            raise SystemExit(f"{p}: --prodolzhit — отказ, не совпали {raz}")
    return int(json.loads(chp.read_text(encoding="utf-8"))["posledniy_i"]) + 1 if chp.exists() else 0


def pisat_chekpoint(d: Path, otp: dict, posl: int, n: int) -> None:
    tmp, t = d / f".{CHP}.{os.getpid()}.{uuid.uuid4().hex}.tmp", time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp.write_text(json.dumps({**otp, "posledniy_i": posl, "N": n, "shag": CHEKPOINT, "vremya": t}, sort_keys=True), "utf-8")
    os.replace(tmp, d / CHP)


def zadachi(start: dict, n: dict, p1: dict, rezhim: str, ist=None) -> list:
    """(sid, a, b, istochnik): куски ≤ KUSOK, не через границу чекпоинта и конца прохода 1 (sobol1 / sobol2; v5 — ist драйвера)."""
    out = []
    for s, a in start.items():
        while a < n[s]:
            b = min(a + ps.KUSOK, n[s], (a // CHEKPOINT + 1) * CHEKPOINT, *([p1[s]] if a < p1[s] else []))
            out.append((s, a, b, ist or ("sobol1" if a < p1[s] else "sobol2")))
            a = b
    return sorted(out, key=lambda z: (z[1] // CHEKPOINT, z[0], z[1])) if rezhim == "obshchie" else out


def _kusok(zad: tuple) -> tuple:
    """В ребёнке: кусок progon_sobol.schitat_kusok + (pid, id оценщика, адрес буфера дат) для проверки общих массивов."""
    ps._W["ist"] = zad[3]
    res, sr = ps.schitat_kusok(zad[:3]), ps._W["sreda"]
    return res + (os.getpid(), id(sr.evr), sr.daty.__array_interface__["data"][0])


def podgotovka(kubs: dict, out: Path, prodolzhit: bool, kod_slit: bool, itog: dict, zamki: list, rz, v: str) -> tuple:
    """Родитель: оценщик, отпечаток, замок, старт, manifest. Сбой оценщика одной стратегии — в itog, остальные идут;
    отказ сверки (SystemExit) останавливает весь запуск до счёта."""
    pre, otp, start = {}, {}, {}
    for s, (path, c) in kubs.items():
        try:
            ps._W.clear()
            sr = ps.sreda(s, rz)                               # проверка дат < 2020 и fdr — та же, что у драйвера
            mf = ps.manifest(s, sr, c, kod_slit, rz)
            otp[s] = {k: mf[k] for k in POLYA if k in mf}
            zamki.append(zamok(out / s))
            start[s] = nachalo(out / s, otp[s], prodolzhit)
            # manifest.json — до fork: Pisatel пишет его write_text без tmp, два ребёнка на первых кусках одной стратегии
            # читали полупустой файл (JSONDecodeError; test_manifest_iz_roditelya_do_fork)
            pis = {lm.V22: lm.Pisatel22, lm.V21: lm.Pisatel21}.get(v)
            (pis(out, s, 0, mf, daty=sr.daty, poz=hasattr(sr.evr, "poz")) if pis else lm.Pisatel(out, s, 0, mf))
            pre[s] = sr
        except Exception as e:                                 # noqa: BLE001 — сбой стратегии не роняет остальные
            itog[s]["sboy"] = f"{type(e).__name__}: {e}"[:300]
    ps._W.clear()
    return pre, otp, start


def schet(zad: list, ctx: dict, procs: int, itog: dict, pre: dict, d_otp: dict, n: dict) -> None:
    """Один пул на весь прогон; чекпоинт стратегии — по последнему непрерывно закрытому блоку CHEKPOINT точек.
    obshchie: ребёнок обязан видеть объект родителя (тот же id и адрес буфера), иначе fork-путь копирует — стоп."""
    ost = {}
    for s, a, _b, _i in zad:
        ost[s, a // CHEKPOINT] = ost.get((s, a // CHEKPOINT), 0) + 1
    sled = {s: min(a // CHEKPOINT for s2, a, _b, _i in zad if s2 == s) for s in {z[0] for z in zad}}
    fk = mp.get_context("fork")
    with cf.ProcessPoolExecutor(procs, mp_context=fk, initializer=ps._init, initargs=(ctx, fk.Value("i", 0))) as ex:
        fut = {ex.submit(_kusok, z): z for z in zad}
        for f in cf.as_completed(fut):
            (s, a, _b, _i), st = fut[f], itog[fut[f][0]]
            if f.cancelled():
                continue
            if f.exception() is not None:          # блок с упавшим куском не закроется — чекпоинт встанет перед ним
                st["sboy"] = st["sboy"] or f"{type(f.exception()).__name__}: {f.exception()}"[:300]
                [g.cancel() for g, z2 in fut.items() if z2[0] == s]
                continue
            _sid, k, ob, _t, _rss, pid, id_evr, adr = f.result()
            if pre and (pid == os.getpid() or (id_evr, adr) != (id(pre[s].evr), pre[s].daty.__array_interface__["data"][0])):
                raise RuntimeError(f"{s}: ребёнок {pid} видит не те объекты, что родитель — fork-путь копирует входы")
            st["n"], st["oshibok"], ost[s, a // CHEKPOINT] = st["n"] + k, st["oshibok"] + ob, ost[s, a // CHEKPOINT] - 1
            st["deti"].add((pid, id_evr, adr))
            posl = None
            while ost.get((s, sled[s])) == 0:
                posl, sled[s] = min((sled[s] + 1) * CHEKPOINT, n[s]) - 1, sled[s] + 1
            if posl is not None:
                pisat_chekpoint(*d_otp[s], posl, n[s])
                st["posledniy_i"] = posl


def progon(kubs: dict, out, procs: int, n=None, prodolzhit=False, kod_slit=False, fast=False, rezhim=REZHIM, rz=ps.R0, ledger=None) -> dict:
    """kubs {sid: (путь, куб)} → {sid: n, oshibok, sboy, posledniy_i, deti}; n — точек (число или {sid: n}), по умолчанию из куба;
    rz — режим драйвера (R0 — designs/progon; v5 — куб Ж-127: период, seed, k3); ledger None — 2.0 для v2, иначе 2.1;
    "2.0" | "2.1" — явно (2.1 на период v2 — 20 кварталов 2015Q1…2019Q4, лист Б8)."""
    out, sids = Path(out), list(kubs)
    v = ledger or (lm.V22 if rz.rol else lm.V if rz.period == "v2" else lm.V21)
    if v not in (lm.V, lm.V21, lm.V22) or bool(rz.rol) != (v == lm.V22):
        raise SystemExit(f"ledger {v!r}: известны {lm.V}, {lm.V21}; {lm.V22} — только с --rol (и --rol — только с ним)")
    ps.proverit_ledger(rz, v)
    if rz.setka == "v6" and (not rz.rol or v != lm.V22 or (not rz.k3 and not rz.zam)):
        raise SystemExit("куб v6: ledger 2.2 с rol (sobol | kontrol | spravka) и заморозкой Морриса (кроме k3) — отказ")
    if procs < 1 or rezhim not in ("obshchie", "prezhniy") or len({Path(p).parent for p, _ in kubs.values()}) > 1:
        raise SystemExit("procs ≥ 1; rezhim obshchie | prezhniy; кубы — из одного каталога")
    itog = {s: dict(n=0, oshibok=0, sboy=None, posledniy_i=None, deti=set()) for s in sids}
    staryj, zamki = (ps.fabrika, ps.KUB, ps.KUB5, ps.KUB6), []
    ps.KUB = ps.KUB5 = ps.KUB6 = Path(next(iter(kubs.values()))[0]).parent / "kub_{sid}.yaml"
    try:
        pre, otp, start = podgotovka(kubs, out, prodolzhit, kod_slit, itog, zamki, rz, v)
        sb = {s: kubs[s][1]["sobol"] for s in start}                     # всего точек — как progon_sobol: prohod2 → p1
        nn = {s: (n[s] if isinstance(n, dict) else n) or int(sb[s].get("prohod2", sb[s].get("prohod1", sb[s].get("N1")))) for s in start}
        p1 = {s: int(sb[s].get("prohod1", nn[s])) for s in start}
        if rz.setka in ("v5", "v6"):                           # v5, v6: один проход [0, N1), источник строки — как у драйвера
            p1 = nn = n if isinstance(n, dict) else ps.n1_po_sid({s: kubs[s][1] for s in start}, str(n) if n else None, rz)
        ist = None if rz.setka == "progon" else "k3" if rz.k3 else f"sobol1_seed{2 if rz.seed == 'seed2' else 1}"
        ps.proverit_disk(sum(max(nn[s] - start[s], 0) for s in start), out, ps.BAIT_V21 if v in lm.SUMMY_V else ps.BAIT_NA_TOCHKU)
        ctx = dict(kubs={s: kubs[s][1] for s in start}, root=out, ist="sobol1", kod_slit=kod_slit, fast=fast, v=v, rz=rz,
                   zakryt={s: ps.zakrytye(out, s) for s in start})
        pre = pre if rezhim == "obshchie" else {}             # prezhniy: дети строят свои оценщики (путь progon_sobol)
        ps.fabrika = pre.__getitem__ if pre else ps.fabrika    # obshchie: дети берут объекты родителя (fork), не строят свои
        gc.collect()
        gc.freeze()                                            # сборщик не трогает заголовки объектов родителя → меньше COW
        schet(zadachi(start, nn, p1, rezhim, ist), ctx, procs, itog, pre, {s: (out / s, otp[s]) for s in start}, nn)
    finally:
        gc.unfreeze()
        ps.fabrika, ps.KUB, ps.KUB5, ps.KUB6 = staryj
        [f.close() for f in zamki]
    return itog


def kuby_iz_argv(a, ap) -> tuple:
    """({sid: (путь, куб)}, rz, n): `--kub v5` — режим и точки драйвера (ps.rezhim, ps.n1_po_sid), кубы designs/v5; иначе пути yaml."""
    if (a.rol is None) != (a.t_vyhod is None):
        ap.error("--rol spravka и --t-vyhod close — только вместе")
    if a.kub not in (["v5"], ["v6"]):
        if a.k3 or a.n1 or a.rol or a.zamorozka:
            ap.error("--k3/--n1/--rol/--zamorozka — только для --kub v5 | v6")
        return {c["sid"]: (p, c) for p, c in ((Path(p), zagruzit_kub(p, a.period, a.seed)) for p in a.kub)}, ps.R0, None
    sids, v = a.strategii.split(","), a.kub[0]
    if v == "v6":
        from fastbt import sobytiya_e1
        sobytiya_e1.trebovat_svod()
    if a.seed not in (None, 1, 2) or (a.zamorozka and v != "v6"):
        ap.error("--seed: 1 | 2; --zamorozka — только для --kub v6")
    rz, _ = ps.rezhim(argparse.Namespace(kub=v, period=a.period, seed=a.seed or 1, k3=a.k3, n1=a.n1, root=None, zamorozka=a.zamorozka), sids, ap)
    kubs = {s: (ps.kub_put(s, rz), zagruzit_kub(ps.kub_put(s, rz), a.period, rz=rz)) for s in sids}
    rz = rz._replace(rol=a.rol or rz.rol, t_vyhod=a.t_vyhod)         # v6: rol прохода (sobol | kontrol) или spravka
    return kubs, rz, ps.n1_po_sid({s: c for s, (_, c) in kubs.items()}, a.n1, rz)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Соболь на сервере: общие входы через fork, чекпоинт, каталог на стратегию")
    ap.add_argument("--kub", nargs="+", required=True, help="v5 (кубы designs/v5), v6 (designs/v6, ledger 2.2) или kub_<sid>.yaml (один каталог)")
    ap.add_argument("--zamorozka", type=Path, help="v6: каталог Морриса (<Sk>/zamorozka.yaml); всем проходам, кроме k3")
    ap.add_argument("--period", required=True, choices=["train4", "v2", "wf"])
    ap.add_argument("--seed", type=int, help="v5: 1 | 2 (sobol.seed | seed2); progon: sobol.seed вместо seed куба")
    ap.add_argument("--k3", action="store_true", help="v5: проход к3 — полная решётка раздела k3")
    ap.add_argument("--n1", help="v5: N1 вместо куба — число для всех или S1=65536,S2=65536")
    ap.add_argument("--strategii", default=",".join(ps.SIDS), help="v5: список стратегий")
    ap.add_argument("--rol", choices=["spravka"], help="справочный проход (Ж-137 п. 6): тот же куб, ledger 2.2 с rol = spravka, вне N")
    ap.add_argument("--t-vyhod", choices=["close"], help="с --rol spravka: выход по close дня (подменяет t_vyhod точек)")
    ap.add_argument("--ledger", choices=[lm.V, lm.V21], help="формат ledger; по умолчанию 2.0 для v2, иначе 2.1; 2.1 на v2 — отказ")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--procs", type=int, default=procs_po_umolchaniyu(), help="по умолчанию ядра − 4")
    ap.add_argument("--prodolzhit", action="store_true", help="с чекпоинта; отказ при чужом kod/vhody/кубе/seed")
    ap.add_argument("--kod-slit", action="store_true")
    ap.add_argument("--fast", action="store_true", help="быстрый слой (Ж-083) для покрытых точек")
    a = ap.parse_args(argv)
    from fastbt import zona
    with zona.zona_perioda(a.period):                               # wf: зона wf до пула (fork — дети наследуют); выход — прежняя зона
        kubs, rz, n = kuby_iz_argv(a, ap)
        if len(kubs) != len(a.kub) and a.kub not in (["v5"], ["v6"]):
            raise SystemExit("одна стратегия указана дважды")
        itog = progon(kubs, a.out, a.procs, n, a.prodolzhit, a.kod_slit, a.fast, rz=rz, ledger=a.ledger)
    for s, v in itog.items():
        print(f"{s}: посчитано {v['n']}, ошибок {v['oshibok']}, чекпоинт {v['posledniy_i']}, сбой {v['sboy']}")
    return 2 if any(v["sboy"] or v["oshibok"] for v in itog.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
