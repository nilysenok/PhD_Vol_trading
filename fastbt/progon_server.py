"""Ж-131 п. 3–4: серверный запуск Соболя — обёртка над fastbt.progon_sobol (драйвер и ledger_v2 не правятся: Л7b).
obshchie (Linux, по умолчанию): оценщик со всеми входами (бары, σ, сигналы) строится ОДИН раз в родителе, gc.freeze, fork —
дети читают те же объекты и буферы numpy без копирования. prezhniy (macOS): путь progon_sobol.schet — fork без предзагрузки.
Стратегия — каталог <out>/<Sk>/ (ledger 2.1, kod и vhody в каждой строке), flock, checkpoint.json каждые CHEKPOINT точек;
сбой стратегии отменяет только её задачи. --prodolzhit: отказ при чужом kod/vhody/кубе/seed/периоде. Запуск:
python3 -m fastbt.progon_server --kub <yaml> --period train4 --seed <n> --out <каталог> [--procs <n>] [--prodolzhit]."""
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

from fastbt import ledger_v2 as lm
from fastbt import progon_sobol as ps

CHEKPOINT, CHP, ZAMOK = 5000, "checkpoint.json", ".zamok"
POLYA = ("kod", "vhody", "kub_sha", "sobol_seed", "period")     # отпечаток: сверяется с чекпоинтом и manifest.json
REZHIM = "obshchie" if sys.platform.startswith("linux") else "prezhniy"


def procs_po_umolchaniyu() -> int:
    return max(1, (os.cpu_count() or 5) - 4)          # Ж-131 п. 4: ядра − 4


def zagruzit_kub(path, period: str, seed=None) -> dict:
    """Куб из kub_<sid>.yaml (progon_sobol.manifest берёт kub_sha по этому имени); seed — переопределение sobol.seed."""
    path = Path(path)
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.name != f"kub_{c['sid']}.yaml":
        raise SystemExit(f"{path}: файл куба обязан называться kub_{c['sid']}.yaml")
    if period != ps.PERIOD or c.get("period") != period:
        raise SystemExit(f"период {period!r} (в кубе {c.get('period')!r}): progon_sobol и ledger 2.1 (кварталы 2016–2019) "
                         f"считают только {ps.PERIOD}; для v2 нужна точка входа в драйвере (Л7b) — отказ")
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
        raz = [k for k in POLYA if k in ex and ex[k] != otp[k]]
        if raz:
            raise SystemExit(f"{p}: --prodolzhit — отказ, не совпали {raz}")
    return int(json.loads(chp.read_text(encoding="utf-8"))["posledniy_i"]) + 1 if chp.exists() else 0


def pisat_chekpoint(d: Path, otp: dict, posl: int, n: int) -> None:
    tmp, t = d / f".{CHP}.{os.getpid()}.{uuid.uuid4().hex}.tmp", time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp.write_text(json.dumps({**otp, "posledniy_i": posl, "N": n, "shag": CHEKPOINT, "vremya": t}, sort_keys=True), "utf-8")
    os.replace(tmp, d / CHP)


def zadachi(start: dict, n: dict, p1: dict, rezhim: str) -> list:
    """(sid, a, b, istochnik): куски ≤ KUSOK, не через границу чекпоинта и конца прохода 1 (sobol1 / sobol2)."""
    out = []
    for s, a in start.items():
        while a < n[s]:
            b = min(a + ps.KUSOK, n[s], (a // CHEKPOINT + 1) * CHEKPOINT, *([p1[s]] if a < p1[s] else []))
            out.append((s, a, b, "sobol1" if a < p1[s] else "sobol2"))
            a = b
    return sorted(out, key=lambda z: (z[1] // CHEKPOINT, z[0], z[1])) if rezhim == "obshchie" else out


def _kusok(zad: tuple) -> tuple:
    """В ребёнке: кусок progon_sobol.schitat_kusok + (pid, id оценщика, адрес буфера дат) для проверки общих массивов."""
    ps._W["ist"] = zad[3]
    res, sr = ps.schitat_kusok(zad[:3]), ps._W["sreda"]
    return res + (os.getpid(), id(sr.evr), sr.daty.__array_interface__["data"][0])


def podgotovka(kubs: dict, out: Path, prodolzhit: bool, kod_slit: bool, itog: dict, zamki: list) -> tuple:
    """Родитель: оценщик, отпечаток, замок, старт, manifest. Сбой оценщика одной стратегии — в itog, остальные идут;
    отказ сверки (SystemExit) останавливает весь запуск до счёта."""
    pre, otp, start = {}, {}, {}
    for s, (path, c) in kubs.items():
        try:
            ps._W.clear()
            sr = ps.sreda(s)                                   # проверка дат < 2020 — та же, что у драйвера
            otp[s] = {"kod": sr.kod, "vhody": sr.vhody, "kub_sha": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      "sobol_seed": c["sobol"]["seed"], "period": ps.PERIOD}
            zamki.append(zamok(out / s))
            start[s] = nachalo(out / s, otp[s], prodolzhit)
            # manifest.json — до fork: Pisatel пишет его write_text без tmp, два ребёнка на первых кусках одной стратегии
            # читали полупустой файл (JSONDecodeError; test_manifest_iz_roditelya_do_fork)
            lm.Pisatel21(out, s, 0, ps.manifest(s, sr, c, kod_slit), daty=sr.daty, poz=hasattr(sr.evr, "poz"))   # как schitat_kusok
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


def progon(kubs: dict, out, procs: int, n=None, prodolzhit=False, kod_slit=False, fast=False, rezhim=REZHIM) -> dict:
    """kubs {sid: (путь, куб)} → {sid: n, oshibok, sboy, posledniy_i, deti}; n — точек на стратегию (по умолчанию из куба)."""
    out, sids = Path(out), list(kubs)
    if procs < 1 or rezhim not in ("obshchie", "prezhniy") or len({Path(p).parent for p, _ in kubs.values()}) > 1:
        raise SystemExit("procs ≥ 1; rezhim obshchie | prezhniy; кубы — из одного каталога")
    itog = {s: dict(n=0, oshibok=0, sboy=None, posledniy_i=None, deti=set()) for s in sids}
    staryj, zamki, ps.KUB = (ps.fabrika, ps.KUB), [], Path(next(iter(kubs.values()))[0]).parent / "kub_{sid}.yaml"
    try:
        pre, otp, start = podgotovka(kubs, out, prodolzhit, kod_slit, itog, zamki)
        sb = {s: kubs[s][1]["sobol"] for s in start}                     # всего точек — как progon_sobol: prohod2 → p1
        nn = {s: n or int(sb[s].get("prohod2", sb[s].get("prohod1", sb[s].get("N1")))) for s in start}
        p1 = {s: int(kubs[s][1]["sobol"].get("prohod1", nn[s])) for s in start}
        ps.proverit_disk(sum(max(nn[s] - start[s], 0) for s in start), out, ps.BAIT_V21)
        ctx = dict(kubs={s: kubs[s][1] for s in start}, root=out, ist="sobol1", kod_slit=kod_slit, fast=fast, v=lm.V21,
                   zakryt={s: ps.zakrytye(out, s) for s in start})
        pre = pre if rezhim == "obshchie" else {}             # prezhniy: дети строят свои оценщики (путь progon_sobol)
        ps.fabrika = pre.__getitem__ if pre else ps.fabrika    # obshchie: дети берут объекты родителя (fork), не строят свои
        gc.collect()
        gc.freeze()                                            # сборщик не трогает заголовки объектов родителя → меньше COW
        schet(zadachi(start, nn, p1, rezhim), ctx, procs, itog, pre, {s: (out / s, otp[s]) for s in start}, nn)
    finally:
        gc.unfreeze()
        ps.fabrika, ps.KUB = staryj
        [f.close() for f in zamki]
    return itog


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Соболь на сервере: общие входы через fork, чекпоинт, каталог на стратегию")
    ap.add_argument("--kub", nargs="+", required=True, type=Path, help="kub_<sid>.yaml (один или несколько, один каталог)")
    ap.add_argument("--period", required=True, choices=["train4", "v2"])
    ap.add_argument("--seed", type=int, help="sobol.seed вместо seed куба")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--procs", type=int, default=procs_po_umolchaniyu(), help="по умолчанию ядра − 4")
    ap.add_argument("--prodolzhit", action="store_true", help="с чекпоинта; отказ при чужом kod/vhody/кубе/seed")
    ap.add_argument("--kod-slit", action="store_true")
    ap.add_argument("--fast", action="store_true", help="быстрый слой (Ж-083) для покрытых точек")
    a = ap.parse_args(argv)
    kubs = {c["sid"]: (p, c) for p, c in ((p, zagruzit_kub(p, a.period, a.seed)) for p in a.kub)}
    if len(kubs) != len(a.kub):
        raise SystemExit("одна стратегия указана дважды")
    itog = progon(kubs, a.out, a.procs, None, a.prodolzhit, a.kod_slit, a.fast)
    for s, v in itog.items():
        print(f"{s}: посчитано {v['n']}, ошибок {v['oshibok']}, чекпоинт {v['posledniy_i']}, сбой {v['sboy']}")
    return 2 if any(v["sboy"] or v["oshibok"] for v in itog.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
