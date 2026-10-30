"""Эквивалентность Мак ↔ сервер, набор v6 (Ж-137 п. 4; лист «эталоны v6»). Отдельно от ekviv.py, чтобы v5-часть npz не менялась побайтно.

Набор: якорь v6 каждой Sk на train4 (середина сетки осей v5 + `yakorya_progon` куба, новые оси «выкл.»: {**yakorya, **канон}, как в
`progon_v6.tochki`; одной «якорной» точки с сигнальными осями в кубе нет — оценщик v6 требует их все) и N_V6 = 200 точек
Соболя куба v6 (seed `seed` куба, как боевой драйвер: `ps.Rezhim("v6", "train4", "seed", …, rol = "sobol")`, оценщик EvaluatorV6 с рядами
`istochniki_v6.zagruzit`). Точки — первые канонические (не None) по порядку индексов Соболя, СО СТАТУСОМ ok; доля по Sk — 34/34/33/33/33/33.
Заморозка Морриса при сборке эталона пустая (Морриса ещё нет; эталон проверяет вычислитель, а не выбор осей) — отступление, см. отчёт.
Имена записей: `v6yakor__Sk`, `v6tochka__Sk__i` (эталонный путь, fast = False), `v6bystr__Sk__i` (быстрый слой, fast = True, только где
`fast_ok(pt)`). Формат записи — как у v5 (ekviv._zapis); хэши входов — в meta (`v6_vhody`, `v6_kub`, `v6_daty`), строго."""
import concurrent.futures as cf
import multiprocessing as mp

import ekviv

N_V6, CHUNK, PREDEL = 200, 50, 20000


def _ps():
    return ekviv._ps()


def kvoty(n: int = N_V6) -> dict:
    """{sid: сколько точек}: поровну, остаток — первым Sk по порядку (200 → 34, 34, 33, 33, 33, 33)."""
    sids = _ps().SIDS
    return {s: n // len(sids) + (k < n % len(sids)) for k, s in enumerate(sids)}


def rezhim(sid: str, period: str = None):
    ps = _ps()
    fdr = ps.load_kub(sid, ps.R0._replace(setka="v6")).get("fdr")
    return ps.Rezhim("v6", period or ps.PERIOD, "seed", False, fdr, "sobol", None, ((sid, "etalon-bez-zamorozki", ()),))


def _tochka(sr, sid, i, pt, rz, fast) -> tuple:
    """-> (status, запись) одной точки; запись снимается сразу: poz оценщика — от последнего вызова."""
    from fastbt import ledger_v2 as lm
    r, p, t = _ps().schitat_tochku(sr, sid, i, pt, "ekviv", False, fast, lm.V22, rz)
    return r.get("status"), ekviv._zapis(r, p, t, sr.evr)


def _nabrat(sr, c, sid, rz, kvota) -> tuple:
    """-> ([(i, pt, запись)] первые kvota точек со статусом ok, число пропущенных с другим статусом)."""
    ps, ok, plokho, n0 = _ps(), [], 0, 0
    while len(ok) < kvota:
        if n0 >= PREDEL:
            raise RuntimeError(f"{sid}: за {PREDEL} индексов Соболя набрано {len(ok)} из {kvota} точек ok")
        for i, pt in ps.tochki(c, n0, n0 + CHUNK, rz):
            if len(ok) >= kvota:
                break
            st, z = _tochka(sr, sid, i, pt, rz, False)
            if st == "ok":
                ok.append((i, pt, z))
            else:
                plokho += 1
        n0 += CHUNK
    return ok, plokho


def _po_indeksam(sr, c, sid, rz, idx) -> list:
    ps, out = _ps(), []
    for i in idx:
        got = ps.tochki(c, i, i + 1, rz)
        if not got or got[0][0] != i:
            raise RuntimeError(f"{sid}: точки {i} куба v6 нет (канон дал None)")
        out.append((i, got[0][1], _tochka(sr, sid, i, got[0][1], rz, False)[1]))
    return out


def tochka_yakorya(c: dict, sdvig: int = 0) -> dict:
    """Якорь v6: середина сетки каждой оси v5 (ячейка len // 2 + sdvig), новые оси «выкл.», оси zh047 — `yakorya_progon` куба
    (как `progon_v6.tochki`: {**yakorya, **канон}). Несовместимая по канону — None."""
    from fastbt import canon_v5 as c5, canon_v6
    pt = {n: g[(len(g) // 2 + sdvig) % len(g)] for n, g in c5.osi(c["v5"])}
    q = canon_v6.canon_v6(c["sid"], pt, c)
    return None if q is None else {**c["v5"].get("yakorya_progon", {}), **q}


def _yakor(sr, c, sid, period: str) -> dict:
    """Первый сдвиг 0, 1, … , где точка совместима и оценщик дал ряд со статусом ok; иначе отказ (молчаливой подмены нет)."""
    for sd in range(8):
        pt = tochka_yakorya(c, sd)
        if pt is None:
            continue
        r, p, t = sr.evr.evaluate_v2(pt, period)
        if p is not None and r.get("status", "ok") == "ok":
            return ekviv._zapis(r, p, t, sr.evr)
    raise RuntimeError(f"якорь v6 {sid}: ни один из 8 сдвигов середины сетки не дал ряда ok")


def _zadacha(z: tuple) -> tuple:
    """(sid, idx | None, bystr, kvota, period) -> (sid, {имя: запись}, kod, vhody, kub_sha, daty_sha, пропущено). idx None — набор по квоте."""
    sid, idx, bystr, kvota, period = z
    ps = _ps()
    rz = rezhim(sid, period)
    sr, c, out = ps.sreda(sid, rz), ps.load_kub(sid, rz), {}
    out[f"v6yakor__{sid}"] = _yakor(sr, c, sid, period)
    tochki, plokho = _nabrat(sr, c, sid, rz, kvota) if idx is None else (_po_indeksam(sr, c, sid, rz, idx), 0)
    for i, pt, zap in tochki:
        out[f"v6tochka__{sid}__{i}"] = zap
        if (idx is None or i in bystr) and ps.fast_ok(pt):
            out[f"v6bystr__{sid}__{i}"] = _tochka(sr, sid, i, pt, rz, True)[1]
    mf = ps.manifest(sid, sr, c, False, rz)
    return sid, out, sr.kod, sr.vhody, mf["kub_sha"], mf["daty_sha"], plokho


def poschitat(tochki, bystr: set, procs: int, n: int = N_V6, period: str = None) -> tuple:
    """tochki None — набрать по квотам (сборка эталона); иначе [(sid, i)] из эталона (проверка). -> (rez, meta v6_*, kod).
    period: train4 (по умолчанию) или wf — тот же набор (200 точек по квотам), куб v6 объявляет оба периода."""
    ps, kv, period = _ps(), kvoty(n), period or _ps().PERIOD
    zad = [(s, None if tochki is None else [i for q, i in tochki if q == s], {i for q, i in bystr if q == s}, kv[s], period)
           for s in ps.SIDS]
    if procs <= 1:
        it = map(_zadacha, zad)
    else:
        ex = cf.ProcessPoolExecutor(min(procs, len(zad)), mp_context=mp.get_context("fork"))
        it = ex.map(_zadacha, zad)
    rez, kod, m = {}, set(), dict(v6_vhody={}, v6_kub={}, v6_daty={}, v6_propusheno={})
    for sid, out, k, vh, kub, daty, plokho in it:
        rez.update(out)
        kod.add(k)
        m["v6_vhody"][sid], m["v6_kub"][sid], m["v6_daty"][sid], m["v6_propusheno"][sid] = vh, kub, daty, plokho
    if procs > 1:
        ex.shutdown()
    if len(kod) != 1:
        raise RuntimeError(f"v6: разный хэш кода в процессах: {sorted(kod)}")
    return rez, m, kod.pop()


def iz_etalona(rez: dict) -> tuple:
    """([(sid, i)] точек эталонного пути, {(sid, i)} быстрого слоя) по именам записей эталона; пусто — эталон без v6."""
    def razbor(pref):
        return sorted((m.split("__")[1], int(m.split("__")[2])) for m in rez if m.startswith(pref))
    return razbor("v6tochka__"), set(razbor("v6bystr__"))


def est_v6(rez: dict) -> bool:
    return any(m.startswith("v6yakor__") for m in rez)


def dobavit(rez: dict, meta: dict, rez6: dict, m6: dict, kod6: str) -> None:
    """Слить v6 в общий набор; код v5 и v6 обязан совпасть (один процесс-код)."""
    if kod6 != meta["kod"]:
        raise RuntimeError(f"хэш кода v6 {kod6} ≠ v5 {meta['kod']}")
    rez.update(rez6)
    meta.update(m6)
