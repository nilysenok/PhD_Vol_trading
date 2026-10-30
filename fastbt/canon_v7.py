"""V7-1a: канон точки куба v7 S1–S6 (Ж-153, Ж-155 п. 2, Ж-156 п. 3, Ж-157 п. 2; план V7-1a). Данных не читает.

Куб v7 — fastbt/designs/v7/kub_S{k}.yaml (v: 5, setka: v7) = куб v6 (`baza.kub_sha` — sha256 kub_S{k}.yaml v6) со сжатыми сигнальными осями
(раздел `signal`: центр ± 1 шаг; центр — setka[len // 2] оси v6, соседи — по порядку значений; у оси из двух значений остаются оба)
+ новые оси раздела `osi7` (`k_okup`, `otnositelno` у S1–S4; «выкл.» = `net`, первое значение). Общие оси v6 — как есть.
Минимальное удержание 2 дня у контртренда (`min_uderzhanie`) — правило исполнения, не ось, в N не входит.
Точка v7 с выкл. осями v7 имеет канон (и klyuch) точки v6 (v6 ⊂ v7); k_okup = net убирает зависимые (`roditel: [k_okup]`) и сам.
"""
from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import yaml
from scipy.stats import qmc

from fastbt import canon_v5 as c5, canon_v6 as c6

ROOT = c5.ROOT
KUB7 = ROOT / "fastbt/designs/v7"
SIDS = c5.SIDS
NET = c5.NET
NOVYE7 = ("k_okup", "otnositelno")
OSI7 = "osi7"
SUMS = "SHA256SUMS"
GOTOVY_V7 = c6.GOTOVY | {"V7-1a", "V7-3"}      # для счёта годных: оси v7 открыты (оценщик — V7-3, E1)


def suzhenie(g: list) -> list:
    """Центр ± 1 шаг: центр g[len // 2], соседи по порядку; на краю — сколько есть (две точки → обе)."""
    i = len(g) // 2
    return list(g[max(0, i - 1): i + 2])


def sha_v6(sid: str) -> str:
    return hashlib.sha256((c6.KUB6 / f"kub_{sid}.yaml").read_bytes()).hexdigest()


def load_kub(sid: str, root: Path = KUB7) -> dict:
    """Куб v7 с подгруженной основой: ключ `v6` — куб v6 как есть (его `baza.kub_sha` сверяется в proverit)."""
    sverit_sha(sid, root)
    k = yaml.safe_load((Path(root) / f"kub_{sid}.yaml").read_text(encoding="utf-8"))
    k["v6"] = c6.load_kub(sid)
    return k


def sverit_sha(sid: str, root: Path = KUB7) -> None:
    """Д5: sha256 kub_{sid}.yaml = строка SHA256SUMS; иначе (или строки нет) — ValueError с обоими хэшами."""
    name, tab = f"kub_{sid}.yaml", {}
    for ln in (Path(root) / SUMS).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, n = ln.split(None, 1)
            tab[n.strip().lstrip("*")] = h
    fakt = hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
    if tab.get(name) != fakt:
        raise ValueError(f"{name}: sha256 файла {fakt} ≠ {SUMS} {tab.get(name, 'строки нет')} (куб изменён без обновления хэша)")


def novye(kub: dict) -> dict:
    return dict(kub.get(OSI7, {}))


def osi(kub: dict, gotovy=c6.GOTOVY) -> list:
    """[(имя, значения, спецификация)]: оси v6, сигнальные — сжатые; затем новые оси v7 (gotovy=None — все, номинал)."""
    out = [(n, list(kub["signal"][n]["setka"]) if n in kub["signal"] else g, sp) for n, g, sp in c6.osi(kub["v6"], gotovy)]
    out += [(n, list(v["setka"]), v) for n, v in novye(kub).items() if c6.gotovo(v.get("zhdet"), gotovy, n)]
    return out


def proverit(kub: dict) -> None:
    """Формат v: 5; основа v6 проходит proverit v6 и совпадает по sha; сигнал = центр ± 1 от сетки v6; новые оси: net первым, p_vykl."""
    sid, t = kub["sid"], c5.tip(kub["sid"])
    if kub.get("v") != 5 or kub.get("setka") != "v7" or kub.get("tip") != t:
        raise ValueError(f"{sid}: нужен v: 5, setka: v7, tip: {t}")
    if kub["baza"] != {"setka": "v6", "kub_sha": sha_v6(sid)}:
        raise ValueError(f"{sid}: baza.kub_sha не совпадает с sha256 kub_{sid}.yaml v6 (куб v6 изменён после коммита v7)")
    c6.proverit(kub["v6"])
    base = dict((n, g) for n, g, _ in c6.osi(kub["v6"], None))
    if tuple(kub["signal"]) != c5.SIGNAL[sid]:
        raise ValueError(f"{sid}: оси signal {list(kub['signal'])} ≠ {c5.SIGNAL[sid]}")
    for n, v in kub["signal"].items():
        if v["setka"] != suzhenie(base[n]) or v["centr"] != base[n][len(base[n]) // 2]:
            raise ValueError(f"{sid}: {n} — не центр ± 1 шаг от сетки v6 {base[n]}")
    for n, v in novye(kub).items():
        g = v["setka"]
        if n in base or len(g) < 2 or v.get("vykl") != g[0] or g[0] != NET or not v.get("zhdet") or v.get("p_vykl") != 0.5:
            raise ValueError(f"{sid}: {n} — нужны net первым, zhdet, p_vykl 0.5 и имя вне осей v6")
    if (t == "kontr") != (kub.get("min_uderzhanie") == {"dney": 2, "tau1": "spravka"}):
        raise ValueError(f"{sid}: min_uderzhanie: dney 2 — у контртренда, у тренда нет")


def dvojnoj_fon(q: dict, kub: dict) -> bool:
    """otnositelno ≠ net вместе с rel_signal ≠ «выкл.» v6 (syroy): фон вычитается дважды → точка несовместима."""
    rs = kub["v6"].get("kontekst", {}).get("rel_signal")
    return bool(rs) and q.get("otnositelno", NET) != NET and q.get("rel_signal", rs["vykl"]) != rs["vykl"]


def canon_v7(sid: str, pt: dict, kub: dict):
    """Точка → канон v7 (оси v7 в «выкл.» убраны, зависимые от выключенного k_okup — тоже; остальное — canon_v6) или None.
    None и при двойном фоне (`dvojnoj_fon`)."""
    nov, q = novye(kub), dict(pt)
    if dvojnoj_fon(q, kub):
        return None
    for n, v in nov.items():
        if v.get("roditel") and all(q.get(r, NET) == NET for r in v["roditel"]):
            q[n] = NET
    c = c6.canon_v6(sid, {k: v for k, v in q.items() if k not in nov}, kub["v6"])
    if c is None:
        return None
    c.update({n: q[n] for n in nov if q.get(n, NET) != NET})
    return c


def obem(kub: dict) -> float:
    v = 1.0
    for _, g, _ in osi(kub, None):
        v *= len(g)
    return v


def tochka(ox: list, row) -> dict:
    """Строка u ∈ [0,1)^d → точка: u < p_vykl → «выкл.», остаток делится поровну; прочие оси равномерно (как tochki_v6)."""
    return {n: g[c6._indeks(x, len(g), sp.get("p_vykl"))] for (n, g, sp), x in zip(ox, row)}


def tochki_v7(kub: dict, n0: int, n1: int, seed: str = "seed", gotovy=c6.GOTOVY) -> tuple:
    """([(i, канон)], {vybrosheno}): i-я точка Соболя куба v7 (scramble, kub.sobol[seed]) → канон v7."""
    proverit(kub)
    ox, s = osi(kub, gotovy), kub["sobol"]
    sob = qmc.Sobol(len(ox), scramble=s["scramble"], seed=s[seed])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n0:
            sob.fast_forward(n0)
        u = sob.random(n1 - n0)
    out, vyb = [], 0
    for j, row in enumerate(u):
        c = canon_v7(kub["sid"], tochka(ox, row), kub)
        if c is None:
            vyb += 1
        else:
            out.append((n0 + j, c))
    return out, dict(vybrosheno=vyb)


def sha_nesovpadenia(root: Path = KUB7) -> list:
    """Файлы kub_S*.yaml, чей sha256 не совпадает со строкой SHA256SUMS (или строки нет); пусто — сходится."""
    tab = {}
    for ln in (Path(root) / SUMS).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, name = ln.split(None, 1)
            tab[name.strip().lstrip("*")] = h
    return [f.name for f in sorted(Path(root).glob("kub_S*.yaml")) if tab.get(f.name) != hashlib.sha256(f.read_bytes()).hexdigest()]


def main() -> int:
    import numpy as np
    rng, ok = np.random.default_rng(20261031), 20000
    print("sid | осей | объём номинальный | годных из 20 000 равномерных u (p_vykl), не Соболь | N1")
    for sid in [s for s in SIDS if (KUB7 / f"kub_{s}.yaml").exists()]:
        k = load_kub(sid)
        proverit(k)
        ox = osi(k, GOTOVY_V7)
        g =sum(canon_v7(sid, tochka(ox, rng.random(len(ox))), k) is not None for _ in range(ok))
        print(f"{sid} | {len(osi(k, None))} | {obem(k):.4e} | {g} | {k['sobol']['N']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
