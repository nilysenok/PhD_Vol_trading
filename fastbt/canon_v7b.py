"""V7-1b: канон кубов v7 S8–S12 (Ж-153 пп. 1–4, Ж-156 п. 3; план §1.1). Данных не читает; S7 в v7 нет. Продолжение canon_v7 (S1–S6).

S10–S12 (импульс 1ч / 2ч / согласие): сигнал `tau ∈ {2,3,5}` (τ = 1 — справочно, вне N), `vhod ∈ {d1, T}`, `T` — дочерняя ось при vhod = T
(при d1 сворачивается); общие оси выхода и риска — те же значения, что в кубе S3 v6 (`baza.iz: S3`, список `osi_s3`); `k_okup ∈ {net, 3}`.
Не входят: `tau` v5, k_e и зависящие от лимитки, t_vyhod, time_decay, фильтры, контекст, пауза. Канон — правила тренда `canon_v6` (S3).
S8, S9 (моментум / сила к сектору): полная решётка porog × bufer × cap × rebal × k_okup (108 различных точек); правил несовместимости нет.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from fastbt import canon_v6 as c6, canon_v7 as c7

SIDS_B = ("S8", "S9", "S10", "S11", "S12")
IMPULS = ("S10", "S11", "S12")
NET, SIGNAL_IMP = c7.NET, ("tau", "vhod", "T")
GOTOVY_B = c7.GOTOVY_V7


def load_kub(sid: str, root: Path = c7.KUB7) -> dict:
    """Куб v7; у импульса ключ `s3` — куб S3 v6 как есть (источник значений общих осей)."""
    c7.sverit_sha(sid, root)
    k = yaml.safe_load((Path(root) / f"kub_{sid}.yaml").read_text(encoding="utf-8"))
    if sid in IMPULS:
        k["s3"] = c6.load_kub("S3")
    return k


def osi(kub: dict, gotovy=GOTOVY_B) -> list:
    """[(имя, значения, спецификация)]: сигнал; у импульса — общие оси S3 (по списку osi_s3, в порядке куба S3); k_okup."""
    out = [(n, list(v["setka"]), v) for n, v in kub["signal"].items()]
    if "s3" in kub:
        out += [(n, g, sp) for n, g, sp in c6.osi(kub["s3"], gotovy) if n in kub["osi_s3"]]
    return out + [(n, list(v["setka"]), v) for n, v in kub["osi7"].items() if c6.gotovo(v.get("zhdet"), gotovy, n)]


def obem(kub: dict) -> float:
    v = 1.0
    for _, g, _ in osi(kub, None):
        v *= len(g)
    return v


def proverit(kub: dict) -> None:
    sid, imp = kub["sid"], kub["sid"] in IMPULS
    if sid not in SIDS_B or kub.get("v") != 5 or kub.get("setka") != "v7" or kub.get("tip") != ("impuls" if imp else "sechenie"):
        raise ValueError(f"{sid}: нужен v: 5, setka: v7 и семья S8–S12")
    if imp and kub["baza"] != {"setka": "v6", "kub_sha": c7.sha_v6("S3"), "iz": "S3"}:
        raise ValueError(f"{sid}: baza.kub_sha не совпадает с sha256 kub_S3.yaml v6")
    if imp:
        c6.proverit(kub["s3"])
        names = {n for n, _, _ in c6.osi(kub["s3"], None)}
        if not set(kub["osi_s3"]) <= names or tuple(kub["signal"]) != SIGNAL_IMP:
            raise ValueError(f"{sid}: osi_s3 вне осей S3 или signal ≠ {SIGNAL_IMP}")
    k = kub["osi7"].get("k_okup")
    if not k or k["setka"] != [NET, 3] or k.get("vykl") != NET or k.get("p_vykl") != 0.5 or not k.get("zhdet"):
        raise ValueError(f"{sid}: k_okup — нужны {{net, 3}}, vykl net, p_vykl 0.5, zhdet")


def canon_v7b(sid: str, pt: dict, kub: dict):
    """Точка → канон (T при d1 и k_okup = net убраны; у импульса — правила тренда canon_v6 по общим осям S3) или None."""
    q = dict(pt)
    if q.get("vhod") == "d1":
        q.pop("T", None)
    if q.get("k_okup", NET) == NET:
        q.pop("k_okup", None)
    if sid not in IMPULS:
        return q
    own = SIGNAL_IMP + ("k_okup",)
    c = c6.canon_v6("S3", {n: v for n, v in q.items() if n not in own}, kub["s3"])
    return None if c is None else dict(c, **{n: q[n] for n in own if n in q})


def main() -> int:
    import numpy as np
    rng = np.random.default_rng(20261031)
    print("sid | осей | объём номинальный | годных из 20 000 равномерных u (не Соболь) | N1")
    for sid in SIDS_B:
        k = load_kub(sid)
        proverit(k)
        ox = osi(k)
        g = sum(canon_v7b(sid, c7.tochka(ox, rng.random(len(ox))), k) is not None for _ in range(20000))
        print(f"{sid} | {len(osi(k, None))} | {obem(k):.4e} | {g} | {k['sobol']['N']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
