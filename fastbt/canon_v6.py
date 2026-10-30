"""V6-13: канон точки куба v6 (Ж-132 п. 2–3; docs/otchety/2026-10-29_setka_v6_inventarizaciya.md §1, §3.2, §3.5, §4). Данных не читает.

Куб v6 — fastbt/designs/v6/kub_S{k}.yaml (v: 4, формат — README каталога) = куб v5 (`baza.kub_sha`; в разделах signal/filtr/vhod/vyhod
лежат только оси v5 с `v6_dobavleno` — добавленные значения) + новые разделы `kontekst` и `ispolnenie`. У каждой новой оси «выкл.» —
первое значение (`vykl`), поэтому точка v6 со всеми новыми осями «выкл.» есть точка v5 (v5 ⊂ v6). Ось с `roditel` — дочерняя:
при выключенных родителях схлопывается в «выкл.» (в N один раз).
Оси и значения, чьего кода ещё нет, несут `zhdet: [листы V6-…]` и в Соболь не входят, пока все листы не названы в `gotovy`
(GOTOVY — см. ниже: открыты только оси, подключённые в eval_v6). canon_v6 выбрасывает из точки новые оси со значением «выкл.»:
канон точки v6 без новых осей совпадает с canon_v5 (ключ `klyuch_n` ledger, §4).
"""
from __future__ import annotations

import functools
import hashlib
import sys
import warnings
from pathlib import Path

import yaml
from scipy.stats import qmc

from fastbt import canon_v5 as c5, canon_v6_nesovm as nsv

ROOT = c5.ROOT
KUB6 = ROOT / "fastbt/designs/v6"
SIDS, NET, NOVYE = c5.SIDS, c5.NET, ("kontekst", "ispolnenie")
# GOTOVY: листы V6-…, влитые в main, плюс «V6-12:<ось>» — ось, подключённая в eval_v6 (лист V6-12 общий для 3 PR, один токен «V6-12»
# открыл бы и неподключённые). Список осей V6-12 = eval_v6.PODKLYUCHENO ∩ новые оси куба (сверяет tests/test_eval_v6b.py).
V12 = ("rel_signal", "dlit", "asim", "norm", "hurst_th", "naklon", "vol_regime_max", "tihiy_othod", "sektor_trend", "shirina", "ma_n",
       "porog_shiriny", "relf_napr", "relf_k", "relf_okno", "relf_bench", "post_shok", "szhatie", "rv_perevorot", "obyom_anom", "rezhim",
       "sob_tip", "sob_do", "sob_posle", "vn_uslovie", "vn_faktor", "time_decay", "sigma_ist", "ed_lim", "zhizn", "pauza_porog", "pauza_okno",
       "lestnica", "gap", "t_vyhod", "vyhod_lim", "f_neisp", "piv_buf", "bezubytok", "stop_v_plyus", "pir", "reentry")
V12_RAS = ("stop_nach", "tejk")            # оси v5 с v6_dobavleno (piv:, vwap:, pol_puti), подключённые в V6-12d
# V6-10 (пивот / VWAP в движке) и V6-11 (тренд на стоп-заявке) — листы влиты целиком; не подключены: T, tf, smena (ось тренда действует только
# при vhod_tip = podtverzhdenie, а vhod_tip в куб не открыт — без него smena мертва)
GOTOVY: frozenset = frozenset(("V6-1a", "V6-1b", "V6-2", "V6-3", "V6-4", "V6-5", "V6-6", "V6-7", "V6-8", "V6-9", "V6-10", "V6-11")) \
    | {f"V6-12:{o}" for o in V12 + V12_RAS}
P_FILTR = 0.9                              # вероятность «выкл.» у запрещающих фильтров (Ж-132, решение 13:20; §3.5)
KE_NUZHEN = ("ed_lim", "lestnica", "vyhod_lim")   # действуют, только если вход лимиткой (k_e > 0), §1.2 И2/И4/И7
_K = c5.DOPUSK
DOPUSK = {                                 # допустимые единицы по типу: v5 + то, что добавляет v6 (§1.2 И11–И13, И16)
    "kontr": dict(_K["kontr"], stop_nach=_K["kontr"]["stop_nach"] | {"piv", "vwap"},
                  tejk=_K["kontr"]["tejk"] | {"pol_puti", "piv"}),
    "trend": dict(stop_nach={"pct", "atr", "sigma"}, tejk={NET, "rr", "pct", "atr", "sigh", "sila"},
                  trail={NET, "atr", "sigma", "step", "parab"}, bezubytok={NET, "atr", "rang"}),
}


def gotovo(zhdet, gotovy, os: str | None = None) -> bool:
    """Все листы zhdet названы в gotovy; лист "V6-12" оси `os` — также токеном "V6-12:<os>" (подключение по осям)."""
    return gotovy is None or all(z in gotovy or (os is not None and f"{z}:{os}" in gotovy) for z in zhdet or ())


def sha_v5(sid: str) -> str:
    return hashlib.sha256((c5.KUB / f"kub_{sid}.yaml").read_bytes()).hexdigest()


def load_kub(sid: str, root: Path = KUB6) -> dict:
    """Куб v6 с подгруженной основой: ключ `v5` — куб v5 как есть (`baza.kub_sha` сверяется в proverit)."""
    k = yaml.safe_load((Path(root) / f"kub_{sid}.yaml").read_text(encoding="utf-8"))
    k["v5"] = c5.load_kub(sid)
    return k


@functools.cache
def _kub(sid: str) -> dict:
    return load_kub(sid)


def rasshireno(kub: dict) -> dict:
    """{ось v5: спецификация} — оси v5, к значениям которых v6 добавляет свои (поле `v6_dobavleno`, README)."""
    return {n: v for sec in c5.SEKCII for n, v in kub.get(sec, {}).items() if "v6_dobavleno" in v}


def osi(kub: dict, gotovy=GOTOVY) -> list:
    """[(имя, значения, спецификация)] в порядке размерностей Соболя; gotovy=None — все оси (номинал)."""
    out, ras = [], rasshireno(kub)
    for n, g in c5.osi(kub["v5"]):
        r = ras.get(n)
        if r and gotovo(r.get("zhdet"), gotovy, n):
            g = g + list(r["v6_dobavleno"])
        out.append((n, g, {}))
    for sec in NOVYE:
        out += [(n, list(v["setka"]), v) for n, v in kub.get(sec, {}).items() if gotovo(v.get("zhdet"), gotovy, n)]
    return out


def novye(kub: dict) -> dict:
    """{имя: спецификация} всех новых осей (в том числе ещё не готовых)."""
    return {n: v for sec in NOVYE for n, v in kub.get(sec, {}).items()}


def proverit(kub: dict) -> None:
    """Формат v: 4; основа проходит proverit v5; «выкл.» первым; фильтры p = 0,9; дети без p; расширения допустимы по типу."""
    sid, t = kub["sid"], c5.tip(kub["sid"])
    if kub.get("v") != 4 or kub.get("setka") != "v6" or kub.get("tip") != t:
        raise ValueError(f"{sid}: нужен v: 4, setka: v6, tip: {t}")
    if kub["baza"] != {"setka": "v5", "kub_sha": sha_v5(sid)}:
        raise ValueError(f"{sid}: baza.kub_sha не совпадает с sha256 kub_{sid}.yaml v5 (куб v5 изменён после коммита v6)")
    c5.proverit(kub["v5"])
    nov, base = novye(kub), dict(c5.osi(kub["v5"]))
    if set(nov) & set(base) or sum(len(kub.get(sec, {})) for sec in NOVYE) != len(nov):
        raise ValueError(f"{sid}: новая ось повторяет ось v5 или другую новую")
    for n, v in nov.items():
        g = v["setka"]
        if len(g) < 2 or len(set(map(str, g))) != len(g) or not v.get("zhdet") or v.get("vykl") != g[0]:
            raise ValueError(f"{sid}: {n} — нужны ≥ 2 разных значения, vykl = первое значение и zhdet")
        if not v.get("smysl") or not v.get("otkuda"):
            raise ValueError(f"{sid}: {n} — нет smysl/otkuda (правило 1 README)")
        if bool(v.get("roditel")) == (v.get("p_vykl") is not None) or any(r not in nov for r in v.get("roditel", [])):
            raise ValueError(f"{sid}: {n} — либо roditel (дочерняя, без p_vykl), либо p_vykl")
        if bool(v.get("filtr")) != (v.get("p_vykl") == P_FILTR):
            raise ValueError(f"{sid}: {n} — filtr: true ⇔ p_vykl = {P_FILTR}")
    for n, r in rasshireno(kub).items():
        d = r["v6_dobavleno"]
        if n not in base or set(map(str, d)) & set(map(str, base[n])) or not r.get("zhdet"):
            raise ValueError(f"{sid}: {n}.v6_dobavleno — значение v5 уже есть или нет zhdet")
        for x in d:
            if c5.edinica(x) not in DOPUSK[t].get(n, {c5.edinica(x)}):
                raise ValueError(f"{sid}: {n}: {x} недопустимо для {t}")
    for sec, d in kub.get("snyato", {}).items():
        if any(k not in kub["v5"][sec] or a not in nov for k, a in d.items()):
            raise ValueError(f"{sid}: snyato.{sec} — якорь не в основе или ось не новая")


def canon_v6(sid: str, pt: dict, kub: dict | None = None):
    """Точка → канон v6 (dict без новых осей в «выкл.») или None (несовместима по canon_v5)."""
    nov = novye(kub or _kub(sid))
    q = dict(pt)
    off = lambda n: nov[n]["setka"][0]                                    # noqa: E731
    for n, v in nov.items():                                              # родители в yaml идут раньше детей
        if v.get("roditel") and all(q.get(r, off(r)) == off(r) for r in v["roditel"]):
            q[n] = off(n)
    ke = q.get("k_e", 0)
    for n in KE_NUZHEN:
        if n in nov and not ke > 0:
            q[n] = off(n)
    if "dlit" in nov and q.get("vhod_tip", "stop_zayavka") != "podtverzhdenie" and c5.tip(sid) == "trend":
        q["dlit"] = off("dlit")
    if "piv_buf" in nov and c5.edinica(q.get("stop_nach")) != "piv":       # буфер — только у стопа piv:next | piv:half (Н4 #677)
        q["piv_buf"] = off("piv_buf")
    if q.get("time_decay", NET) == "da" and (q.get("tau", NET) == NET or q.get("tejk", NET) != NET):
        q["time_decay"] = off("time_decay")                               # без tau или при тейке ≠ net ничего не делает (#679) → «выкл.», точка остаётся (V6-12d)
    if "reentry" in nov and q.get("vhod_tip", "stop_zayavka") != "stop_zayavka":
        q["reentry"] = off("reentry")                                     # повторный вход тренда — только на стоп-заявке (V6-11c)
    if sid == "S4" and q.get("norm") == "s252" and "period" in q:        # V6-5 (#648): при s252 period не влияет
        q["period"] = (kub or _kub(sid))["v5"]["signal"]["period"]["setka"][0]
    if nsv.nesovm(sid, nsv.svernut(q)):                                            # 7 сочетаний, где оценщик отказывает (canon_v6_nesovm)
        return None
    c = c5.canon_v5(sid, {k: v for k, v in q.items() if k not in nov})
    if c is None:
        return None
    c.update({n: q[n] for n in nov if n in q and q[n] != off(n)})
    return c


def obem(kub: dict, zamorozheno=()) -> float:
    """Номинальный объём (все оси, готовые и нет); замороженная новая ось даёт множитель 1, расширенная ось v5 — только значения v5."""
    zam, v, base = set(zamorozheno), 1.0, dict(c5.osi(kub["v5"]))
    for n, g, _ in osi(kub, None):
        v *= (len(base[n]) if n in base else 1) if n in zam else len(g)
    return v


def _indeks(u: float, L: int, p) -> int:
    return min(int(u * L), L - 1) if p is None else (0 if u < p else 1 + min(int((u - p) / (1 - p) * (L - 1)), L - 2))


def tochki_v6(kub: dict, n0: int, n1: int, seed: str = "seed", gotovy=GOTOVY) -> tuple:
    """([(i, канон)], {vybrosheno}): u<p → «выкл.», остаток делится поровну (§3.5); оси v5 и дочерние — равномерно."""
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
        pt = {n: g[_indeks(x, len(g), sp.get("p_vykl"))] for (n, g, sp), x in zip(ox, row)}
        c = canon_v6(kub["sid"], pt, kub)
        if c is None:
            vyb += 1
        else:
            out.append((n0 + j, c))
    return out, dict(vybrosheno=vyb)


def main() -> int:
    print("sid | осей | объём номинальный | без оси tf")
    for sid in [s for s in SIDS if (KUB6 / f"kub_{s}.yaml").exists()]:
        k = load_kub(sid)
        proverit(k)
        print(f"{sid} | {len(osi(k, None))} | {obem(k):.4e} | {obem(k, ['tf']):.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
