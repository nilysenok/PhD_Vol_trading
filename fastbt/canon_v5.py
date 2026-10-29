"""Л8: канон точки куба v5 (Ж-127 п. 2–3; docs/otchety/2026-10-29_setka_v5.md §1, §2.3, §3 «N для DSR»). Данных не читает.

Куб — fastbt/designs/v5/kub_S{k}.yaml (v: 3). Значение составной оси — строка «единица:число[:число]» (`stop_nach: atr:2.0`).
canon_v5(sid, pt) → точка в каноне или None (комбинация несовместима — выбрасывается до счёта):
  свёртки (псевдонимы N не плодят): rang_tip → tikernyj, если ранг не используется (m_sily = 1, тейк не от силы, безубыток не
  от ранга); rsi_th тренда > 0 → RSI_TREND (у тренда порог не действует, только RSI > 50 / < 50, zh126_obshchee.yaml);
  stop_nach chand:k → atr:k (Ъ Ж-129 (6): fixed_atr = chandelier — одна ось);
  выбрасываются: безубыток atr:k без стопа в ATR и rang:c при стопе в % (ed_s — одна единица у стопа и безубытка,
  exec_e_types.Params.ed_s); у тренда — значения, которых нет у стоп-заявки (exec_e.py:112, exec_e_v6.py:27; см. DOPUSK).
Линия сигнала у тренда (S3/S4) не ось: вход и выход — стоп-заявки на уровнях канала (Ж-091, etap1_eval_v2._params), поэтому
trail liniya, tejk rodnoy и smena у тренда в куб не входят; у контртренда liniya и parab — недопустимы (proverit).
"""
from __future__ import annotations

import itertools
import math
import sys
import warnings
from pathlib import Path

import yaml
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent.parent
KUB = ROOT / "fastbt/designs/v5"
SIDS = [f"S{k}" for k in range(1, 7)]
TREND = {"S3", "S4"}
SEKCII = ("signal", "filtr", "vhod", "vyhod")
NET, RSI_TREND = "net", 20
SIGNAL = {"S1": ("ma_tip", "ma_window", "dev_tip", "z_entry", "z_exit"), "S2": ("ma_tip", "bb_window", "bb_std", "podtv"),
          "S3": ("okno", "podtv"), "S4": ("period", "mnozh"),
          "S5": ("pivot_type", "pivot_period", "uroven", "atr_filtr", "filter_set"),
          "S6": ("vwap_window", "dev_mult", "dir_bias", "filter_set")}
# допустимые единицы составных осей по типу (у тренда — что умеет стоп-заявка)
DOPUSK = {
    "kontr": dict(stop_nach={"pct", "atr", "chand", "sigma"}, tejk={NET, "rr", "pct", "atr", "rodnoy", "sigh", "sila"},
                  trail={NET, "atr", "sigma", "step"}, bezubytok={NET, "atr", "rang"}),
    "trend": dict(stop_nach={"sigma"}, tejk={NET, "rr", "sigh", "sila"}, trail={NET, "atr", "sigma", "parab"},
                  bezubytok={NET}, stop_v_plyus={NET}, pir={0}),
}
_F = {"adx_th", "atr_lo", "vol_th", "rsi_th", "mom_n", "dow"}       # ключи fastbt/signals/v5_filtr.primenit
FILTR = {"kontr": _F | {"squeeze_pct"}, "trend": _F}                  # squeeze — только контр/диапазон (zh126_obshchee.yaml)
PRAVILA = ("stop_nach", "bezubytok", "m_sily", "tejk", "rang_tip", "rsi_th")   # оси, которые читает canon_v5


def edinica(v) -> str:
    """Единица составного значения (`atr:2.0` → atr); простое значение — само значение."""
    return v.split(":")[0] if isinstance(v, str) and ":" in v and v[0].isalpha() else v


def tip(sid: str) -> str:
    return "trend" if sid in TREND else "kontr"


def load_kub(sid: str, root: Path = KUB) -> dict:
    return yaml.safe_load((Path(root) / f"kub_{sid}.yaml").read_text(encoding="utf-8"))


def osi(kub: dict) -> list:
    """[(имя, значения)] в порядке размерностей Соболя."""
    return [(n, list(v["setka"])) for sec in SEKCII for n, v in kub.get(sec, {}).items()]


def proverit(kub: dict) -> None:
    """Формат v: 3 сетки v5; сигнальные оси = оси семьи; каждое значение допустимо для типа (иначе Соболь тратит точки)."""
    sid = kub["sid"]
    if kub.get("v") != 3 or kub.get("setka") != "v5" or kub.get("tip") != tip(sid):
        raise ValueError(f"{sid}: нужен v: 3, setka: v5, tip: {tip(sid)}")
    if tuple(kub["signal"]) != SIGNAL[sid]:
        raise ValueError(f"{sid}: оси signal {list(kub['signal'])} ≠ {SIGNAL[sid]}")
    if set(kub["filtr"]) - FILTR[tip(sid)]:
        raise ValueError(f"{sid}: фильтры {sorted(set(kub['filtr']) - FILTR[tip(sid)])} v5_filtr.primenit не считает")
    imena = [n for n, _ in osi(kub)]
    if len(set(imena)) != len(imena) or set(imena) & set(kub.get("yakorya_progon", {})):
        raise ValueError(f"{sid}: ось повторяется или стоит и в кубе, и в якорях zh047")
    for n, g in osi(kub):
        dop = DOPUSK[tip(sid)].get(n)
        if dop is not None and {edinica(v) for v in g} - dop:
            raise ValueError(f"{sid}: {n} — недопустимо для {tip(sid)}: {sorted(map(str, {edinica(v) for v in g} - dop))}")
        if len(set(map(str, g))) != len(g):
            raise ValueError(f"{sid}: в оси {n} повтор значения")


def canon_v5(sid: str, pt: dict):
    """Точка → канон (dict) или None (несовместима). Читает только оси PRAVILA; остальные не трогает."""
    q = dict(pt)
    st = q.get("stop_nach")
    if edinica(st) == "chand":                                  # Ъ Ж-129 (6)
        q["stop_nach"] = st = "atr:" + st.split(":", 1)[1]
    bz = edinica(q.get("bezubytok", NET))
    if (bz == "atr" and edinica(st) != "atr") or (bz == "rang" and edinica(st) == "pct"):
        return None                                             # ed_s одна на стоп и безубыток
    rang_nuzhen = str(q.get("m_sily", "1")) != "1" or edinica(q.get("tejk")) == "sila" or bz == "rang"
    if "rang_tip" in q and not rang_nuzhen:
        q["rang_tip"] = "tikernyj"
    if sid in TREND and q.get("rsi_th", 0) > 0:
        q["rsi_th"] = RSI_TREND
    return q


def obem(kub: dict) -> int:
    """Номинальный объём куба (произведение уровней осей)."""
    return math.prod(len(g) for _, g in osi(kub))


def obem_kanon(kub: dict) -> tuple:
    """(после свёрток — различных канонов, выброшено комбинаций): перебор по осям PRAVILA × произведение остальных."""
    sid, ox = kub["sid"], osi(kub)
    pr = [(n, g) for n, g in ox if n in PRAVILA]
    ost = math.prod(len(g) for n, g in ox if n not in PRAVILA)
    kanony, vybr = set(), 0
    for comb in itertools.product(*[g for _, g in pr]):
        c = canon_v5(sid, dict(zip([n for n, _ in pr], comb)))
        if c is None:
            vybr += 1
        else:
            kanony.add(tuple(sorted((k, str(v)) for k, v in c.items())))
    return len(kanony) * ost, vybr * ost


def tochki_v5(kub: dict, n0: int, n1: int, seed: str = "seed") -> tuple:
    """([(i, канон)], {vybrosheno, svernuto}): точка i — floor(u·len(ось)) i-й точки Соболя (scramble, kub.sobol[seed])."""
    proverit(kub)
    ox, s = osi(kub), kub["sobol"]
    sob = qmc.Sobol(len(ox), scramble=s["scramble"], seed=s[seed])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                         # n не степень двойки — продолжение последовательности
        if n0:
            sob.fast_forward(n0)
        u = sob.random(n1 - n0)
    out, sch = [], dict(vybrosheno=0, svernuto=0)
    for j, row in enumerate(u):
        pt = {n: g[min(int(x * len(g)), len(g) - 1)] for (n, g), x in zip(ox, row)}
        c = canon_v5(kub["sid"], pt)
        if c is None:
            sch["vybrosheno"] += 1
            continue
        sch["svernuto"] += c != pt
        out.append((n0 + j, c))
    return out, sch


def k3_kub(kub: dict) -> dict:
    """Куб прохода к3 (Ж-130): оси из раздела k3 — подмножество v4, остальные — одно значение yakor (вне Соболя)."""
    k3, out = kub["k3"], {k: v for k, v in kub.items() if k not in SEKCII}
    for sec in SEKCII:
        out[sec] = {}
        for n, v in kub.get(sec, {}).items():
            g = k3.get(n, [v.get("yakor", v["setka"][0])] if sec != "signal" else None)
            if g is None or any(x not in v["setka"] for x in g):
                raise ValueError(f"{kub['sid']}: k3.{n} — нужен список значений из setka")
            out[sec][n] = {"setka": list(g)}
    return out


def main() -> int:
    print("sid | осей | объём номинальный | после свёрток | выброшено | к3 номинальный")
    for sid in SIDS:
        k = load_kub(sid)
        proverit(k)
        v, (vk, vb) = obem(k), obem_kanon(k)
        print(f"{sid} | {len(osi(k))} | {v:.4e} | {vk:.4e} | {vb:.4e} | {obem(k3_kub(k)):.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
