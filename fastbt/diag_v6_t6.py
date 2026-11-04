"""Диагностика v6 train4 (Ж-145), лист Д-8: таблица (6) — топ-10 каждой Sk против 20 к2 и 10 плацебо. Данных ≥ 2020 не читает.
Шаги: `schitat <top10.csv> <ledger> <кэш> <Sk,…>` — по точке 31 счёт (факт + 10 плацебо + 20 к2) через `vozm={"sig_fn"}` оценщика (не ledger),
JSON в кэш; `tablica <top10.csv> <кэш> <t6.csv>` — таблица и строка вывода. Запуск с FASTBT_MAIN=<теневой корень>.
Все счёты — медленный слой, одинаково для факта и контролей. Метрики: Sharpe дн. (основная), mSR нед., Ж-148 (диагностика): Sharpe норм.,
по сделкам, цена простоев. p = (1 + #{контроль ≥ факт}) / (n + 1) (`progon_kontrol.p_perm`). Все числа — потолок (in-sample)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import metriki_diag as md, metriki_e as me, progon_kontrol as pk, progon_sobol as ps, progon_v6 as v6
from fastbt.diag_v6_perescet import PERIOD, proverit_klyuch, zhurnal
from fastbt.diag_v6_pravila import PRAVILA

DOLYA_MIN = 0.5      # класс B: слабое место, если p95-перцентиль контролей превзошли < 50 % точек топ-10 Sk
SEED_PROHOD = {"s1": "seed", "s2": "seed2"}
VID = ("k2", "plac")
HAPKA = ('# статус: разработка train4 2016–2019, потолок (in-sample), после аудита',
         '# Ж-148 столбцы (Sharpe норм., по сделкам, цена простоев) — «диагностика», в пороги не входят; сделочный PnL = g − комиссия (над ставкой не очищен)',
         '# формат: "v": "1"')


def kontroli() -> list:
    """[(вид, параметр, sig_fn)]: 10 плацебо (seed 0…9) и 20 к2 (лаги k2_lags)."""
    return [("plac", s, pk.placebo_sig_fn(s)) for s in range(pk.N_PLACEBO)] + [("k2", g, pk.k2_sig_fn(g)) for g in pk.k2_lags()]


def metriki(x, daty, jr, tp) -> dict:
    """Метрики одного ряда x (дневной excess портфеля) и журнала сделок; tp — [тикеры, дни] знаковые позиции."""
    pnl = [t["pnl_do_izderzhek"] - t["komissiya"] for t in jr]
    ned = me.metriki_ned(x, daty)["msr_ned"]
    return dict(sharpe=float(md.sharpe_dn(x)), msr_ned=float(ned), norm=md.sharpe_norm(x, np.asarray(tp).T),
                sdelok=md.sharpe_sdelok(pnl, len(x) / md.GOD), cena=md.cena_prostoev(x, np.asarray(tp).T), n=len(pnl))


def bez_kraya(ev, fn, schet: list):
    """Сигнал контроля не ставится там, где у реального ряда его быть не может, а движок на NaN падает: (а) в прогреве, до s0 тикера
    (циклический сдвиг к2 заносит туда хвост ряда; σ движка там NaN); (б) на день d, если σ дня d+1 не определена (вход в последний
    день окна b). Число обнулённых дней — в schet[0] (в таблицу, `obrez`)."""
    def g(t, sig, dates):
        out = np.array(fn(t, sig, dates), float)
        sg = np.asarray(ev.inp.sigma[ev.r5.pt4["sigma"]][t], float)[:len(out)]
        bad = (~np.isfinite(np.r_[sg[1:], np.nan]) | (np.arange(len(out)) < ev.inp.meta[t]["s0"])) & (out != 0)
        schet[0] += int(bad.sum())
        out[bad] = 0.0
        return out
    return g


def schitat_ryad(sr, pt: dict, fn=None) -> dict:
    """Один счёт точки: fn = None — факт, иначе sig_fn контроля (с bez_kraya). Статус не ok — метрики NaN (n = 0)."""
    schet = [0]
    r, x, _ = sr.evr.evaluate_v2(pt, PERIOD, {"sig_fn": bez_kraya(sr.evr, fn, schet)} if fn else None)
    if r["status"] != "ok":
        return dict(sharpe=np.nan, msr_ned=np.nan, norm=np.nan, sdelok=np.nan, cena=np.nan, n=0, obrez=schet[0])
    jr, tp, _ = zhurnal(sr, ps.KOMISSIYA)
    return {**metriki(x, sr.daty, jr, tp), "obrez": schet[0]}


def schitat_tochku(sid: str, prohod: str, i: int, led: Path, zam) -> dict:
    pt, rz, _ = proverit_klyuch(sid, SEED_PROHOD[prohod], i, led / f"train4_v6_{prohod}", zam)   # klyuch точки = klyuch строки ledger
    ps.fabrika, sr = None, ps.sreda(sid, rz)
    fakt = schitat_ryad(sr, pt)
    return dict(sid=sid, prohod=prohod, sobol_i=i, fakt=fakt,
                kontroli=[dict(vid=v, par=p, **schitat_ryad(sr, pt, fn)) for v, p, fn in kontroli()])


def stroka(rec: dict, rang: int, p_fn=pk.p_perm) -> dict:
    """Строка точки: факт, по виду контроля p (Sharpe дн., mSR нед.), p95, медианы Ж-148; net_signala — Ж-148 п. 4."""
    f, q = rec["fakt"], PRAVILA["kontrol_percentil"]
    out = dict(sid=rec["sid"], rang=rang, sobol_i=rec["sobol_i"], **{f"fakt_{k}": f[k] for k in ("sharpe", "msr_ned", "norm", "sdelok", "cena", "n")})
    for v in VID:
        c = pd.DataFrame([k for k in rec["kontroli"] if k["vid"] == v])
        out[f"p_sharpe_{v}"], out[f"p_msr_{v}"] = p_fn(f["sharpe"], c["sharpe"]), p_fn(f["msr_ned"], c["msr_ned"])
        out[f"nan_{v}"] = int(c["sharpe"].isna().sum())          # контролей без Sharpe (нет сделок/статус не ok): в p они «не лучше факта»
        for k in ("sharpe", "norm", "sdelok"):
            ok = c[k].dropna()
            out[f"p95_{k}_{v}"], out[f"med_{k}_{v}"] = (float(np.percentile(ok, q)), float(np.median(ok))) if len(ok) else (np.nan, np.nan)
    out["net_signala"] = bool(f["norm"] <= out["p95_norm_plac"] and f["sdelok"] <= out["p95_sdelok_plac"])   # nan → False (не утверждаем)
    return out


def svod(df: pd.DataFrame) -> tuple:
    """(строки svod по Sk, вывод). Точка «лучше» вида v, если Sharpe дн. факта > p95 его контролей; слабое место: доля таких < DOLYA_MIN."""
    rows, slabye, nety = [], [], []
    for sid, g in df.groupby("sid", sort=True):
        d = {f"dolya_luchshe_{v}": float((g.fakt_sharpe > g[f"p95_sharpe_{v}"]).mean()) for v in VID}
        d |= {f"dolya_p01_{v}": float((g[f"p_sharpe_{v}"] <= 0.1).mean()) for v in VID}
        rows.append(dict(sid=sid, rang="svod", **d))
        if any(d[f"dolya_luchshe_{v}"] < DOLYA_MIN for v in VID):
            slabye.append(sid)
        nety += [f"{sid}#{r}" for r in g[g.net_signala].rang]
    sl = f"слабое место: kontrol_ne_luchshe, Sk: {', '.join(slabye)}" if slabye else "не слабое место"
    return pd.DataFrame(rows), f"(6): {sl}; Ж-148 п. 4 — сигнала нет у точек: {', '.join(nety) or 'нет'} (потолок: ранжирование и метрики на одних данных)"


def tablica(top10: Path, kesh: Path, out: Path) -> str:
    t = pd.read_csv(top10, comment="#")
    rows = []
    for r in t.itertuples():
        p = kesh / f"t6_{r.sid}_{r.prohod}_{r.sobol_i:06d}.json"
        rows.append(stroka(json.loads(p.read_text(encoding="utf-8")), int(r.rang)))
    df = pd.DataFrame(rows)
    sv, vyvod = svod(df)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"{HAPKA[0]}\n# вывод: {vyvod}\n{HAPKA[1]}\n{HAPKA[2]}\n")
        pd.concat([df, sv], ignore_index=True).to_csv(f, index=False)
    return vyvod


def schitat(top10: Path, led: Path, kesh: Path, sids) -> None:
    t = pd.read_csv(top10, comment="#")
    kesh.mkdir(parents=True, exist_ok=True)
    for sid in sids:
        zam = v6.zamorozka(led / "train4_morris", [sid], lambda s: ps.kub_put(s, ps.Rezhim("v6", PERIOD, "seed", False, None)))[0][2]
        for r in t[t.sid == sid].sort_values(["prohod", "rang"]).itertuples():
            res = schitat_tochku(sid, r.prohod, int(r.sobol_i), led, zam)
            (kesh / f"t6_{sid}_{r.prohod}_{r.sobol_i:06d}.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
            print(sid, r.prohod, r.sobol_i, "факт", res["fakt"]["sharpe"], flush=True)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 5 and a[0] == "schitat":
        schitat(Path(a[1]), Path(a[2]), Path(a[3]), a[4].split(","))
    elif len(a) == 4 and a[0] == "tablica":
        print(tablica(Path(a[1]), Path(a[2]), Path(a[3])))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
