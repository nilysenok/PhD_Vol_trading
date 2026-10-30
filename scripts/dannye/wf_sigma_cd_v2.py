"""σ̂ C/D на wf, версия 2 (А-1; docs/wf_sigma_910_dsvd.md разд. 5; [30.10 19:30], [30.10 20:02], [30.10 20:49]).

Тот же формат, что у v1 (scripts/dannye/wf_sigma_cd.py, не тронут): sigma_hyb_h{1,5}_dev.parquet (date, ticker, sigma_pred_h{h} float32,
сортировка date, ticker), shagi.csv, proverka_sdvig.csv, wf_sigma_run.json ("v": 2), SHA256SUMS. Отличия от v1:
- источник — итог К-6 (<root>/data/regen/wf_k6/final_d0_s2emb_h{h}_test.parquet + sborka_shapka.json): шаги 1–8 прежние, 9–10 — на матрицах
  без дней выходных сессий (ДСВД); SHA итога сверяется с sborka_shapka.json до чтения (вместо sobrano спецификации R2);
- календарь и цель Д1 — features_pit_v3 без ДСВД (<root>/data/regen/features_pit_v3.parquet), конец ряда 2026-02-03;
- гейт «ДСВД нет»: ДСВД = сб/вс ≥ 2025 в v1 вне календаря матрицы (рабочая суббота 2025-11-01 торговая); их в v2 нет, дат после
  2026-02-03 нет; строки v1, которых нет в v2, — ровно даты ДСВД (перечень в json);
- гейт «шаги 1–8 (< 2025) = v1 побайтно»; Δ v1→v2 — медиана |Δ ln σ̂| по шагам и отдельно по понедельникам после ДСВД.
Части нарезки 29.10 (гейт 3 v1) не сверяются: их holdout 2025 — шаг 9 до К-6; равенство шагов 1–8 с v1 (а v1 с частями) — гейт «= v1».
Запуск: python3 scripts/dannye/wf_sigma_cd_v2.py --root КОРЕНЬ --out КАТАЛОГ"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wf_sigma_cd as W  # noqa: E402  гейты v1 (шаги, склейка, ≤ 2021 = ch4_dev, Д1) и формат

V = 2
KONEC = pd.Timestamp("2026-02-03")
WF0 = pd.Timestamp("2025-01-01")                    # с этой даты бывают ДСВД; < WF0 — шаги 1–8, прежние
K6 = "data/regen/wf_k6"
V1 = "data/regen/wf/ch4_sigma/sigma_hyb_h{h}_dev.parquet"
K6F = "final_d0_s2emb_h{h}_test.parquet"
KEY = W.KEY


def chitat_k6(root: Path, h: int) -> tuple[pd.DataFrame, str, str]:
    """SHA итога К-6 = sborka_shapka.json до чтения данных; -> (данные, sha, путь)."""
    p = root / K6 / K6F.format(h=h)
    W.period_v2.proverit_chtenie("wf", put=p)
    want = json.loads((root / K6 / "sborka_shapka.json").read_text())[f"vyhod_h{h}"]
    got = W.sha(p)
    if got != want:
        raise SystemExit(f"СТОП гейт 1: {p.name}: SHA {got[:16]}… ≠ sborka_shapka {want[:16]}… — не итог К-6")
    d = pd.read_parquet(p, columns=["date", "ticker", "year", "step", "rv_pred", "leg_raw", "F_sm", "design"])
    d["date"] = pd.to_datetime(d["date"])
    return d, got, str(p)


def dni_dsvd(v1: pd.DataFrame, kal: pd.DataFrame) -> set:
    """ДСВД = сб/вс-даты ≥ 2025 в v1, которых нет в календаре матрицы без ДСВД (рабочая суббота 2025-11-01 остаётся торговой)."""
    d = pd.to_datetime(v1["date"])
    return set(d[(d >= WF0) & (d.dt.dayofweek >= 5)]) - set(kal["date"])


def proverit_dni(s: pd.DataFrame, dsvd: set, h: int) -> None:
    """Дат ДСВД в v2 нет, конец ряда — 2026-02-03, дальше ничего."""
    vyh = s[s["date"].isin(dsvd)]
    if len(vyh):
        raise SystemExit(f"СТОП гейт ДСВД: h{h}: {vyh['date'].nunique()} дат ДСВД в v2 (напр. {vyh['date'].min().date()})")
    if s["date"].max() > KONEC:
        raise SystemExit(f"СТОП гейт конца: h{h}: данные после {KONEC.date()} (до {s['date'].max().date()})")
    if s["date"].max() != KONEC:
        raise SystemExit(f"СТОП гейт конца: h{h}: ряд кончается {s['date'].max().date()}, нужно {KONEC.date()}")


def sverit_v1(s: pd.DataFrame, v1: pd.DataFrame, dsvd: set, h: int) -> dict:
    """Шаги 1–8 (< 2025) = v1 побайтно; строки v1 вне v2 — ровно даты ДСВД, v2 ⊂ v1. -> перечень ДСВД-строк v1."""
    c = W.oc(h)
    v1 = v1.assign(date=pd.to_datetime(v1["date"]))[["date", "ticker", c]].sort_values(KEY).reset_index(drop=True)
    a, b = s[s["date"] < WF0].reset_index(drop=True), v1[v1["date"] < WF0].reset_index(drop=True)
    if not (a.equals(b) and a[c].dtype == b[c].dtype):
        raise SystemExit(f"СТОП гейт «= v1»: h{h}: шаги 1–8 (< 2025) не совпали с v1 побайтно ({len(a)} vs {len(b)} строк)")
    m = v1.merge(s[KEY], on=KEY, how="outer", indicator=True)
    if (m["_merge"] == "right_only").any():
        raise SystemExit(f"СТОП гейт «= v1»: h{h}: в v2 есть ключи, которых нет в v1")
    lo = m[m["_merge"] == "left_only"]
    if set(lo["date"]) != dsvd:
        raise SystemExit(f"СТОП гейт «= v1»: h{h}: v1-строки вне v2 ≠ даты ДСВД ({lo['date'].nunique()} vs {len(dsvd)} дат)")
    return dict(strok_v1_vne_v2=int(len(lo)), dat=int(lo["date"].nunique()), po_mesyacam=lo.groupby(lo["date"].dt.strftime("%Y-%m"))
                .size().to_dict(), dsvd_dat=sorted(str(x.date()) for x in lo["date"].unique()))


def delta(s: pd.DataFrame, v1: pd.DataFrame, d: pd.DataFrame, dsvd: set, h: int) -> dict:
    """|Δ ln σ̂| v1→v2 на общих ключах: по шагам и по понедельникам после ДСВД (за ним в v1 стоит дата ДСВД: вс или сб)."""
    c = W.oc(h)
    x = s.merge(v1.assign(date=pd.to_datetime(v1["date"])), on=KEY, suffixes=("", "_v1")).merge(
        d[KEY + ["step"]], on=KEY)
    x["dl"] = np.abs(np.log(x[c].astype("float64")) - np.log(x[c + "_v1"].astype("float64")))
    pn = x["date"].map(lambda t: t.dayofweek == 0 and any(t - pd.Timedelta(days=k) in dsvd for k in (1, 2)))
    f = lambda y: dict(n=int(len(y)), med=float(y.median()), p90=float(y.quantile(0.9)), max=float(y.max())) if len(y) else dict(n=0)  # noqa: E731
    return dict(po_shagam={int(k): f(g["dl"]) for k, g in x.groupby("step")}, ponedelniki_posle_dsvd=f(x.loc[pn, "dl"]),
                ostalnye_dni_shagov_9_10=f(x.loc[~pn & (x["step"] >= 9), "dl"]))


def kalendar_moex_sha(kal: pd.DataFrame) -> str:
    """SHA перечня торговых дат матрицы ≥ 2025 (основная сессия, без ДСВД)."""
    dd = sorted(kal.loc[kal["date"] >= WF0, "date"].dt.strftime("%Y-%m-%d").unique())
    return hashlib.sha256("\n".join(dd).encode()).hexdigest()


def run(root: Path, out: Path | None) -> dict:
    W.proverit_out(out)
    kp = root / W.KAL
    kal = pd.read_parquet(kp, columns=["date", "ticker", "rv"])
    kal["date"] = pd.to_datetime(kal["date"])
    rep, ser, shagi, d1t = dict(v=V), {}, [], []
    for h in W.HORIZONS:
        d, s_ist, put = chitat_k6(root, h)
        shagi.append(W.proverit_shagi(d, h))
        s = W.H.sigma_ch3(d, "rv_pred", W.oc(h))
        v1 = pd.read_parquet(root / V1.format(h=h))
        dsvd = dni_dsvd(v1, kal)
        proverit_dni(s, dsvd, h)
        rep[f"hyb_h{h}"] = dict(istochnik_910=dict(put=put, sha256=s_ist), design=W.DESIGN[h], skleyka=W.proverit_skleyku(s, kal, h),
                                stroki_do_2022_pobajtno_ch4_dev=W.proverit_dev(s, pd.read_parquet(root / W.CH4.format(h=h)), h),
                                v1=sverit_v1(s, v1, dsvd, h), v1_sha256=W.sha(root / V1.format(h=h)))
        rep[f"hyb_h{h}"]["delta_v1_v2"] = delta(s, v1, d, dsvd, h)
        d1t.append(W.d1(s, W.oc(h), kal, f"hyb_h{h}"))
        ser[h] = s
    rep["konec"] = str(max(x["date"].max() for x in ser.values()).date())
    kod = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=W.ROOT).stdout.strip()
    rep.update(okno=f"{min(x['date'].min() for x in ser.values()).date()}…{rep['konec']}", dni="osnovnaya_sessiya MOEX >= 2025, DSVD isklyucheny",
               zaderzhka="σ̂ на дату d известна на закрытии d (d0); шаг года t обучен на данных < t",
               chestnaya_polovina="2022+ — вне отбора дизайна; 2017–2021 — дизайн выбран на этих годах валидации",
               kod=kod, kod_sha256=W.sha(Path(__file__)), kalendar=W.KAL, kalendar_sha256=W.sha(kp),
               kalendar_moex_sha256=kalendar_moex_sha(kal), d1_ok=True)
    shagi_t, d1_t = pd.concat(shagi, ignore_index=True), pd.concat(d1t, ignore_index=True)
    if out is not None:
        pisat(out, ser, shagi_t, d1_t, rep)
    rep["_shagi"], rep["_d1"] = shagi_t, d1_t
    return rep


def pisat(out: Path, ser: dict, shagi_t: pd.DataFrame, d1_t: pd.DataFrame, rep: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for h, s in ser.items():
        s.to_parquet(out / Path(W.CH4.format(h=h)).name, index=False)
    shagi_t.to_csv(out / "shagi.csv", index=False)
    d1_t.to_csv(out / "proverka_sdvig.csv", index=False, float_format="%.6f")
    rep["vyhod"] = {p.name: W.sha(p) for p in sorted(out.glob("*.parquet"))}
    (out / "wf_sigma_run.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    (out / "SHA256SUMS").write_text("".join(f"{W.sha(p)}  {p.name}\n" for p in sorted(out.iterdir())
                                            if p.is_file() and p.name != "SHA256SUMS"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rep = run(a.root, a.out)
    print(rep["_d1"][["ryad", "period", "corr_k-2", "corr_k-1", "corr_k+0", "corr_k+1", "argmax", "ok"]].to_string(index=False))
    for h in W.HORIZONS:
        r = rep[f"hyb_h{h}"]
        print(f"hyb_h{h}: {r['skleyka']}; v1 вне v2: {r['v1']['strok_v1_vne_v2']}; Δ пн после ДСВД {r['delta_v1_v2']['ponedelniki_posle_dsvd']}")
    print(f"выход: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
