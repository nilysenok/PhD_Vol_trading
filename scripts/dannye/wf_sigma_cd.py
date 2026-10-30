"""σ̂ для C/D на walk-forward (решение дирижёра 30.10 14:20, п. 3; Ж-136, Ж-140): ряд того же фиксированного дизайна, обученный
покадрово, без переотбора, — единый ряд 2017-01-03 … конец данных σ̂ (2026-02-03) в формате data/regen/ch4_dev/.

Источник — выход сборки R2 26.10 21:35 (scripts/analysis/r2_sborka_test.py, SHA кода 62f9e401…; спецификация
models/designs/r2/r2_A_sborka.yaml) F_h(d0) final_d0_s2emb_h{1,5}_test.parquet: тот же файл, из которого ch4_dev_sigma.py построил
data/regen/ch4_dev/sigma_hyb_h{1,5}_dev.parquet (≤ 2021). После нарезки 29.10 (scripts/dannye/suzhenie_rezka.py) исходник цел и лежит
в test_locked/_ishodnye/…/R2/final (SHA = sobrano спецификации); части нарезки (усечённая копия, t1 2022–2024, holdout 2025) — его
срезы, сверяются построчно. Чтение — только при отметке снятия замка (period_v2.proverit_chtenie, docs/resheniya/wf_zamok_snyat.json).

Покадровость (контракт строки): строка года t — из шага k = t − 2016 (шаги 1–10 = тестовые годы 2017–2026). Шаг k обучен на данных
< 1 янв. года t: ноги и бустинг — годы < 2015 + k с ранней остановкой по году 2015 + k (vyr_etalony_finaly.py, r2_nogi.trmask);
множитель s² — по остаткам val шагов ≤ k, то есть годов ≤ t − 1, с эмбарго окна цели (r2_sborka_test.s2_steps, assert год < t);
гиперпараметры шагов 7–10 — точка шага 6 (без переотбора). Проверка в коде: year == 2016 + step и год даты == year у каждой строки,
одна F_sm на шаг, rv_pred = leg_raw × F_sm, один design (= fastbt.sigma_z2.DESIGN[h]).

Преобразование — h2_d0_ch4.sigma_ch3 (σ = √rv_pred, float32, сортировка date, ticker), как ch4_dev. Значение на дату d известно на
закрытии d (d0). Честная половина: 2022+ — вне отбора дизайна; 2017–2021 — дизайн выбирался на этих годах валидации (как ch4_dev).

Гейты (отказ — SystemExit, файлы не пишутся): (1) SHA исходника = sobrano до чтения; (2) шаги (выше); (3) части нарезки = исходник;
(4) ≤ 2021 побайтно = ch4_dev; (5) склейка: нет дублей date×ticker, у каждого тикера даты = даты features_pit_v3 (календарь входов
модели) от первой до последней даты σ̂ — без дыр; (6) Д1 по периодам 2017–2021, 2022–2024, 2025–конец: Спирмен ln σ̂²_d с
ln RV_{d+k} (RV = features_pit_v3.rv, как ch4_dev_sigma.py), k = −2…+1 в торговых днях тикера, RV не позже конца периода, медиана
по тикерам; проходит, если max_{k≠0} − corr_0 ≤ ε = 0,01; контроли σ_d := σ_{d−1} и σ_d := σ_{d+1} обязаны блокироваться.
Выход (<out>/): sigma_hyb_h{1,5}_dev.parquet (date, ticker, sigma_pred_h{h}; имена ch4_dev — каталог подставляется вместо ch4_dev),
proverka_sdvig.csv, shagi.csv, wf_sigma_run.json ("v": 1), SHA256SUMS.
Запуск: python3 scripts/dannye/wf_sigma_cd.py [--main ОСНОВНАЯ_КОПИЯ] --out КАТАЛОГ"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import h2_d0_ch4 as H  # noqa: E402  только sigma_ch3 (тот же код, что ch4_dev_sigma.py)
from fastbt import period_v2  # noqa: E402
from fastbt.sigma_z2 import DESIGN  # noqa: E402  строка design семейства по горизонту

V = 1
HORIZONS = (1, 5)
GOD0 = 2016                                   # шаг k — тестовый год GOD0 + k
SHAGI = tuple(range(1, 11))
DEV_END = pd.Timestamp("2022-01-01")
SPEC = "models/designs/r2/r2_A_sborka.yaml"
ISTOCHNIK = "data/regen/test_locked/_ishodnye/data/regen/test_locked/R2/final/final_d0_s2emb_h{h}_test.parquet"
CHASTI = {"usechennaya": "data/regen/test_locked/R2/final/final_d0_s2emb_h{h}_test.parquet",
          "t1_2022_2024": "data/regen/test1_2022_2024/regen/R2/final/final_d0_s2emb_h{h}__t1.parquet",
          "holdout_2025": "data/regen/test_locked/holdout_2025/data/regen/test_locked/R2/final/final_d0_s2emb_h{h}__h25.parquet"}
KAL = "data/regen/features_pit_v3.parquet"    # календарь входов модели и цель Д1 (столбец rv)
CH4 = "data/regen/ch4_dev/sigma_hyb_h{h}_dev.parquet"
PERIODY_D1 = (("2017-01-01", "2021-12-31"), ("2022-01-01", "2024-12-31"), ("2025-01-01", "2026-02-28"))
KS, EPS = (-2, -1, 0, 1), 0.01
KEY = ["date", "ticker"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def oc(h: int) -> str:
    return f"sigma_pred_h{h}"


# ======================= чтение исходника =======================

def proverit_sha(p: Path, spec: Path, h: int) -> str:
    """Гейт 1: SHA исходника = sobrano спецификации сборки — до чтения данных."""
    want = yaml.safe_load(Path(spec).read_text())["sobrano"][f"R2/final/final_d0_s2emb_h{h}_test.parquet"]
    got = sha(p)
    if got != want:
        raise SystemExit(f"СТОП гейт 1: {p.name}: SHA {got[:16]}… ≠ sobrano {want[:16]}… — не выход сборки R2")
    return got


def chitat(main: Path, h: int, spec: Path) -> tuple[pd.DataFrame, str]:
    p = main / ISTOCHNIK.format(h=h)
    period_v2.proverit_chtenie("wf", put=p)
    s = proverit_sha(p, spec, h)
    d = pd.read_parquet(p, columns=["date", "ticker", "year", "step", "rv_pred", "leg_raw", "F_sm", "design"])
    d["date"] = pd.to_datetime(d["date"])
    return d, s


# ======================= гейты =======================

def proverit_shagi(d: pd.DataFrame, h: int) -> pd.DataFrame:
    """Гейт 2: строка года t — из шага t − 2016, обученного на данных < t. -> таблица шагов."""
    if d.duplicated(KEY).any():
        raise SystemExit(f"СТОП гейт 2: h{h}: дубли date×ticker")
    st = d["step"].to_numpy()
    if not np.isin(st, SHAGI).all():
        raise SystemExit(f"СТОП гейт 2: h{h}: шаг вне {SHAGI[0]}–{SHAGI[-1]}")
    if not (d["year"].to_numpy() == GOD0 + st).all():
        raise SystemExit(f"СТОП гейт 2: h{h}: year ≠ {GOD0} + step — строка года t из шага, обученного с годом t")
    if not (d["date"].dt.year.to_numpy() == d["year"].to_numpy()).all():
        raise SystemExit(f"СТОП гейт 2: h{h}: год даты ≠ year — строка вне тестового года своего шага")
    if d.groupby("step")["F_sm"].nunique().max() != 1:
        raise SystemExit(f"СТОП гейт 2: h{h}: больше одной F_sm на шаг")
    rv = d["rv_pred"].to_numpy(float)
    if not (np.isfinite(rv).all() and (rv > 0).all()):
        raise SystemExit(f"СТОП гейт 2: h{h}: rv_pred не конечен или ≤ 0")
    if not np.allclose(rv, d["leg_raw"].to_numpy(float) * d["F_sm"].to_numpy(float), rtol=1e-12, atol=0):
        raise SystemExit(f"СТОП гейт 2: h{h}: rv_pred ≠ leg_raw × F_sm")
    u = d["design"].unique()
    if len(u) != 1 or str(u[0]) != DESIGN[h]:
        raise SystemExit(f"СТОП гейт 2: h{h}: design {list(u)} ≠ {DESIGN[h]!r}")
    t = d.groupby("step").agg(god=("year", "first"), F_sm=("F_sm", "first"), n=("date", "size"),
                              pervaya=("date", "min"), poslednyaya=("date", "max")).reset_index()
    t.insert(2, "obuchen_do", [f"{g}-01-01" for g in t["god"]])          # данные шага — строго раньше этой даты
    return t.assign(h=h)


def sverit_chasti(main: Path, d: pd.DataFrame, h: int) -> dict:
    """Гейт 3: каждая часть нарезки 29.10 — построчный срез исходника (ключ, rv_pred побайтно)."""
    out = {}
    for nm, tpl in CHASTI.items():
        p = main / tpl.format(h=h)
        period_v2.proverit_chtenie("wf", put=p)
        x = pd.read_parquet(p, columns=["date", "ticker", "step", "rv_pred"])
        x["date"] = pd.to_datetime(x["date"])
        m = x.merge(d[["date", "ticker", "step", "rv_pred"]], on=["date", "ticker", "step"], how="left",
                    suffixes=("", "_i"), validate="1:1")
        if m["rv_pred_i"].isna().any() or not np.array_equal(m["rv_pred"].to_numpy(), m["rv_pred_i"].to_numpy()):
            raise SystemExit(f"СТОП гейт 3: h{h}: часть {nm} не совпала с исходником сборки")
        out[nm] = dict(strok=len(x), pervaya=str(x["date"].min().date()), poslednyaya=str(x["date"].max().date()), sha256=sha(p))
    return out


def proverit_dev(s: pd.DataFrame, dev: pd.DataFrame, h: int) -> int:
    """Гейт 4: строки < 2022 побайтно = ch4_dev (даты, тикеры, float32). -> число строк."""
    a = s[s["date"] < DEV_END].reset_index(drop=True)
    b = dev.assign(date=pd.to_datetime(dev["date"]))[["date", "ticker", oc(h)]].sort_values(KEY).reset_index(drop=True)
    if not a.equals(b):
        raise SystemExit(f"СТОП гейт 4: h{h}: ≤ 2021 не совпадает с ch4_dev побайтно ({len(a)} vs {len(b)} строк)")
    return len(a)


def proverit_skleyku(s: pd.DataFrame, kal: pd.DataFrame, h: int) -> dict:
    """Гейт 5: нет дублей date×ticker; у каждого тикера даты σ̂ = даты календаря входов от первой до последней даты σ̂ тикера."""
    if s.duplicated(KEY).any():
        raise SystemExit(f"СТОП гейт 5: h{h}: дубли date×ticker после склейки")
    dyry = {}
    for tk, g in s.groupby("ticker"):
        k = kal.loc[kal["ticker"] == tk, "date"]
        k = set(k[(k >= g["date"].min()) & (k <= g["date"].max())])
        lishnie, net = set(g["date"]) - k, k - set(g["date"])
        if lishnie or net:
            dyry[tk] = dict(net=sorted(str(x.date()) for x in net)[:5], lishnie=sorted(str(x.date()) for x in lishnie)[:5])
    if dyry:
        raise SystemExit(f"СТОП гейт 5: h{h}: дыры/лишние даты против календаря входов: {dyry}")
    return dict(strok=len(s), tikerov=int(s["ticker"].nunique()), pervaya=str(s["date"].min().date()),
                poslednyaya=str(s["date"].max().date()))


def sdvig(s: pd.DataFrame, col: str, rv: pd.DataFrame, lo: str, hi: str) -> dict:
    """Д1: Спирмен ln σ̂²_d с ln RV_{d+k}, k в торговых днях тикера; σ̂ — даты в [lo, hi], RV — даты ≤ hi (как r2_loyo.sdvig:
    пары за концом периода не берутся, прошлое до lo — берётся); медиана по тикерам."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    res = {k: [] for k in KS}
    x = s[(s["date"] >= lo) & (s["date"] <= hi)]
    r_all = rv[(rv["date"] <= hi)]
    for tk, g in x.groupby("ticker"):
        r = r_all[r_all["ticker"] == tk].sort_values("date").reset_index(drop=True)
        pos = pd.Series(r.index.to_numpy(), index=r["date"]).reindex(g["date"]).to_numpy()
        m = ~np.isnan(pos)
        pos = pos[m].astype(int)
        ly = np.log(np.clip(g[col].to_numpy(np.float64)[m] ** 2, 1e-12, None))
        lr = np.log(np.clip(r["rv"].to_numpy(np.float64), 1e-12, None))
        for k in KS:
            j = pos + k
            ok = (j >= 0) & (j < len(r))
            if ok.sum() > 10:
                res[k].append(pd.Series(ly[ok]).corr(pd.Series(lr[j[ok]]), method="spearman"))
    med = {k: float(np.nanmedian(v)) if v else float("nan") for k, v in res.items()}
    am = max(med, key=lambda k: -np.inf if np.isnan(med[k]) else med[k])
    ok = bool(not np.isnan(med[0]) and all(med[k] - med[0] <= EPS for k in KS if k != 0 and not np.isnan(med[k])))
    return dict(corr=med, argmax=int(am), ok=ok, tikerov=int(x["ticker"].nunique()), strok=int(len(x)))


def sdvinut(s: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    """σ_d := σ_{d+k} по строкам тикера (−1 — отставание, +1 — утечка)."""
    d = s.sort_values(["ticker", "date"]).copy()
    d[col] = d.groupby("ticker")[col].shift(-k)
    return d.dropna(subset=[col]).sort_values(KEY).reset_index(drop=True)


def d1(s: pd.DataFrame, col: str, rv: pd.DataFrame, ryad: str) -> pd.DataFrame:
    rows = []
    for lo, hi in PERIODY_D1:
        for nm, x, rol in ((ryad, s, "ряд"), (f"kontrol:{ryad}_sdvig-1", sdvinut(s, col, -1), "контроль"),
                           (f"kontrol:{ryad}_utechka+1", sdvinut(s, col, +1), "контроль")):
            r = sdvig(x, col, rv, lo, hi)
            rows.append(dict(ryad=nm, rol=rol, period=f"{lo}…{hi}", tikerov=r["tikerov"], strok=r["strok"],
                             **{f"corr_k{k:+d}": r["corr"][k] for k in KS}, argmax=r["argmax"], ok=r["ok"], eps=EPS))
    t = pd.DataFrame(rows)
    plohie = t[(t["rol"] == "ряд") & ~t["ok"]]
    ne_blok = t[(t["rol"] == "контроль") & t["ok"]]
    if len(plohie) or len(ne_blok):
        raise SystemExit(f"СТОП гейт 6 (Д1): не прошли {plohie[['ryad', 'period']].values.tolist()}, "
                         f"контроли не блокируются {ne_blok[['ryad', 'period']].values.tolist()}")
    return t


# ======================= запуск =======================

def run(main: Path, out: Path | None, spec: Path = ROOT / SPEC) -> dict:
    kal = pd.read_parquet(main / KAL, columns=["date", "ticker", "rv"])
    kal["date"] = pd.to_datetime(kal["date"])
    rep, ser, shagi, d1t = dict(v=V), {}, [], []
    for h in HORIZONS:
        d, s_ist = chitat(main, h, spec)
        shagi.append(proverit_shagi(d, h))
        chasti = sverit_chasti(main, d, h)
        s = H.sigma_ch3(d, "rv_pred", oc(h))
        n_dev = proverit_dev(s, pd.read_parquet(main / CH4.format(h=h)), h)
        sk = proverit_skleyku(s, kal, h)
        d1t.append(d1(s, oc(h), kal, f"hyb_h{h}"))
        vyh = s[s["date"].dt.dayofweek >= 5]
        ser[h] = s
        rep[f"hyb_h{h}"] = dict(istochnik=ISTOCHNIK.format(h=h), istochnik_sha256=s_ist, design=DESIGN[h], chasti=chasti,
                                stroki_do_2022_pobajtno_ch4_dev=n_dev, skleyka=sk,
                                subboty_voskr=dict(dney=int(vyh["date"].nunique()),
                                                   po_godam=vyh.groupby(vyh["date"].dt.year)["date"].nunique().to_dict()))
    shagi_t, d1_t = pd.concat(shagi, ignore_index=True), pd.concat(d1t, ignore_index=True)
    kod = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    rep.update(okno=f"{min(x['date'].min() for x in ser.values()).date()}…{max(x['date'].max() for x in ser.values()).date()}",
               zaderzhka="σ̂ на дату d известна на закрытии d (d0); шаг года t обучен на данных < t",
               chestnaya_polovina="2022+ — вне отбора дизайна; 2017–2021 — дизайн выбран на этих годах валидации",
               kod=kod, kod_sha256=sha(Path(__file__)), spec_sha256=sha(spec), kalendar=KAL, kalendar_sha256=sha(main / KAL),
               d1_ok=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for h, s in ser.items():
            s.to_parquet(out / Path(CH4.format(h=h)).name, index=False)
        shagi_t.to_csv(out / "shagi.csv", index=False)
        d1_t.to_csv(out / "proverka_sdvig.csv", index=False, float_format="%.6f")
        rep["vyhod"] = {p.name: sha(p) for p in sorted(out.glob("*.parquet"))}
        (out / "wf_sigma_run.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
        (out / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in sorted(out.iterdir())
                                                if p.is_file() and p.name != "SHA256SUMS"))
    rep["_shagi"], rep["_d1"] = shagi_t, d1_t
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, default=H.MAIN)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rep = run(a.main, a.out)
    d = rep.pop("_d1")
    rep.pop("_shagi")
    print(d[["ryad", "period", "corr_k-2", "corr_k-1", "corr_k+0", "corr_k+1", "argmax", "ok"]].to_string(index=False))
    for h in HORIZONS:
        r = rep[f"hyb_h{h}"]
        print(f"hyb_h{h}: {r['skleyka']}, ≤2021 = ch4_dev побайтно ({r['stroki_do_2022_pobajtno_ch4_dev']} строк), "
              f"сб/вс {r['subboty_voskr']}")
    print(f"выход: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
