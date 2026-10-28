"""Раунд 1 реверса, лист R-ж (Ж-117): разрез по терцилям σ̂ для S3 — диагностика, не H2.
Запуск (один процесс): python3 scripts/revers/r1_zh.py --period {train4,test1} [--sigma-dir <каталог σ̂ зоны Z2>] [--n-k2 N]
train4 — воспроизводит образец #483 (`scripts/analysis/cd_s3_sigma.py`, docs/tables/cd_s3/sigma_zatvor.csv): окно 2018–2019, к2 100 повторов.
test1 — 2022–2024 (прогрев 2021), к2 20 повторов; требует поддержки period="test1" (лист R-P) и каталога σ̂ зоны Z2 (--sigma-dir).
Затвор: вход запрещён, если ранг σ̂ тикера в окне 252 (доля строго ниже) >= p67, т.е. торгуются два нижних терциля. Выбора и перебора нет: p67 заморожен.
Логика — функции cd_s3_sigma (одно определение); здесь только параметры периода."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import cd_s3_sigma as S  # noqa: E402

PERIODS = {"train4": dict(gody=(2018, 2019), last="2019-12-31", n_k2=100),
           "test1": dict(gody=(2022, 2023, 2024), last="2024-12-31", n_k2=20)}
VSE = "2018-2019"   # метка итога в cd_s3_sigma (все годы окна); для test1 переименовывается в «2022-2024»
NAMES = {"rv22": ("затвор по реализованной σ (rv22), справочно", "spravochno"), "hyb": ("затвор по σ̂ гибрида (H2)", "H2"),
         "harj": ("контроль: затвор по HAR-J (б)", "kontrol"), "naive": ("контроль: затвор по наивной √RV", "kontrol")}


def nastroit(period: str, sigma_dir: Path | None):
    """Подставляет период в модуль-образец (его функции читают GODY/LAST/CH4 как глобальные)."""
    from fastbt import a_strogaya as AS
    p = PERIODS[period]
    S.GODY, S.LAST = p["gody"], pd.Timestamp(p["last"])
    if sigma_dir is not None:
        S.CH4 = sigma_dir
    if period == "test1":
        if sigma_dir is None:
            raise SystemExit("test1: нужен --sigma-dir (σ̂ зоны Z2) и period='test1' из R-P")
        AS.PER, AS.GODY = "test1", p["gody"]
    return AS, p


def zapisat(df: pd.DataFrame, path: Path, period: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if period == "test1":
            f.write("# реверс раунд 1, 2022–2024, справочно, без выбора\n")
        df.to_csv(f, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(PERIODS), required=True)
    ap.add_argument("--sigma-dir", type=Path, default=None)
    ap.add_argument("--n-k2", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    AS, p = nastroit(a.period, a.sigma_dir)
    n_k2 = a.n_k2 or p["n_k2"]
    out = a.out or ROOT / "docs/tables/revers_r1"
    ev = AS.evaluatory("S3")[1]
    baza = S.schitat(ev)
    mod = {k: S.na_dates(S.citat_sigma(v)) for k, v in {**S.FILES, "naive": "sigma_naive_dev"}.items()}
    zat = {k: (lambda t, d, f=f: S.zatvor(f(t, d))) for k, f in mod.items()}
    rv22 = lambda t, dates: S.zatvor(ev.inp.sigma["rv22"][t][:len(dates)])                    # noqa: E731
    res = {"rv22": S.schitat(ev, S.sig_fn(rv22)), **{k: S.schitat(ev, S.sig_fn(z)) for k, z in zat.items()}}
    rng = np.random.default_rng(S.SEED)
    k2 = [S.schitat(ev, S.sig_fn(S.k2_mask(zat["hyb"], rng))) for _ in range(n_k2)]
    msr_k2 = np.array([S.stroki_msr(r, VSE) for r in k2])
    p_k2 = float((msr_k2 >= S.stroki_msr(res["hyb"], VSE)).mean())
    rows = S.stroki("A_strogaya только S3 (база)", "baza", baza, None)
    for k, (nm, tip) in NAMES.items():
        rows += S.stroki(nm, tip, res[k], baza, p_k2=p_k2 if k == "hyb" else np.nan)
    for g in (*p["gody"], VSE):
        ids = p["gody"] if g == VSE else (g,)
        r = dict(stroka="контроль: к2 (сдвиг маски σ̂, среднее %d)" % n_k2, tip="kontrol", god=g,
                 msr=float(np.mean([S.stroki_msr(x, g) for x in k2])),
                 sdelok=float(np.mean([x["cnt"][np.isin(x["god"], ids)].sum() / 2 for x in k2])))
        r["d_vs_baza"] = r["msr"] - S.stroki_msr(baza, g)
        rows.append(r)
    df = pd.DataFrame(rows)
    if a.period == "test1":
        df["god"] = df["god"].replace(VSE, "2022-2024")
    zapisat(df, out / f"zh_sigma_zatvor_{a.period}.csv", a.period)
    print(df.to_string(), p_k2, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
