"""Реверс р.1, лист R-ж для A_zashch (Ж-124 п. 1): затвор σ̂ поверх позиций ансамбля A (s9) на test1. Справочно, без выбора.
Запуск: python3 scripts/revers/r1_zh.py --period test1 --obekt A_zashch --sigma-dir <ch4_sigma Z2> [--n-k2 N] [--zamer]  (env Z2, как у r1_a_cd.py).
Затвор — то же определение (cd_s3_sigma.zatvor: p67, окно 252, ≥126, ранг по дню d включительно), σ̂ — те же файлы --sigma-dir, что у A_strogaya.
Способ как у TEST_C/D (r1_a_cd.portfeli): позиции = A_row s9 × затвор (маска обнуляет и удерживаемую позицию; у A_strogaya затвор запрещает только вход).
Комиссия 0,04 % (AS.COMM), платы за шорт нет (как a_cd_test1.csv; плата — лист r1_a_plata). Контроли: HAR-J (б), наивная √RV, к2 (круговой сдвиг маски по тикеру).
Объявлено до счёта: к2 и по mSR, и по Sharpe (дневной·√252, как a_cd); p = (1 + #{к2 ≥ факт}) / (n_конечных + 1); доля конечных печатается рядом."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis"), str(Path(__file__).resolve().parent)]
import cd_s3_sigma as S  # noqa: E402
import dve_metriki_a as D  # noqa: E402
import r1_a_cd as C  # noqa: E402
import shapka_lib as SL  # noqa: E402
from fastbt import a_strogaya as AS, metriki_e as me, s9, zona  # noqa: E402

VSE = "2022-2024"
GODY = (2022, 2023, 2024)
NAMES = {"hyb": ("затвор по σ̂ гибрида (диагностика, не H2)", "диагностика (не H2)"),
         "harj": ("контроль: затвор по HAR-J (б)", "kontrol"), "naive": ("контроль: затвор по наивной √RV", "kontrol")}
FILES = {"hyb": "sigma_hyb_h1_dev", "harj": "sigma_harj_b_h1_dev", "naive": "sigma_naive_dev"}
METR = ("msr", "sharpe")
NED = ("sharpe_ned", "sortino_ned", "msr_ned", "skew_ned", "kurt_ned", "n_nedel", "msr_dn_opredelen")   # Ж-124 п. 3, в конец строки


def p_znachenie(fakt: float, k2) -> float:
    """(1 + #{k2 ≥ факт}) / (n + 1), n — число КОНЕЧНЫХ k2; факт nan или n = 0 — nan (не подменяется)."""
    v = np.asarray(k2, float)
    v = v[np.isfinite(v)]
    if not np.isfinite(fakt) or len(v) == 0:
        return float("nan")
    return (1 + int((v >= fakt).sum())) / (len(v) + 1)


def ryad(eng, pos: np.ndarray, comm: float) -> dict:
    v = eng.port(pos, comm=comm)
    return {"x": v["excess"].to_numpy(float), "delo": v["exposure"].to_numpy(float) > 0,
            "dates": v.index.to_numpy().astype("datetime64[D]")}


def matrica_zatvora(eng, ser: dict) -> np.ndarray:
    """G[тикер, бар]: 1 — вход разрешён. σ̂ тикера на дату бара (последнее известное на close d); нет тикера в ряду — ошибка."""
    G = np.ones((len(eng.tickers), eng.nmax))
    for ti, t in enumerate(eng.tickers):
        sg = ser[t].reindex(pd.DatetimeIndex(eng.cache[t]["dates"])).to_numpy(float)
        G[ti, :eng.n_t[ti]] = S.zatvor(sg)
    return G


def sdvig_k2(eng, G: np.ndarray, rng) -> np.ndarray:
    """к2: круговой сдвиг маски каждого тикера (S.k2_mask, как у A_strogaya); последний день не сдвигается."""
    out = np.ones_like(G)
    for ti in range(len(eng.tickers)):
        a = G[ti, :eng.n_t[ti]] > 0
        out[ti, :eng.n_t[ti]] = S.k2_mask(lambda t, d, a=a: a, rng)(None, None)
    return out


def metriki(r: dict, g) -> dict:
    yr = r["dates"].astype("datetime64[Y]").astype(int) + 1970
    m = np.ones(len(yr), bool) if g == VSE else yr == g
    return {**D.stroka(r["x"][m], r["delo"][m], None, 0.0, np.zeros(int(m.sum()))), **me.stroka_ned(r["x"][m], r["dates"][m])}


def stroki(imya: str, tip: str, r: dict, baza: dict | None, k2: list | None = None) -> list:
    """Строки по годам и пулу. k2 (список рядов): p по mSR и по Sharpe и доли конечных — на каждой строке с фактом."""
    out = []
    for g in (*GODY, VSE):
        m = metriki(r, g)
        row = dict(stroka=imya, tip=tip, god=g, msr=m["msr"], sharpe=m["sharpe"], skew=m["skew"], kurt=m["kurt"], dni_v_dele=m["dni_v_dele"])
        for k in METR:
            row["d_%s_vs_baza" % k] = np.nan if baza is None else m[k] - metriki(baza, g)[k]
        if k2 is not None:
            kv = {k: np.array([metriki(x, g)[k] for x in k2]) for k in METR}
            row.update(n_k2=len(k2), **{"p_k2_" + k: p_znachenie(m[k], kv[k]) for k in METR},
                       **{"n_k2_konechnyh_" + k: int(np.isfinite(kv[k]).sum()) for k in METR})
            row["dolya_konechnyh_msr"] = row["n_k2_konechnyh_msr"] / len(k2)
            row["dolya_konechnyh_msr_ned"] = float(np.mean([np.isfinite(metriki(x, g)["msr_ned"]) for x in k2]))
        row.update({c: m[c] for c in NED})
        out.append(row)
    return out


def stroki_k2_sredn(k2: list, baza: dict) -> list:
    """Среднее по КОНЕЧНЫМ повторам к2 (n конечных — рядом; все nan — nan, без подмены)."""
    rows = []
    for g in (*GODY, VSE):
        mm = [metriki(x, g) for x in k2]
        b = metriki(baza, g)
        row = dict(stroka="контроль: к2 (сдвиг маски σ̂, среднее по конечным)", tip="kontrol", god=g, n_k2=len(k2), dni_v_dele=float(np.mean([m["dni_v_dele"] for m in mm])))
        for k in METR:
            v = np.array([m[k] for m in mm])
            f = v[np.isfinite(v)]
            row[k] = float(f.mean()) if len(f) else float("nan")
            row["d_%s_vs_baza" % k], row["n_k2_konechnyh_" + k] = row[k] - b[k], len(f)
        rows.append(row)
    return rows


def schitat(eng, ser: dict, comm: float, n_k2: int, seed: int = S.SEED) -> tuple:
    """(таблица, повторы к2): база A_row; затворы hyb/harj/naive; к2 по hyb."""
    A = eng.A_row
    baza = ryad(eng, A, comm)
    G = {k: matrica_zatvora(eng, ser[k]) for k in NAMES}
    res = {k: ryad(eng, A * eng.g_row(g), comm) for k, g in G.items()}
    rng = np.random.default_rng(seed)
    k2 = [ryad(eng, A * eng.g_row(sdvig_k2(eng, G["hyb"], rng)), comm) for _ in range(n_k2)]
    rows = stroki("A_zashch (s9) без затвора (база)", "baza", baza, None)
    for k, (nm, tip) in NAMES.items():
        rows += stroki(nm, tip, res[k], baza, k2 if k == "hyb" else None)
    df = pd.DataFrame(rows + stroki_k2_sredn(k2, baza))
    return df[[c for c in df if c not in NED] + list(NED)], k2


def tablica_k2(k2: list) -> pd.DataFrame:
    """По повтору: mSR, Sharpe, асимметрия и эксцесс пула — почему mSR не определён (mVaR ≤ 0 при тяжёлом хвосте)."""
    return pd.DataFrame([dict(povtor=i, **{k: metriki(x, VSE)[k] for k in (*METR, "skew", "kurt")}) for i, x in enumerate(k2)])


def zamer(eng, ser: dict, comm: float) -> None:
    """Время частей на одном тикере (затвор) и на одном повторе к2; оценка полного счёта по ним."""
    k = list(NAMES)[0]
    t = eng.tickers[0]
    t0 = time.perf_counter()
    S.zatvor(ser[k][t].reindex(pd.DatetimeIndex(eng.cache[t]["dates"])).to_numpy(float))
    zat = time.perf_counter() - t0
    G, rng, rep = matrica_zatvora(eng, ser[k]), np.random.default_rng(1), []
    for _ in range(5):
        t0 = time.perf_counter()
        ryad(eng, eng.A_row * eng.g_row(sdvig_k2(eng, G, rng)), comm)
        rep.append(time.perf_counter() - t0)
    p50, p90 = np.percentile(rep, [50, 90])
    print(f"замер: затвор 1 тикера {zat:.2f} с (×{len(eng.tickers)}×3 источника); повтор к2 p50 {p50:.2f} / p90 {p90:.2f} с")
    print(f"оценка полного счёта без загрузки кэша: p50 {zat * len(eng.tickers) * 3 + 20 * p50:.0f} / p90 {zat * len(eng.tickers) * 3 * 1.3 + 20 * p90:.0f} с")


def zapisat(out: Path, df: pd.DataFrame, k2: pd.DataFrame, period: str, sigma_dir: Path, n_k2: int) -> None:
    """Обе таблицы со статусом, коммитом и входами; σ̂ зоны Z2 (--sigma-dir) — с путём: одноимённый файл в ch4_dev — другой (dev)."""
    out.mkdir(parents=True, exist_ok=True)
    vh = SL.blok(period, [sigma_dir / (FILES[k] + ".parquet") for k in FILES])
    sh = f"{C.SHAPKA}; период {period}\n" + vh + zona.shapka_dividendy()
    sh2 = "# объект: A_zashch (s9), затвор σ̂ p67 поверх позиций A; плата за шорт 0; к2 %d повторов\n" % n_k2
    (out / "zh_sigma_zatvor_a_zashch_test1.csv").write_text(sh + sh2 + df.to_csv(index=False), encoding="utf-8")
    (out / "zh_sigma_zatvor_a_zashch_k2_test1.csv").write_text(
        sh + "# к2: контроль «сдвиг затвора», %d повторов\n" % n_k2 + k2.to_csv(index=False), encoding="utf-8")


def main_zashch(period: str, sigma_dir: Path | None, n_k2: int, out: Path, zam: bool = False) -> int:
    if period != "test1" or sigma_dir is None:
        raise SystemExit("A_zashch: только --period test1 и нужен --sigma-dir (σ̂ зоны Z2)")
    S.CH4, S.LAST = sigma_dir, pd.Timestamp("2024-12-31")
    AS.PER, AS.GODY = "test1", GODY            # period/зона в процессе; воркеров нет
    C.ustanovit_period(period)
    t0 = time.perf_counter()
    eng = s9.Engine(C.kesh(period))
    ser = {k: S.citat_sigma(v) for k, v in FILES.items()}    # citat_sigma режет по S.LAST: подставляем конец test1
    print(f"кэш и движок: {time.perf_counter() - t0:.1f} с", flush=True)
    if zam:
        return zamer(eng, ser, AS.COMM) or 0
    df, k2 = schitat(eng, ser, AS.COMM, n_k2)
    zapisat(out, df, tablica_k2(k2), period, sigma_dir, n_k2)
    print(df.round(3).to_string())
    return 0
