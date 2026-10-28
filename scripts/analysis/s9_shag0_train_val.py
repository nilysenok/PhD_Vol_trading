"""Шаг 0, A и её части: train 2018–2019 и validation 2020–2021 отдельно (0,04 %) → shag0_semyi_train_val.csv.

    FASTBT_MAIN=<своя копия> python3 scripts/analysis/s9_shag0_train_val.py     # реальный счёт (нужен кэш s9 v2)
    python3 scripts/analysis/s9_shag0_train_val.py --selftest                   # синтетика; без аргументов — то же
Анализ готовых рядов A (не сетка, не подбор): строки A, к3, «A только Sk», «A без Sk» (k = 1…6).
Столбцы на каждый период: Шарп (ddof=1), mSR и MDD (fastbt.metriki_e на избыточной), доля дней в рынке,
сделок в год (смен позиции пары «тикер × семья» за год), оборот (сторон в год, как в shag0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import metriki_e as me  # noqa: E402
from fastbt import shag0  # noqa: E402

OUT = ROOT / "docs" / "tables" / "strat_dev_v3" / "shag0_semyi_train_val.csv"
PERIODS = {"train_2018_2019": (2018, 2019), "val_2020_2021": (2020, 2021)}
COMM = 0.0004


def split_masks(index) -> dict:
    """Маски периодов по годам индекса дат; периоды не пересекаются (assert)."""
    y = pd.DatetimeIndex(index).year
    m = {k: np.asarray((y >= a) & (y <= b)) for k, (a, b) in PERIODS.items()}
    ms = list(m.values())
    assert not (ms[0] & ms[1]).any(), "train и validation пересекаются"
    return m


def trades_per_year(df: pd.DataFrame, mask_dates) -> float:
    """df: date, key (тикер×семья), pos. Сделка — день, когда позиция ключа отличается от предыдущей
    (вход в первый день с позицией ≠ 0 считается). Делим на число лет периода."""
    d = df.sort_values(["key", "date"])
    prev = d.groupby("key")["pos"].shift(1).fillna(0.0)
    chg = (d["pos"].values != prev.values) & mask_dates(d["date"])
    yrs = max(len(pd.unique(d.loc[mask_dates(d["date"]), "date"].dt.year)), 1)
    return float(chg.sum() / yrs)


def row_for(label: str, p: pd.DataFrame, posdf: pd.DataFrame) -> dict:
    out = dict(pometka=shag0.POMETKA, row=label, comm_pct=COMM * 100)
    for name, m in split_masks(p.index).items():
        q = p[m]
        ex = q["excess"].values
        lo, hi = PERIODS[name]
        inper = lambda s, lo=lo, hi=hi: ((s.dt.year >= lo) & (s.dt.year <= hi)).values   # noqa: E731
        out.update({f"sharpe_{name}": shag0.sharpe(ex), f"msr_{name}": me.msr(ex), f"mdd_pct_{name}": me.mdd(ex) * 100,
                    f"dni_v_rynke_pct_{name}": float(np.mean(q["exposure"].values > 0) * 100),
                    f"sdelok_v_god_{name}": trades_per_year(posdf, inper),
                    f"oborot_storon_v_god_{name}": float(q["turnover"].mean() * 252)})
    return out


def run() -> None:
    from fastbt import s9
    from fastbt.fast_kontrol import k3_sids
    s9.use_v2(COMM)
    eng = s9.Engine(s9.load_cache())
    A, sidr = np.asarray(eng.A_row, float), shag0.sid_row(eng)
    dates = pd.to_datetime(eng.dates[np.asarray(eng.r["date"])])
    key = pd.Series(np.array(eng.tickers)[np.asarray(eng.r["tick"])]).astype(str).values + "|" + np.asarray(sidr).astype(str)

    def one(label, pos):
        return row_for(label, eng.port(pos, comm=COMM), pd.DataFrame(dict(date=dates, key=key, pos=pos)))
    rows = [one("A", A), one("к3: A без S5/S6", shag0.zero_sids(A, sidr, k3_sids()))]
    for s in shag0.SIDS:
        rows.append(one(f"A только {s}", shag0.keep_sids(A, sidr, [s])))
    for s in shag0.SIDS:
        rows.append(one(f"A без {s}", shag0.zero_sids(A, sidr, [s])))
    pd.DataFrame(rows).to_csv(OUT, sep=";", decimal=",", index=False, float_format="%.4f", encoding="utf-8-sig")
    print("записано", OUT, len(rows), "строк")


def selftest() -> None:
    idx = pd.bdate_range("2018-01-01", "2021-12-31")
    m = split_masks(idx)
    assert not (m["train_2018_2019"] & m["val_2020_2021"]).any()
    assert (m["train_2018_2019"] | m["val_2020_2021"]).all() and idx[m["val_2020_2021"]].year.min() == 2020
    # мутация «validation включает 2019» должна ловиться проверкой непересечения
    good = dict(PERIODS)
    PERIODS["val_2020_2021"] = (2019, 2021)
    try:
        split_masks(idx)
        raise SystemExit("мутация «validation включает 2019» не поймана")
    except AssertionError:
        pass
    finally:
        PERIODS.clear()
        PERIODS.update(good)
    ex = np.random.default_rng(3).normal(0.0005, 0.01, len(idx))
    p = pd.DataFrame(dict(excess=ex, total=ex, exposure=0.5, turnover=0.01), index=idx)
    pos = pd.DataFrame(dict(date=idx, key="X|S1", pos=np.where(idx.year == 2018, 1.0, 0.0)))   # 1 вход (2018) + 1 выход (2019-01)
    r = row_for("t", p, pos)
    assert abs(r["sharpe_train_2018_2019"] - shag0.sharpe(ex[idx.year <= 2019])) < 1e-12
    assert abs(r["sharpe_val_2020_2021"] - shag0.sharpe(ex[idx.year >= 2020])) < 1e-12
    assert r["sdelok_v_god_train_2018_2019"] == 1.0 and r["sdelok_v_god_val_2020_2021"] == 0.0   # 2 сделки / 2 года; в val пусто
    assert abs(r["oborot_storon_v_god_val_2020_2021"] - 2.52) < 1e-9
    print("OK")


if __name__ == "__main__":
    selftest() if (len(sys.argv) == 1 or "--selftest" in sys.argv) else run()
