"""W5 — диагностика перед выбором гипотезы (Ж-250 пп. 3–4). Перенос `w5/diag.py` из scratchpad без изменения логики. Только сохранённые ряды; обучения нет.
Запуск из корня репозитория:
`python3 -W ignore -m models_v2.s16.x1_w5_diag --check` — пересчёт и побитная сверка строк данных (без шапки-комментария) `W5_DIAG.csv` и `W5_TABLE0.csv`; ничего не пишет (sha файлов до и после);
`--check-verdict` — вердикт выведен из `W5_DIAG.csv` (гейт A.G2: `прошли: B_level/gap`, проверено 24); `--write` — единственный пишущий режим (явным флагом); без флагов — сухой прогон.
Публичное API: `build() -> (x, st)` — окно M (колонки `x1_w3_core.load()` + `ZB`, `yr`, `half`, `l_X1|M|C|K1|K2|K3|ZB`) и словарь состояний `st` (строковые группы, `na` — меньше 60 прошлых значений);
`events(x)`, `terc_past(v, minpast=60)` (PIT-терцили: пороги по значениям строго до дня i), `compute(x, st) -> DataFrame` (блоки A–E), `diag_lines(df)`, `check(diag, table0) -> list` замечаний,
`sha(rel)`. Блоки: `x1_w5_diag_abc` (A–C), `x1_w5_diag_d` (D), `x1_w5_diag_t` (E и W5_TABLE0), `x1_w5_diag_v` (вердикт).

ОБЪЯВЛЕНО ДО СЧЁТА (11.11.2026, архитектор):
Окно M — как W1/W3: x1 архив + продолжение, t⁺ 2018-01-04…2026-11-10, n = 2198. Половины: h1 = t⁺ 2018–2021, h2 = t⁺ 2022–2026.
Потери l = RV/F − ln(RV/F) − 1. D_M = l_X1 − l_M, D_K2 = l_X1 − l_K2.

Состояния (все известны на закрытии d; терцили — пороги по значениям строго до дня i, не меньше 60 прошлых, иначе группа na):
  ev_cbr   — плановое заседание ЦБ по ставке в день t⁺ (cbr.csv, known_from ≤ d; внеочередные не PIT);
  ev_cpius — публикация CPI США в день t⁺ (15:30/16:30 МСК, внутри основной сессии);
  ev_fomc  — решение FOMC в интервале [d, t⁺) (после закрытия основной сессии d, до сессии t⁺);
  ev_exp   — квартальная экспирация MOEX в день t⁺ (expirations.csv, версия с known_from ≤ d < known_to);
  ev_rebal — t⁺ — последний торговый день перед вступлением ребалансировки IMOEX (imoex_rebal_official.csv, known_from ≤ d);
  ev_any   — любое из пяти;
  dow      — день недели t⁺;  gap — (t⁺ − d) − 1 календарных дней > 0;
  e        — e_d (колонка s x1);
  sgnCM    — знак ln(C/M);  tCM — терцили ln(C/M);  absCM — терцили |ln(C/M)|;
  tK1M     — терцили ln(K1/M);  absK23 — терцили |ln(K2/K3)|;  lvl — терцили ln M;
  v        — терцили ln RV_d − ln mean(RV_{d−1…d−20});  err — терцили последней закрытой лог-ошибки X1 ln(RV_d / X1 пары с t⁺ = d);
  pi       — терцили π X1.

Блок A (где X1 проигрывает): по группам и окнам (h1, h2, all, годы) — n, средние D_M, D_K2, вклад в Δ X1 к K2 (п. п., как W1),
  число периодов (2018…2025, 2026) с тем же знаком среднего D_K2, что на окне M.
Блок B (может ли механизм по состоянию дать выигрыш в ОБЕИХ половинах) — перенос между половинами (оценка на одной, счёт на другой):
  B_level: F = X1·c_G, c_G = mean(RV/X1) группы (QLIKE-оптимальный множитель); контроль — общий c;
  B_trust (только состояния C/M, π, e): F = M^(1−π_G)·C^π_G, π_G — лучший на сетке 0…1 шаг 0,05; контроль — один π.
  Δ — в % QLIKE X1 и в % QLIKE контроля.
Критерий «основание в обеих половинах» (объявлен до счёта): перенос h1→h2 и h2→h1 оба < 0 к контролю и к X1;
  перенос h1→(h2 без 2022) < 0 к контролю; направление внутриполовинного эффекта одно в h1 и h2.
Блок C (события): по годам среднее ln(RV/F) в днях события минус в остальных — для X1, K1, K2 (смещение, которое модели не видят).
Блок D (`x1_w5_diag_d`) добавлен ПОСЛЕ просмотра A–C: устойчивость единственного состояния, прошедшего критерий (gap); критерия не меняет.
Блок E (`x1_w5_diag_t`, Ж-251): бутстрап-SE разностей X1−K2, X1−K3, ZB−X1, ZB0−X1 по окну, половинам и годам — калибровка защитных условий W5."""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s16 import x1_w5_diag_abc as ABC
from models_v2.s16 import x1_w5_diag_d as DD
from models_v2.s16 import x1_w5_diag_t as TT
from models_v2.s16 import x1_w5_diag_v as VV

ROOT = C.ROOT
OUT = ROOT / "docs/otchety/harj_x1/W5_DIAG.csv"
OUT0 = ROOT / "docs/otchety/harj_x1/W5_TABLE0.csv"
CAL = ROOT / "data/calendar"
MINPAST = ABC.MINPAST
SRC_EXTRA = ["data/regen/harj_x1/Z/zb.parquet", "data/fwd/models/new/zb.parquet",
             "data/calendar/cbr.csv", "data/calendar/cpi_us.csv", "data/calendar/fomc.csv", "data/calendar/expirations.csv",
             "data/calendar/imoex_rebal_official.csv", "data/fwd/models/calendar.parquet"]


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]


def _cal(name):
    c = pd.read_csv(CAL / name)
    c["date"] = pd.to_datetime(c["date"])
    c["known_from"] = pd.to_datetime(c["known_from"])
    c["known_to"] = pd.to_datetime(c["known_to"]) if "known_to" in c else pd.NaT
    return c


def _known(c, d):
    """Версии событий, известные на закрытии d."""
    ok = (c["known_from"] <= d) & (c["known_to"].isna() | (c["known_to"] > d))
    return c.loc[ok, "date"]


def events(x):
    d, tp = x["d"].to_numpy(), x["t_plus"].to_numpy()
    cbr, cpi, fomc, exp, reb = (_cal(n) for n in ("cbr.csv", "cpi_us.csv", "fomc.csv", "expirations.csv", "imoex_rebal_official.csv"))
    cal = pd.read_parquet(ROOT / "data/fwd/models/calendar.parquet")
    tdays = pd.to_datetime(cal["date"]).sort_values().to_numpy()
    out = {k: np.zeros(len(x), bool) for k in ("ev_cbr", "ev_cpius", "ev_fomc", "ev_exp", "ev_rebal")}
    for i in range(len(x)):
        di, ti = pd.Timestamp(d[i]), pd.Timestamp(tp[i])
        out["ev_cbr"][i] = (_known(cbr, di) == ti).any()
        out["ev_cpius"][i] = (_known(cpi, di) == ti).any()
        f = _known(fomc, di)
        out["ev_fomc"][i] = ((f >= di) & (f < ti)).any()
        out["ev_exp"][i] = (_known(exp, di) == ti).any()
        r = _known(reb, di)
        # t⁺ — последний торговый день перед вступлением: следующий торговый день после t⁺ ≥ дата вступления > t⁺
        k = np.searchsorted(tdays, np.datetime64(ti), side="right")
        nxt = pd.Timestamp(tdays[k]) if k < len(tdays) else ti + pd.Timedelta(days=3)
        out["ev_rebal"][i] = ((r > ti) & (r <= nxt)).any()
    out["ev_any"] = np.any(np.column_stack(list(out.values())), axis=1)
    return out


def terc_past(v, minpast=MINPAST):
    """Терциль дня i по порогам из значений строго до i (PIT); na — меньше minpast прошлых."""
    v = np.asarray(v, float)
    g = np.full(len(v), "na", object)
    for i in range(minpast, len(v)):
        q1, q2 = np.quantile(v[:i], [1 / 3, 2 / 3])
        g[i] = "T1" if v[i] <= q1 else ("T2" if v[i] <= q2 else "T3")
    return g


def build():  # xtask: длинная — перенос diag.build без изменения логики
    x = C.load()
    # RV дня d и прошлых дней — из пар K23 (t⁺ = день); архив + продолжение
    ka = pd.read_parquet(ROOT / C.SRC["k23_a"]).rename(columns={"date": "d"})
    kn = pd.read_parquet(ROOT / C.SRC["k23_n"]).rename(columns={"date": "d"})
    for k in (ka, kn):
        k["t_plus"] = pd.to_datetime(k["t_plus"])
    k = pd.concat([ka[ka["t_plus"] <= C.ARCH_END], kn], ignore_index=True).sort_values("t_plus")
    rvmap = pd.Series(k["rv"].to_numpy(), index=k["t_plus"].to_numpy())
    rv_d = rvmap.reindex(x["d"].to_numpy()).to_numpy()
    miss = x.loc[np.isnan(rv_d), "d"].dt.date.tolist()
    print("нет пары с t⁺ = d (RV дня d в таблице пар нет) — группа na:", miss)
    tps = rvmap.index.to_numpy()
    v = np.empty(len(x))
    for i, di in enumerate(x["d"].to_numpy()):
        j = np.searchsorted(tps, di)  # позиция дня d; прошлые 20 — j−20…j−1
        v[i] = np.log(rv_d[i]) - np.log(rvmap.to_numpy()[j - 20:j].mean())
    # последняя закрытая ошибка X1: пара с t⁺ = d
    f_by_tp = pd.Series(x["F"].to_numpy(), index=x["t_plus"].to_numpy())
    err = np.log(rv_d / f_by_tp.reindex(x["d"].to_numpy()).to_numpy())  # 2018-01-03: na
    zb = pd.concat([pd.read_parquet(ROOT / p) for p in SRC_EXTRA[:2]], ignore_index=True)
    zb["d"] = pd.to_datetime(zb["d"])
    x = x.merge(zb[["d", "F"]].rename(columns={"F": "ZB"}), on="d", how="left")
    st = {}
    st.update(events(x))
    st["dow"] = x["t_plus"].dt.dayofweek.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}).to_numpy(object)
    st["gap"] = ((x["t_plus"] - x["d"]).dt.days - 1 > 0).to_numpy()
    st["e"] = x["s"].to_numpy(int) == 1
    lcm = np.log(x["C"] / x["M"]).to_numpy()
    st["sgnCM"] = np.where(lcm > 0, "C>M", "C<=M").astype(object)
    st["tCM"] = terc_past(lcm)
    st["absCM"] = terc_past(np.abs(lcm))
    st["tK1M"] = terc_past(np.log(x["K1"] / x["M"]).to_numpy())
    st["absK23"] = terc_past(np.abs(np.log(x["K2"] / x["K3"])).to_numpy())
    st["lvl"] = terc_past(np.log(x["M"]).to_numpy())
    gv = terc_past(np.where(np.isnan(v), 0.0, v))
    gv[np.isnan(v)] = "na"
    st["v"] = gv
    e2 = np.where(np.isnan(err), 0.0, err)
    g = terc_past(e2)
    g[np.isnan(err)] = "na"
    st["err"] = g
    st["pi"] = terc_past(x["pi"].to_numpy())
    for kk in list(st):
        if st[kk].dtype == bool:
            st[kk] = np.where(st[kk], "1", "0").astype(object)
    x["yr"] = x["t_plus"].dt.year.astype(str)
    x["half"] = np.where(x["t_plus"].dt.year <= 2021, "h1", "h2")
    for kk in ("X1", "M", "C", "K1", "K2", "K3", "ZB"):
        x["l_" + kk] = C.qlike(x["rv"], x["F" if kk == "X1" else kk])
    return x, st


def compute(x, st):
    """Блоки A–E. Блок E идёт через текст CSV (как в оригинале: `noise.csv` → `read_csv`), чтобы строки совпадали побитно."""
    rows = ABC.block_a(x, st) + ABC.block_b(x, st) + ABC.block_c(x, st) + DD.rows_d(x, st)
    _, s = TT.series(x)
    nz = pd.read_csv(io.StringIO(TT.noise(x, s).to_csv(index=False, lineterminator="\n")))
    return pd.concat([pd.DataFrame(rows), nz], ignore_index=True)


def diag_lines(df):
    return df.to_csv(index=False, lineterminator="\n").splitlines()


def _data_lines(path):
    return [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]


def _file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(diag=OUT, table0=OUT0):
    """Только чтение: пересчёт и побитная сверка строк данных W5_DIAG.csv и W5_TABLE0.csv; sha файлов до и после равны. -> список замечаний."""
    before = (_file_sha(diag), _file_sha(table0))
    x, st = build()
    bad = []
    if diag_lines(compute(x, st)) != _data_lines(diag):
        bad.append("строки данных %s не равны пересчёту" % Path(diag).name)
    if TT.table(*TT.series(x)).to_csv(index=False, lineterminator="\n").splitlines() != _data_lines(table0):
        bad.append("строки данных %s не равны пересчёту" % Path(table0).name)
    if before != (_file_sha(diag), _file_sha(table0)):
        bad.append("--check изменил файл")
    return bad


def main():
    x, st = build()
    df = compute(x, st)
    head = ["# W5_DIAG — диагностика перед выбором гипотезы W5 (Ж-250 пп. 3–4); DEVELOPMENT / HISTORICAL WALK-FORWARD, после просмотра окна M; не отбор",
            "# окно M n=%d, t+ %s..%s; генератор models_v2.s16.x1_w5_diag; объявление — docstring" % (len(x), x["t_plus"].min().date(), x["t_plus"].max().date()),
            "# sha256[:16] входов: " + "; ".join("%s=%s" % (k, sha(v)) for k, v in C.SRC.items()) + "; " + "; ".join("%s=%s" % (Path(p).name, sha(p)) for p in SRC_EXTRA)]
    if "--write" in sys.argv:
        OUT.write_text("\n".join(head) + "\n" + df.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
        t0 = TT.table(*TT.series(x))
        OUT0.write_text("\n".join(TT.head_lines()) + "\n" + t0.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
        print("wrote", OUT, len(df), "rows;", OUT0, len(t0), "rows")
    else:
        print("dry run", len(df), "rows (запись — --write)")


if __name__ == "__main__":
    if "--check-verdict" in sys.argv:
        sys.exit(VV.main(OUT))
    if "--check" in sys.argv:
        b = check()
        print("check %s" % ("ok: строки данных W5_DIAG.csv и W5_TABLE0.csv равны пересчёту; sha файлов до и после равны" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    main()
