"""V7-DX (Ж-236 п. 4), общее: повтор помесячного walk-forward конфигураций 1, 4, 6 С ПЕРЕХВАТОМ обучающих батчей, метрики «снижение QLIKE к H».
Новых прогнозов нет: повтор обязан воспроизвести сохранённые `1b/cfg1.parquet`, `2/cfg4.parquet`, `2/cfg6.parquet` побитно (`same_as_saved`), иначе диагностика не принимается.
Ничего не пишет (CHECKPOINT `1b`, `2` и прогнозы не трогаются); все вычисления torch — в одном потоке (`FM.deterministic`). Данные читаются при вызове функций."""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s14 import v7_0_io as IO, v7_1b_cfg as CFG, v7_1b_cfg3 as C3, v7_1b_cfgrun as CR, v7_1m_feat as FE, v7_1m_fit as FT, v7_1m_forms as FM, v7_2_cfg as C2, v7_fx_warm as WM

CFGS = ("1", "4", "6")
STATUS = "диагностика, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-236 п. 4, лист DX (конфигурации 1, 4, 6 версии 2; новых конфигураций и прогнозов нет; по 2022+ ничего не выбирается)"
SAVED = {"1": IO.OUT / "1b" / "cfg1.parquet", "4": IO.OUT / "2" / "cfg4.parquet", "6": IO.OUT / "2" / "cfg6.parquet"}
GCOL = {"1": "g", "4": "G", "6": "G"}


def q_red(rv, H, F) -> tuple:
    """-> (QLIKE(H), QLIKE(F), снижение в % QLIKE(H)): 100·(Q(H) − Q(F))/Q(H); плюс = лучше H. Определение QLIKE — `models_v2.metrics.qlike`."""
    qh, qf = float(np.mean(metrics.qlike(rv, H))), float(np.mean(metrics.qlike(rv, F)))
    return qh, qf, 100.0 * (qh - qf) / qh


@contextlib.contextmanager
def capture(sink: list):
    """На время вызова `FT.fit_config` дописывает в sink батчи обучения/валидации, выбранные λ и ln F ансамбля на обучающих парах (tr ∪ va); результат функции не меняется. Прежняя функция ВОССТАНАВЛИВАЕТСЯ (класс № 43/52)."""
    orig = FT.fit_config

    def wrapped(spec, tr, va, seeds=FT.SEEDS, grid=FT.GRID):
        r = orig(spec, tr, va, seeds, grid)
        both = FT.cat(tr, va)
        sink.append(dict(tr=tr, va=va, lam=r["lam"], val=r["val"], H=both["H"].numpy(), rv=both["rv"].numpy(), lnf=FT.ensemble_lnF(r["models"], both).numpy()))
        return r
    FT.fit_config = wrapped
    try:
        yield
    finally:
        FT.fit_config = orig


def inputs() -> dict:
    """tab SBER, прогрев H, Z, тип энкодера (`1b.choice3`)."""
    from models_v2.s14 import v7_1m_data as DATA
    tab = DATA.load_tab()
    return dict(tab=tab, warm=WM.load_warm(), Z=FE.build_Z(tab).to_numpy(float), kind=C3.kind3(CR.read_cp()))


def _seed_run(cfg: str, inp: dict, s: int) -> tuple:
    """-> (G по строкам, fits, sink) одного seed; та же цепочка вызовов, что в `CFG.run_kind` / `C2.walk_seed`."""
    sink = []
    with capture(sink):
        if cfg == "1":
            full = CFG.with_warm(inp["tab"], inp["warm"])
            with CFG.lin5_patch():
                r = CFG.walk(full, np.zeros((len(full), 1)), CFG.z_by_year("tcn", s, full.index), CFG.SPEC, seeds=(s,), grid=CFG.GRID)
            return r["G"].to_numpy(float), r["fits"], sink, full.index
        tab = inp["tab"]
        p = C2.walk_seed(cfg, tab, inp["Z"], CFG.z_by_year(inp["kind"], s, tab.index), s)
        return p["G"], p["fits"], sink, tab.index


def replay(cfg: str, inp: dict) -> dict:
    """Повтор конфигурации cfg ('1', '4', '6') по 5 seeds. -> dict(index, G (5, n), fits[seed], caps[seed] — перехваты по переоценкам с fitted, df — как в parquet)."""
    G, fits, caps, idx = [], [], [], None
    with FM.deterministic():
        for s in FT.SEEDS:
            g, f, sink, idx = _seed_run(cfg, inp, s)
            G.append(g)
            fits.append(f)
            caps.append(sink)
    Gm = np.mean(np.stack(G), axis=0)
    full = CFG.with_warm(inp["tab"], inp["warm"]) if cfg == "1" else inp["tab"]
    df = pd.DataFrame({"d": idx, "H": full["H"].to_numpy(float), "G": Gm})
    df["F"] = df["H"] * np.exp(df["G"])
    return dict(index=idx, G=np.stack(G), fits=fits, caps=caps, df=df[df["F"].notna()].reset_index(drop=True))


def same_as_saved(cfg: str, rep: dict) -> bool:
    """Побитное равенство повтора сохранённому прогнозу (колонка g / G)."""
    sv = pd.read_parquet(SAVED[cfg])
    return bool(np.array_equal(rep["df"]["G"].to_numpy(float), sv[GCOL[cfg]].to_numpy(float)) and np.array_equal(rep["df"]["d"].to_numpy("datetime64[ns]"), sv["d"].to_numpy("datetime64[ns]")))


def refit_table(cfg: str, rep: dict) -> pd.DataFrame:
    """По каждой переоценке с моделью: d0, год, выбранные λ каждого seed, снижение QLIKE к H на обучающих парах ансамбля (tr ∪ va, в выборке)."""
    rows = []
    for k, f in enumerate(rep["fits"][0]):
        if not f["fitted"]:
            continue
        j = sum(1 for g in rep["fits"][0][:k + 1] if g["fitted"]) - 1
        caps = [c[j] for c in rep["caps"]]
        H, rv = caps[0]["H"], caps[0]["rv"]
        F = np.exp(np.mean(np.stack([c["lnf"] for c in caps]), axis=0))
        qh, qf, red = q_red(rv, H, F)
        rows.append(dict(cfg=cfg, d0=f["d0"], year=f["d0"].year, n_pairs=len(H), q_H=qh, q_F=qf, red_in_pct=red, lam_g=[c["lam"][0] for c in caps], lam_lb=[c["lam"][1] for c in caps]))
    return pd.DataFrame(rows)
