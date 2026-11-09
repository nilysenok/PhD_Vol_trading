"""V7-4a: конфигурация 10 — совместное обучение (Stage B) для конфигурации 6 (MoE, маршрутизатор B, K = 2, энкодер TCN-w5) по календарю владельца (Ж-235; план, редакция 3, раздел 4):
 • раз в год, 1-й торговый день января Y = 2019…2026 — совместное переобучение энкодера, экспертов и маршрутизатора (`v7_1m_joint.PCAEncoder` + `LS.Joint`, штраф λ_e‖φ − φ₀‖², φ₀ = веса Stage A года Y того же seed, λ_e = `LS.REG`),
   пары t⁺ < 01.01.Y (граница `cut(Y)` = 31.12.Y−1, закрытые пары `WF.closed_rows`), бары < 01.01.Y; решётка λ_g × λ_LB, ранняя остановка по последним 126 закрытым парам — как у 6 (`WF._refit`);
 • внутри года — помесячно (1-й торговый день месяца) только эксперты и маршрутизатор (спецификация 6, голова с нуля), энкодер года ЗАМОРОЖЕН: z всех обучающих пар и прогнозов блока — от одного энкодера года Y
   (главные компоненты — по z ЭТОГО энкодера на датах < 01.01.Y: `REP.fit_pca_past`), пары t⁺ ≤ d; в самом январском блоке прогноз — совместная модель (её главные компоненты — Stage A, замороженные буферы `PCAEncoder`);
 • 2018 — прогноз Stage A (конфигурация 6, копия строк) с пометкой `stageA_2018`; 5 seeds, у каждого свой энкодер Stage A и своя цепочка (seed корректора = seed энкодера); прогноз = среднее ln F.
Данные читаются только при вызове функций; ничего не пишется (запись — `v7_4_run`)."""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import torch

from models_v2.s13 import v6_1_wf as W6
from models_v2.s14 import v7_1b_cfg as CFG1, v7_1b_sta as STA, v7_1e_net as NET, v7_1e_pre as PRE, v7_1e_rep as REP, v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_joint as JT, v7_1m_wf as WF, v7_2_cfg as C2

JOINT_MONTHS = (1,)                                   # календарь Ж-235: совместное переобучение — только в январе
YEARS_B = tuple(range(2019, 2027))
FIRST_B = pd.Timestamp("2019-01-01")
SPEC10 = dict(C2.SPECS["6"])                          # MoE, маршрутизатор B, K = 2 — спецификация 6 (энкодер добавляется в `fit_joint`)
GRID10 = C2.GRIDS["6"]
PARAMS_HEAD = 67 + 10


def declared_problems() -> list:
    """Нарушения объявленного: календарь только январь, 2019…2026, голова и решётка равны 6. МУТАНТЫ подменяют JOINT_MONTHS, YEARS_B, SPEC10, GRID10."""
    out = []
    if JOINT_MONTHS != (1,) or YEARS_B != tuple(range(2019, 2027)):
        out.append("календарь: совместное переобучение не только в январе 2019…2026")
    if SPEC10 != C2.SPECS["6"] or GRID10 != C2.GRIDS["6"] or FM.n_params(FM.ResMoE("B", 2)) != PARAMS_HEAD:
        out.append("голова или решётка 10 не равны конфигурации 6")
    return out


def cut(year: int) -> pd.Timestamp:
    """Граница совместного обучения года Y: закрытые пары t⁺ ≤ 31.12.(Y−1), то есть t⁺ < 01.01.Y. МУТАНТ «граница в январе» подменяет эту функцию."""
    return pd.Timestamp(year - 1, 12, 31)


def plan(index: pd.DatetimeIndex) -> list:
    """Блоки месяцев: (a, b, d0, year, kind), kind = stageA (d0 < 2019) | joint (1-й торговый день января) | head (остальные месяцы)."""
    pos = W6.refit_positions(index, WF.FIRST_REFIT)
    out = []
    for a, b in zip(pos, pos[1:] + [len(index)]):
        d0 = index[a]
        kind = "stageA" if d0 < FIRST_B else "joint" if d0.month in JOINT_MONTHS else "head"
        out.append(dict(a=a, b=b, d0=d0, year=d0.year, kind=kind))
    return out


def base_from(art: dict) -> torch.nn.Module:
    """TCN из весов артефакта (float64, точное преобразование float32 → float64)."""
    m = NET.build("tcn").double()
    m.load_state_dict({k[3:]: torch.as_tensor(v) for k, v in art.items() if k.startswith("w__")})
    return m


def factory(art: dict):
    """Фабрика энкодера Stage B для `FT.make`: свежая копия весов Stage A года Y и ЗАМОРОЖЕННЫХ главных компонент Stage A (`PCAEncoder`)."""
    return lambda: JT.PCAEncoder(base_from(art), {k: art["pca_" + k] for k in ("mu", "sd", "comps", "ssd")})


@contextlib.contextmanager
def capture():
    """Запоминает (модели, батч) последнего `ensemble_lnF` — прогноз блока; прежняя функция ВОССТАНАВЛИВАЕТСЯ (класс № 43/52)."""
    last, orig = [], FT.ensemble_lnF

    def ens(models, bt):
        last[:] = [(models, bt)]
        return orig(models, bt)
    FT.ensemble_lnF = ens
    try:
        yield last
    finally:
        FT.ensemble_lnF = orig


def fit_joint(cols, Zg, seq, art: dict, year: int, bl: dict, seed: int, lam_e: float | None = None) -> tuple:
    """Январское совместное переобучение (один seed): -> (результат блока {G, lam, val, epochs, pi, gk}, обученный TCN). seq (n, 12, L) — последовательности всех дат (NaN — нет окна)."""
    spec = dict(SPEC10, encoder=factory(art), **({} if lam_e is None else dict(lam_e=lam_e)))
    with capture() as last:
        res, _ = WF._refit(cols, Zg, seq, spec, cut(year), bl["a"], bl["b"], (seed,), GRID10)
    if res is None:
        raise ValueError("совместное переобучение %d: мало закрытых пар" % year)
    models, bt = last[0]
    with torch.no_grad(), FM.deterministic():
        pi = torch.stack([m(bt["Zg"], bt["z"], bt["ell"])[1] for m in models]).mean(dim=0).numpy()
        gk = torch.stack([m.head.experts(m.encoder(bt["z"])) for m in models]).mean(dim=0).numpy()
    return dict(res, pi=pi, gk=gk), models[0].encoder.base


def identity_encoder(base) -> torch.nn.Module:
    """Тот же `PCAEncoder`, но без сжатия (16 координат: среднее и последнее скрытое состояние) — сырой z обученного энкодера."""
    return JT.PCAEncoder(base, dict(mu=np.zeros(16), sd=np.ones(16), comps=np.eye(16), ssd=np.ones(16))).double()


def z_of_encoder(base, pack: dict, index: pd.DatetimeIndex, year: int) -> tuple:
    """z ОДНОГО обученного энкодера года Y на всех датах до 31.12.Y (согласованные координаты, FX); главные компоненты — по z этого энкодера на датах < 01.01.Y. -> ((n, 4) с NaN вне, pca)."""
    rows = STA.rows_to_year_end(pack, year)
    xm = torch.as_tensor(JT.pack_seq(pack["X"][rows], pack["mask"][rows]))
    with PRE.deterministic(), torch.no_grad():
        raw = identity_encoder(base)(xm).numpy()
    days = np.asarray(pack["days"])[rows]
    p = REP.fit_pca_past(raw, days, year)
    pos, z = index.get_indexer(pd.DatetimeIndex(days)), np.full((len(index), FM.DZ), np.nan)
    z[pos[pos >= 0]] = REP.transform_pca(p, raw)[pos >= 0]
    return z, p


def fit_head(cols, Zg, z, bl: dict, seed: int) -> dict:
    """Помесячная переоценка при замороженном энкодере: голова 6 с нуля на z года, пары t⁺ ≤ d."""
    sink = {}
    with C2.aux_patch(sink):
        res, _ = WF._refit(cols, Zg, z, SPEC10, bl["d0"], bl["a"], bl["b"], (seed,), GRID10)
    if res is None:
        raise ValueError("переоценка %s: мало закрытых пар" % bl["d0"].date())
    return dict(res, pi=sink[bl["a"]]["pi"], gk=sink[bl["a"]]["gk"])


def run_year(tab, Zg, seq, pack, year: int, seed: int, blocks: list, load=STA.load, on_block=None) -> dict:
    """Один год и seed по календарю: январь — совместное, дальше помесячно голова. -> {a: результат блока, 'enc': {год, веса, pca}}. on_block(bl, kind) — для тестов (журнал вызовов)."""
    art, cols, z, out = load(STA.directory() / STA.name("tcn", year, seed)), CFG1.wf_inputs(tab), None, {}
    for bl in blocks:
        if on_block:
            on_block(bl)
        if bl["kind"] == "joint":
            out[bl["a"]], base = fit_joint(cols, Zg, seq, art, year, bl, seed)
            z, pca = z_of_encoder(base, pack, tab.index, year)
            out["enc"] = dict(year=year, seed=seed, state={k: v.detach().numpy() for k, v in base.state_dict().items()}, pca=pca)
        elif z is None:
            raise ValueError("%d: месячная переоценка раньше январского совместного обучения" % year)
        else:
            out[bl["a"]] = fit_head(cols, Zg, z, bl, seed)
    return out


def seq_matrix(pack: dict, index: pd.DatetimeIndex) -> np.ndarray:
    """(n, 12, L) float64: последовательности окна дня d (бары ≤ d) на датах таблицы; даты без пригодного окна (нет бара, нормировка не зрелая) — NaN (в обучение не входят)."""
    rows = np.flatnonzero(pack["norm_ok"] & pack["mask"][:, -1].any(1))
    pos = index.get_indexer(pd.DatetimeIndex(np.asarray(pack["days"])[rows]))
    ok = pos >= 0
    x = JT.pack_seq(pack["X"][rows[ok]], pack["mask"][rows[ok]])
    out = np.full((len(index),) + x.shape[1:], np.nan)
    out[pos[ok]] = x
    return out
