"""V7 лист 4a, G45: конфигурация 10 (совместное обучение Stage B для 6) через РЕАЛЬНЫЙ путь `run_year` на синтетике (энкодер TCN на случайных «барах», артефакт Stage A собран тем же кодом, что `v7_1b_sta.compute`).
Календарь Ж-235: совместное переобучение — только в январе, внутри года помесячно голова при замороженном энкодере; пары t⁺ < 01.01.Y; штраф λ_e — к весам Stage A; z всех строк — от одного энкодера года (класс № 146, FX); 2018 = прогноз 6 с пометкой.
Мутации: совместное переобучение не в январе, граница в январе, энкодер разморожен (переобучается каждый месяц), окно последовательностей со сдвигом, z строк прошлых лет от другого энкодера, штраф к нулю, 2018 пересчитан. Файлы SBER не читаются."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s14 import v7_0_io as IO, v7_1b_sta as STA, v7_1e_net as NET, v7_1e_rep as REP, v7_1m_fit as FT, v7_1m_joint as JT, v7_1m_loss as LS, v7_1m_syn as SY, v7_2_cfg as C2, v7_4_cfg as C4, v7_4_chk as CK, v7_4_run as R4

S = SY.make(800, 0)
TAB, ZG = S["tab"], S["Zg"]
D = len(TAB)
G0 = np.random.default_rng(1)
PACK = dict(X=G0.normal(size=(D, 1, 53, 11)).astype(np.float32), mask=np.ones((D, 1, 53), bool), norm_ok=np.ones(D, bool), days=TAB.index.to_numpy("datetime64[ns]"), r2=np.abs(G0.normal(size=(D, 53))) * 1e-4)
YEAR = 2019
SEQ = C4.seq_matrix(PACK, TAB.index)
BLOCKS = [b for b in C4.plan(TAB.index) if b["year"] == YEAR]


def make_art(year=YEAR, seed=0) -> dict:
    torch.manual_seed(seed)
    base, rows = NET.build("tcn"), STA.rows_to_year_end(PACK, year)
    raw = REP.encode(base, PACK["X"][rows], PACK["mask"][rows])
    p = REP.fit_pca_past(raw, PACK["days"][rows], year)
    a = {"w__" + k: v.detach().numpy() for k, v in base.state_dict().items()}
    a.update(z=REP.transform_pca(p, raw), days=PACK["days"][rows].astype(np.int64), rows=rows, meta=np.array([1e-6, 0, 0, 0, 0, 905.0]))
    a.update({"pca_" + k: np.asarray(p[k], float) for k in ("mu", "sd", "comps", "ssd")})
    return a


ART = make_art()


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(FT, "MAX_EP", 4)
    monkeypatch.setattr(FT, "PATIENCE", 2)


def run_logged(monkeypatch=None):
    """Реальный `run_year` с журналом: вызовы совместной и месячной переоценки, обучающие строки совместной, z месячных."""
    log = dict(joint=0, head=0, z=[], rows=[], bases=[])
    mp = monkeypatch or pytest.MonkeyPatch()
    fj, fh, fc, zo = C4.fit_joint, C4.fit_head, FT.fit_config, C4.z_of_encoder

    def joint(*a, **k):
        log["joint"] += 1
        return fj(*a, **k)

    def head(cols, Zg, z, bl, seed):
        log["head"] += 1
        log["z"].append(z.copy())
        return fh(cols, Zg, z, bl, seed)

    def fit(spec, tr, va, seeds=FT.SEEDS, grid=FT.GRID):
        if spec.get("encoder") is not None:
            log["rows"].append(np.concatenate([tr["H"].numpy(), va["H"].numpy()]))
        return fc(spec, tr, va, seeds, grid)
    for n, f in (("fit_joint", joint), ("fit_head", head)):
        mp.setattr(C4, n, f)
    mp.setattr(FT, "fit_config", fit)
    try:
        log["out"] = C4.run_year(TAB, ZG, SEQ, PACK, YEAR, 0, [b for b in C4.plan(TAB.index) if b["year"] == YEAR], load=lambda p: ART)       # календарь — по текущему `plan` (мутант JOINT_MONTHS меняет его)
    finally:
        if monkeypatch is None:
            mp.undo()
    log["ref_z"] = zo
    return log


@pytest.fixture(scope="module")
def honest():
    with pytest.MonkeyPatch.context() as mp:                                           # фикстура модуля создаётся раньше функциональной `fast` — те же короткие обучения задаём здесь
        mp.setattr(FT, "MAX_EP", 4)
        mp.setattr(FT, "PATIENCE", 2)
        return run_logged()


def test_plan_calendar_one_joint_per_year_in_january_and_stage_a_before_2019():
    p = C4.plan(TAB.index)
    j = [b for b in p if b["kind"] == "joint"]
    assert C4.declared_problems() == [] and all(b["d0"].month == 1 and b["d0"].year >= 2019 for b in j) and len(j) == len({b["year"] for b in j})
    assert all(b["kind"] == "stageA" for b in p if b["d0"] < C4.FIRST_B) and sum(b["kind"] == "head" for b in p if b["year"] == YEAR) == 11


@pytest.mark.parametrize("months", [(1, 7), tuple(range(1, 13))])
def test_mutation_joint_not_only_in_january_is_caught_by_declared_and_real_path(monkeypatch, months):
    monkeypatch.setattr(C4, "JOINT_MONTHS", months)
    assert C4.declared_problems() and sum(b["kind"] == "joint" for b in C4.plan(TAB.index) if b["year"] == YEAR) == len(months)


def test_run_year_joint_once_in_january_and_head_eleven_times(honest):
    assert honest["joint"] == 1 and honest["head"] == 11 and set(honest["out"]) == {b["a"] for b in BLOCKS} | {"enc"}
    assert all(np.isfinite(honest["out"][b["a"]]["G"]).all() and len(honest["out"][b["a"]]["G"]) == b["b"] - b["a"] for b in BLOCKS)


def test_mutation_encoder_unfrozen_inside_year_is_caught(monkeypatch):
    monkeypatch.setattr(C4, "JOINT_MONTHS", tuple(range(1, 13)))                       # мутант: энкодер переобучается каждый месяц
    log = run_logged(monkeypatch)
    assert log["joint"] == 12 and log["head"] == 0                                    # честный путь: 1 и 11 (тест выше)


def test_months_share_one_frozen_encoder_z_equal_to_january_encoder_output(honest):
    base = C4.base_from({"w__" + k: v for k, v in honest["out"]["enc"]["state"].items()})
    ref, _ = honest["ref_z"](base, PACK, TAB.index, YEAR)
    assert len(honest["z"]) == 11 and all(np.array_equal(z, ref, equal_nan=True) for z in honest["z"])
    assert np.isfinite(ref[TAB.index.year <= YEAR]).all() and np.isnan(ref[TAB.index.year > YEAR]).all()          # до 31.12.Y включительно; позже — NaN


def test_mutation_z_of_past_rows_from_another_encoder_is_caught(honest, monkeypatch):
    """Класс № 146 (FX): z обучающих пар прошлых лет — от другого энкодера, чем z прогнозов блока."""
    zo = C4.z_of_encoder

    def mixed(base, pack, index, year):
        z, p = zo(base, pack, index, year)
        z[index < pd.Timestamp(year, 1, 1)] += 0.5                                     # строки прошлых лет — «другие координаты»
        return z, p
    monkeypatch.setattr(C4, "z_of_encoder", mixed)
    log = run_logged(monkeypatch)
    base = C4.base_from({"w__" + k: v for k, v in log["out"]["enc"]["state"].items()})
    ref, _ = honest["ref_z"](base, PACK, TAB.index, YEAR)
    assert not all(np.array_equal(z, ref, equal_nan=True) for z in log["z"])


def test_joint_training_pairs_are_closed_before_january_and_cover_all_of_them(honest):
    pos = TAB.index.get_indexer([TAB.index[np.flatnonzero(TAB["H"].to_numpy() == h)[0]] for h in honest["rows"][0]])
    tp = TAB["t_plus"].to_numpy()
    assert (tp[pos] < np.datetime64("2019-01-01")).all() and TAB.index[pos].max() < pd.Timestamp(YEAR, 1, 1)
    assert len(pos) == int(((tp < np.datetime64("2019-01-01")) & np.isfinite(ZG).all(axis=1) & np.isfinite(SEQ).all(axis=(1, 2))).sum())             # все закрытые пригодные пары до 01.01.Y, ни одной лишней


def test_mutation_cut_in_january_is_caught(monkeypatch):
    monkeypatch.setattr(C4, "cut", lambda y: pd.Timestamp(y, 1, 31))
    log = run_logged(monkeypatch)
    pos = [np.flatnonzero(TAB["H"].to_numpy() == h)[0] for h in log["rows"][0]]
    assert (TAB["t_plus"].to_numpy()[pos] >= np.datetime64("2019-01-01")).any()


def test_seq_matrix_row_is_bars_of_day_d_and_shifted_window_mutation_is_caught(monkeypatch):
    for i in (5, 300, D - 1):
        assert np.array_equal(SEQ[i], JT.pack_seq(PACK["X"][i:i + 1], PACK["mask"][i:i + 1])[0])
    p2 = dict(PACK, X=np.roll(PACK["X"], -1, axis=0))                                  # мутант: окно дня d + 1
    assert not np.array_equal(C4.seq_matrix(p2, TAB.index)[300], SEQ[300])


def test_penalty_is_to_stage_a_weights_not_zero_and_mutation_is_caught(monkeypatch):
    g = np.random.default_rng(3)
    n = 48
    b = FT.batch(g.normal(size=(n, 9)), SEQ[400:400 + n], g.normal(0, 0.1, n), np.exp(g.normal(size=n)), np.exp(g.normal(size=n)))
    spec = dict(C4.SPEC10, encoder=C4.factory(ART), lam_e=1e4)

    def dist(s):
        m = FT.train_one(s, 0, 0.1, 0.0, b, None, epochs=15)[0]
        return max(float(np.abs(p.detach().numpy() - ART["w__" + k]).max()) for k, p in m.encoder.base.named_parameters())
    m0 = FT.make(spec, 0)
    assert all(np.array_equal(p0.numpy(), ART["w__" + k]) for (k, _), p0 in zip(m0.encoder.base.named_parameters(), m0.phi0)) and float(m0.extra_penalty()) == 0.0
    honest_d = dist(spec)
    assert honest_d < 0.06                                                              # Adam с lr 0,02 колеблется около φ₀; без штрафа к φ₀ дрейф за 15 шагов — порядка 0,3
    orig = LS.Joint.__init__

    def zero_anchor(self, head, encoder, lam_e=LS.REG["lam_e"]):
        orig(self, head, encoder, lam_e)
        self.phi0 = [torch.zeros_like(p) for p in self.phi0]                           # мутант: штраф к нулю
    monkeypatch.setattr(LS.Joint, "__init__", zero_anchor)
    assert dist(spec) > 0.12 > honest_d
    assert LS.REG["lam_e"] == 1e-2 and FT.make(dict(C4.SPEC10, encoder=C4.factory(ART)), 0).lam_e == LS.REG["lam_e"]


def test_declared_head_and_grid_equal_to_config_6():
    assert C4.SPEC10 == C2.SPECS["6"] and C4.GRID10 == C2.GRIDS["6"] and C4.PARAMS_HEAD == 77


def parts_for(seeds=(0, 1), tab=TAB.loc[:"2019-12-31"]):
    g, n = np.random.default_rng(5), len(tab)
    rows = np.flatnonzero(tab.index.year == YEAR)
    pi = lambda: g.dirichlet([5, 5, 5], len(rows))
    return {(s, YEAR): dict(rows=rows, G=g.normal(0, 0.1, len(rows)), pi=pi(), gk=g.normal(0, 0.1, (len(rows), 2))) for s in seeds}, tab


def df6_of(tab):
    g, d = np.random.default_rng(6), tab.loc["2018-01-03":].index
    H, G = tab.loc[d, "H"].to_numpy(), np.random.default_rng(7).normal(0, 0.1, len(d))
    df = pd.DataFrame({"d": d, "t_plus": tab.loc[d, "t_plus"].to_numpy(), "rv": tab.loc[d, "rv_next"].to_numpy(), "H": H, "G": G, "F": H * np.exp(G), "fitted": 0.8})
    for c in ("pi0", "pi1", "pi2"):
        df[c] = 1 / 3
    df["g1"], df["g2"] = g.normal(size=len(d)), g.normal(size=len(d))
    return df


def test_assemble_2018_is_copy_of_6_with_flag_and_later_rows_are_seed_mean(monkeypatch):
    parts, tab = parts_for()
    df6 = df6_of(tab)
    df = R4.assemble(tab, df6, parts, seeds=(0, 1))
    a18, later = df["d"] < pd.Timestamp("2019-01-01"), df["d"] >= pd.Timestamp("2019-01-01")
    assert list(df.columns) == R4.COLS and (df.loc[a18, "stageA_2018"] == 1).all() and (df.loc[later, "stageA_2018"] == 0).all() and a18.sum() > 200
    assert np.array_equal(df.loc[a18, ["G", "F", "pi0", "g1"]].to_numpy(), df6.loc[a18, ["G", "F", "pi0", "g1"]].to_numpy())
    rows = parts[(0, YEAR)]["rows"]
    G = (parts[(0, YEAR)]["G"] + parts[(1, YEAR)]["G"]) / 2
    assert np.allclose(df.loc[later, "G"].to_numpy(), G) and np.array_equal(df.loc[later, "F"].to_numpy(), df.loc[later, "H"].to_numpy() * np.exp(df.loc[later, "G"].to_numpy())) and len(rows) == later.sum()
    monkeypatch.setattr(C4, "FIRST_B", pd.Timestamp("2018-07-01"))                     # мутант: 2018 частично считается Stage B (частей нет) — отказ
    with pytest.raises(ValueError, match="нечисловые"):
        R4.assemble(tab, df6, parts, seeds=(0, 1))


def test_save_year_roundtrip_and_encoder_file_has_weights_and_new_pca(honest, tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "OUT", tmp_path)
    sh = R4.save_year(0, YEAR, honest["out"], 1.5)
    part, enc = STA.load(R4.part_path(0, YEAR)), STA.load(R4.enc_path(YEAR, 0))
    j = BLOCKS[0]
    assert STA.content_hash(part) == sh["part"] and STA.content_hash(enc) == sh["enc"] and part["seconds"][0] == 1.5
    assert np.array_equal(part["rows"], np.arange(BLOCKS[0]["a"], BLOCKS[-1]["b"])) and np.array_equal(part["G"][: j["b"] - j["a"]], honest["out"][j["a"]]["G"])
    assert part["joint"][: j["b"] - j["a"]].all() and not part["joint"][j["b"] - j["a"]:].any()
    assert set(enc) >= {"pca_mu", "pca_sd", "pca_comps", "pca_ssd", "meta"} and any(k.startswith("w__") for k in enc)
    assert not np.array_equal(enc["w__inp.weight"], ART["w__inp.weight"])                # энкодер действительно обучен (отличается от Stage A)


def test_frame_problems_honest_and_mutations_2018_flag_f_equals_h_simplex():
    parts, tab = parts_for()
    df6 = df6_of(tab)
    df = R4.assemble(tab, df6, parts, seeds=(0, 1))
    assert CK.frame_problems(df, df6) == []
    m = df.copy()
    m.loc[0, "F"] *= 1.01                                                              # 2018 не равно прогнозу 6
    assert any("2018" in b for b in CK.frame_problems(m, df6))
    assert any("2018" in b for b in CK.frame_problems(df.assign(stageA_2018=0.0), df6))                        # нет пометки
    m = df.copy()
    m.loc[m.index[-1], "G"] = 0.0                                                      # G = 0, но F ≠ H
    assert any("F ≠ H" in b for b in CK.frame_problems(m, df6))
    m = df.copy()
    m.loc[300, "pi0"] += 0.1
    assert any("симплексе" in b for b in CK.frame_problems(m, df6)) and CK.frame_problems(df.drop(columns="g2"), df6) == ["cfg10: колонки ≠ объявленным"]


def test_recount_joint_and_monthly_bitwise_and_tampered_part_or_encoder_is_caught(honest, tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "OUT", tmp_path)
    R4.save_year(0, YEAR, honest["out"], 1.0)
    run = lambda: CK.recount(TAB, PACK, SEQ, ZG, seed=0, year=YEAR, load=lambda p: ART)
    assert run() == []
    part = STA.load(R4.part_path(0, YEAR))
    enc = STA.load(R4.enc_path(YEAR, 0))
    STA.save(dict(part, G=part["G"] + 1e-9), R4.part_path(0, YEAR))                    # подмена части на 1e-9: побитная сверка видит
    assert sum("не совпала побитно" in b for b in run()) >= 2
    STA.save(part, R4.part_path(0, YEAR))
    STA.save(dict(enc, **{"w__inp.weight": enc["w__inp.weight"] + 1e-9}), R4.enc_path(YEAR, 0))
    assert any("веса энкодера" in b for b in run())
