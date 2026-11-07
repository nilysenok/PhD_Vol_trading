"""P3-17, G17: проверки выхода стадии `s17` (`--check`; код возврата 0 — гейт закрыт). Каждая часть возвращает список ошибок (пусто = ок). Результаты на 28 — тот же SHA, что до листа."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_hyb_out as O, hp_p17_out as OUT28, hp_paths as P, hp_s17_sel as SEL, hp_stats_sel as S

LOTO28_SHA = "1e57cb885ebc66eb"            # loto.parquet на 28 (G12)
STATS28 = ("SELECTION", "COMPARISON", "TESTS", "MCS", "PER_TICKER", "HARJ_VALUE", "LEVELS", "LOTO")


def csv17(name: str) -> pd.DataFrame:
    return pd.read_csv(SEL.D17 / (name + ".csv"), comment="#")


def check_28() -> list:
    bad = OUT28.check_28_unchanged()
    p = P.OUT / "loto.parquet"
    bad += [] if p.exists() and P.sha16(p) == LOTO28_SHA else ["loto.parquet (28): SHA изменился"]
    return bad + SEL.check_inputs()


def check_sel() -> list:
    bad = [] if (SEL.D17 / "SELECTION.csv").exists() and (SEL.D17 / "SELECTION.json").exists() else ["нет statistics_p17/SELECTION.csv / .json"]
    if bad:
        return bad
    sel, cp = json.loads((SEL.D17 / "SELECTION.json").read_text()), json.loads((P.REP / "CHECKPOINT.json").read_text())
    sc = csv17("SELECTION")
    bad += [] if len(sc) == 25 and sc["crit_panel"].notna().all() else ["SELECTION: ожидалось 25 рядов с crit_panel"]
    n = SEL.N_PRIOR + SEL.N_P17 + len(sel["loto"])
    bad += [] if cp.get("N_declared_before_stats_p17") == n and sel["N_declared"] == n else ["N_declared_before_stats_p17 %s != 608 + 20 + %d" % (cp.get("N_declared_before_stats_p17"), len(sel["loto"]))]
    bad += [] if len(sel["loto"]) <= 5 else ["LOTO больше 5 прогонов"]
    bad += ["хэш входа %s изменился после выбора" % f for f, h in sel["hashes"].items() if P.sha16(P.OUT / f) != h]
    sc2 = S.crit_panel(SEL.load17(), S.model_cols()).set_index("series")["crit_panel"]
    bad += [] if (sc2 - sc.set_index("series")["crit_panel"]).abs().max() < 1e-9 else ["SELECTION.csv не воспроизводится из входов"]
    print("SELECTION: финалисты %s; тройка %s; У-16: %s; LOTO %s; N=%s" % (sel["finalists"], sel["triple_cfgs"], sel["u16"], sel["loto"], cp.get("N_declared_before_stats_p17")))
    return bad


def check_loto() -> list:
    sel = json.loads((SEL.D17 / "SELECTION.json").read_text())
    bad, df = [], []
    for m in ("xgb", "lgbm"):
        p = P.OUT / ("loto_p17_%s.parquet" % m)
        if any(c.startswith(m + "_") for c in sel["loto"]):
            if not p.exists():
                bad.append("нет %s" % p.name)
            else:
                df.append(pd.read_parquet(p))
    if bad:
        return bad
    d = pd.concat(df, ignore_index=True)
    cnt = {c: int(d[d["config"] == c]["ticker"].nunique()) for c in sel["loto"]}
    bad += [] if all(v == 17 for v in cnt.values()) and set(d["ticker"]) <= set(P.MAIN17) else ["LOTO: не 17 бумаг у прогона: %s" % cnt]
    bad += [] if ((d["sigma2"] > 0) & d["sigma2"].notna()).all() else ["LOTO: σ̂² не конечен или ≤ 0"]
    reg = [r for r in O.REG.read_text(encoding="utf-8").splitlines() if r.startswith("Ж-220-p17-LOTO:")]
    bad += [] if len(reg) == len(sel["loto"]) else ["реестр: LOTO-строк %d != %d" % (len(reg), len(sel["loto"]))]
    print("LOTO на 17: прогонов %d, строк %d, бумаг по прогону %s, строк реестра %d" % (len(sel["loto"]), len(d), cnt, len(reg)))
    return bad


def check_loto16() -> list:
    """У-25 (G19): LOTO с поперечными по 16 — 3 прогона × 17 бумаг; реестр 3 + 3 (всего 634); прежние `loto_p17_*` не тронуты (SHA из архива); SELECTION.* побитно прежние; N = 634 объявлено."""
    import hashlib
    sel, cp = json.loads((SEL.D17 / "SELECTION.json").read_text()), json.loads((P.REP / "CHECKPOINT.json").read_text())
    bad, df = [], []
    for m in ("xgb", "lgbm"):
        p = P.OUT / ("loto16_p17_%s.parquet" % m)
        if not any(c.startswith(m + "_") for c in sel["loto"]):
            continue
        df.append(pd.read_parquet(p)) if p.exists() else bad.append("нет %s" % p.name)
    if bad:
        return bad
    d = pd.concat(df, ignore_index=True)
    cnt = {c: int(d[d["config"] == c]["ticker"].nunique()) for c in sel["loto"]}
    bad += [] if all(v == 17 for v in cnt.values()) and set(d["ticker"]) <= set(P.MAIN17) and set(d["config"]) == set(sel["loto"]) else ["LOTO16: не 3 прогона × 17 бумаг: %s" % cnt]
    bad += [] if ((d["sigma2"] > 0) & d["sigma2"].notna()).all() else ["LOTO16: σ̂² не конечен или ≤ 0"]
    reg = pd.read_csv(O.REG)["key"].astype(str)
    n16, n17 = int(reg.str.startswith("Ж-220-p17-LOTO16:").sum()), int(reg.str.startswith("Ж-220-p17-LOTO:").sum())
    bad += [] if (n16, n17, len(reg)) == (3, 3, 634) else ["реестр: LOTO16 %d (3), прежний LOTO %d (3), всего %d (634)" % (n16, n17, len(reg))]
    bad += [] if cp.get("N_declared_before_stats_p17_loto16") == 634 else ["CHECKPOINT: N_declared_before_stats_p17_loto16 != 634"]
    arc = pd.read_csv(P.REP / "statistics_p17/archive_pre_loto16/HASHES.csv").set_index("file")["sha256"]
    for f in ("loto_p17_xgb.parquet", "loto_p17_lgbm.parquet"):
        bad += [] if hashlib.sha256((P.OUT / f).read_bytes()).hexdigest() == arc["data/regen/harj_panel_hybrid/" + f] else ["%s: прежний LOTO изменён" % f]
    for f, h in cp["selection_sha256_before_loto16"].items():
        bad += [] if hashlib.sha256((SEL.D17 / f).read_bytes()).hexdigest() == h else ["%s: SELECTION изменился после объявления N" % f]
    print("LOTO16: прогонов %d, строк %d, бумаг по прогону %s; реестр LOTO16 %d + прежний %d, всего %d; N=%s" % (len(sel["loto"]), len(d), cnt, n16, n17, len(reg), cp.get("N_declared_before_stats_p17_loto16")))
    return bad


def check_holm634() -> list:
    """N = 634 во всех семействах TESTS.csv: p Холма main / transfer пересчитывается из dm_p с N = 634 (с N = числом строк или 631 не совпадёт)."""
    from models_v2.s5.d import stats_d as SD
    t, bad = csv17("TESTS"), []
    for kind in ("main", "transfer"):
        m = t[(t["kind"] == kind) & t["dm_p"].notna()]
        bad += [] if len(m) and np.allclose(m["p_holm"], SD.holm_n(m["dm_p"].to_numpy(), 634), rtol=1e-9, atol=0) else ["TESTS %s: p Холма не равен пересчёту с N = 634" % kind]
    return bad


def check_tables() -> list:
    bad = [] if all((SEL.D17 / (f + ".csv")).exists() for f in STATS28) else ["нет statistics_p17/*.csv"]
    if bad:
        return bad
    cp = json.loads((P.REP / "CHECKPOINT.json").read_text())
    t, pt = csv17("TESTS"), csv17("PER_TICKER")
    for kind in ("main", "transfer"):
        m = t[t["kind"] == kind]
        bad += [] if len(m) > 0 and m["p_holm"].notna().all() and (m["p_holm"] >= m["dm_p"] - 1e-12).all() else ["Холм: %s — p_holm не заполнен или меньше p" % kind]
    bad += [] if not t["scope"].astype(str).str.contains("all28").any() else ["TESTS: область all28 (смешивание 17 и E11)"]
    a, b = set(pt[pt["population"] == "17"]["ticker"]), set(pt[pt["population"] == "E11"]["ticker"])
    bad += [] if a == set(P.MAIN17) and b == set(P.E11) and not (a & b) else ["PER_TICKER: совокупности смешаны или неполны (17: %d, E11: %d)" % (len(a), len(b))]
    lv, mc = csv17("LEVELS"), csv17("MCS")
    bad += [] if len(lv) == 40 else ["LEVELS: строк %d != 40" % len(lv)]
    bad += [] if mc["in_mcs90"].any() else ["MCS: пустое множество"]
    nk = t[t["window"].astype(str).str.startswith("k23_")]["n_k23_tickers"]
    bad += [] if len(nk) and nk.notna().all() else ["TESTS: нет числа бумаг с K2/K3"]
    print("N=%s; TESTS %d строк (main %d, transfer %d); LEVELS %d (макс. уровень %d); MCS %d; бумаг с K2/K3 в TESTS: %s" % (
        cp.get("N_declared_before_stats_p17_loto16"), len(t), int((t["kind"] == "main").sum()), int((t["kind"] == "transfer").sum()), len(lv), lv["level"].max(), len(mc), sorted(set(nk.dropna().astype(int)))))
    return bad


def check(part: str) -> list:
    steps = {"sel": [check_28, check_sel], "loto": [check_28, check_sel, check_loto], "loto16": [check_28, check_sel, check_loto16],
             "stats": [check_28, check_sel, check_loto, check_loto16, check_tables, check_holm634], "e11": [check_28, check_sel, check_tables, check_holm634]}
    todo = steps["stats"] if part in ("all", "stats") else steps[part]
    return [e for f in todo for e in f()]
