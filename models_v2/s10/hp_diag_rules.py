"""P3c, G13: правила вывода R15 (а–з) — ОБЪЯВЛЕНЫ ДО РАСЧЁТА; время и SHA текста пишутся в `CHECKPOINT.json` (`diag_rules_declared_at`, `diag_rules_sha`) до запуска стадии `diag`.
Причина — гипотеза «почему гибрид не даёт подтверждённого преимущества»; «подтверждена» значит: числа M16 (обучение на 17) согласуются с гипотезой по правилу ниже.
Оговорка: отчёты P3-17 (QLIKE, DM, LOTO) к моменту записи правил уже просмотрены; пороги выбраны по смыслу величины (знак, 5 % дисперсии, 0,001 QLIKE ≈ 0,4 % уровня), не подгонялись под R15-числа,
которых к тому времени не существовало (γ, ICC, OOS R², λ = 1 — не считались). Окна: вся (цели ≥ 2017-01-03), выбор 2017–2021, подтверждение 2022 — 02.2026. Вывод — по M16; SBER и E11 — строками рядом, без вывода."""
from __future__ import annotations

import hashlib

import numpy as np

T_GAMMA, T_CD, T_CE, T_ICC, T_B, T_LOTO, T_R2, T_RHO, T_LAM, T_GAP, T_GREL = 1.96, 0.5, 0.95, 0.05, -0.001, 0.001, 0.02, 0.6, 0.001, 0.01, 0.25

CAUSES = {
    "a": ("охват: HAR-J не добавляет информации сверх самостоятельного бустинга",
          "γ при δ = ln K1 − ln K2 (K3) в модели σ̂² = K2·exp(a + γδ), QLIKE-оценка, HAC: подтверждена, если панельный t(γ) < 1,96 для K2 и K3 в окнах «вся» и «подтверждение» (M16)"),
    "b": ("сходство с бустингом: поправка гибрида повторяет то, что уже делает K2/K3",
          "corr(ln f_A1 − ln f_K4/K5, ln f_K2/K3 − ln K1) ≥ 0,5 и corr(ошибка D2, ошибка K2/K3) ≥ 0,95 в обоих модулях (M16, вся история)"),
    "c": ("индивидуальные эффекты бумаг существенны",
          "ICC(1) остатка ln RV − ln f_A1 по бумагам ≥ 0,05 и QLIKE(B1) − QLIKE(A1) ≤ −0,001 в обоих модулях (M16, вся история)"),
    "d": ("поправка плохо переносится между бумагами (неоднородность)",
          "QLIKE(LOTO) − QLIKE(пул), среднее по M16, ≥ 0,001 в окне подтверждения у всех трёх LOTO-прогонов (xgb_A1, lgbm_A1, lgbm_D2)"),
    "e": ("ошибка HAR-J слабо предсказуема",
          "R² Минцера–Заркони (corr²) ln RV − ln K1 на прикладную поправку ln f_A1 − ln K1 < 0,02 в обоих модулях и обоих окнах (M16)"),
    "f": ("мало наблюдений: выигрыш растёт с окном обучения",
          "Спирмен ρ(выигрыш QLIKE к K1 по годам цели, среднее число пар обучения года) без 2022; медиана ρ по A1 и D2 обоих модулей ≥ 0,6"),
    "g": ("поправка шумная: сжатие (λ < 1) необходимо",
          "медиана выбранного λ A1 по годам < 1 и QLIKE(A1 при λ = 1, выбранных c и глубине) − QLIKE(A1) ≥ 0,001 (среднее по модулям, M16, вся история)"),
    "h": ("несогласованность рядов (gap-пары, календари) искажает результат",
          "доля gap-пар ≥ 1 % или Δ(LGBM-D2 − K1) без gap отличается от Δ с gap на ≥ 25 % (или меняет знак) (M16, вся история)"),
}


def rules_text() -> str:
    return "\n".join("%s | %s | %s" % (k, s, r) for k, (s, r) in CAUSES.items()) + "\nпороги:%s" % [T_GAMMA, T_CD, T_CE, T_ICC, T_B, T_LOTO, T_R2, T_RHO, T_LAM, T_GAP, T_GREL]


def rules_sha() -> str:
    return hashlib.sha256(rules_text().encode("utf-8")).hexdigest()[:16]


def _ok(x) -> bool:
    return bool(np.all(np.isfinite(x)))


def verdict(c: str, m: dict) -> str:
    """m — метрики M16 (ключи — ниже); нет значения — «не измеримо»."""
    need = {"a": ["t_k2_all", "t_k3_all", "t_k2_conf", "t_k3_conf"], "b": ["cd_xgb", "cd_lgbm", "ce_xgb", "ce_lgbm"], "c": ["icc_xgb", "icc_lgbm", "dB_xgb", "dB_lgbm"],
            "d": ["loto_xgb_A1", "loto_lgbm_A1", "loto_lgbm_D2"], "e": ["r2_xgb_all", "r2_lgbm_all", "r2_xgb_conf", "r2_lgbm_conf"], "f": ["rho"], "g": ["lam_med", "dq_lam1"], "h": ["gap_share", "rel_gap"]}[c]
    if not _ok([m.get(k, np.nan) for k in need]):
        return "не измеримо"
    v = {k: m[k] for k in need}
    if c == "a":
        r = all(x < T_GAMMA for x in v.values())
    elif c == "b":
        r = v["cd_xgb"] >= T_CD and v["cd_lgbm"] >= T_CD and v["ce_xgb"] >= T_CE and v["ce_lgbm"] >= T_CE
    elif c == "c":
        r = v["icc_xgb"] >= T_ICC and v["icc_lgbm"] >= T_ICC and v["dB_xgb"] <= T_B and v["dB_lgbm"] <= T_B
    elif c == "d":
        r = all(x >= T_LOTO for x in v.values())
    elif c == "e":
        r = all(x < T_R2 for x in v.values())
    elif c == "f":
        r = v["rho"] >= T_RHO
    elif c == "g":
        r = v["lam_med"] < 1 and v["dq_lam1"] >= T_LAM
    else:
        r = v["gap_share"] >= T_GAP or v["rel_gap"] >= T_GREL
    return "подтверждена" if r else "не подтверждена"
