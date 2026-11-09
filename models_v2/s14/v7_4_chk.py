"""V7-4a, G45–G46: проверка конфигурации 10 (`run_v7 --stage 4 --check`). НИЧЕГО НЕ ПИШЕТ: sha256 прогноза, частей и весов энкодеров Stage B по CHECKPOINT `4.cfg10`, F > 0, G = 0 ⇒ F = H, π в симплексе, даты как у конфигурации 1,
2018 = строки конфигурации 6 с пометкой `stageA_2018`, среднее частей по seeds = parquet; ПЕРЕСЧЁТ одной годовой совместной переоценки (seed 0, 2019: G, π, g_k блока и веса энкодера побитно) и одной месячной (первый блок не раньше 2019-07-01
при замороженном энкодере из `4/encB`) — побитно; календарь (`plan`), порядок записи (cfg10 → eval → cond11 → cfg11), отказ 11 при ложном условии. `kw` (load, пути) — только для тестов на синтетике."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_cfg as CFG1, v7_1b_cfgrun as CR, v7_1b_sta as STA, v7_1m_reg as RG, v7_2_run as R2, v7_4_cfg as C4, v7_4_run as R4


def order_problems(cp: dict) -> list:
    """Порядок записи: `4.cond11` — ДО любого счёта 11; 11 допускается только при истинном `4.cond11` (и `cond_11` не стоит при ложном). Время — `written_at` (часы машины)."""
    cp4 = cp.get("4", {})
    out = []
    if cp.get("cond_11") is True and not cp4.get("cond11", {}).get("cond11"):
        out.append("cond_11 = true при ложном или не записанном 4.cond11")
    c11, c = cp4.get("cond11"), cp4.get("cfg11")
    if c is not None and c11 is None:
        out.append("4.cfg11 записан без 4.cond11")
    if c is not None and c11 is not None and not (c11["written_at"] <= c["written_at"]):
        out.append("4.cond11 записан после счёта 11")
    if c is not None and c11 is not None and not c11["cond11"]:
        out.append("конфигурация 11 посчитана при ложном 4.cond11")
    return out


def _block(part: dict, bl: dict) -> np.ndarray:
    return np.flatnonzero((part["rows"] >= bl["a"]) & (part["rows"] < bl["b"]))


def recount(tab, pack, seq, Zg, seed: int = 0, year: int = 2019, load=STA.load, enc_load=STA.load) -> list:
    """Одна годовая совместная и одна месячная переоценка побитно (seed, год)."""
    blocks, cols = [b for b in C4.plan(tab.index) if b["year"] == year], CFG1.wf_inputs(tab)
    part, bad = STA.load(R4.part_path(seed, year)), []
    enc = enc_load(R4.enc_path(year, seed))
    j = blocks[0]
    res, base = C4.fit_joint(cols, Zg, seq, load(STA.directory() / STA.name("tcn", year, seed)), year, j, seed)
    i = _block(part, j)
    for k, c in (("G", "G"), ("pi", "pi"), ("gk", "gk")):
        if not np.array_equal(res[k], part[c][i]):
            bad.append("cfg10: совместная переоценка %d seed %d (%s) не совпала побитно с частью" % (year, seed, k))
    if any(not np.array_equal(v.detach().numpy(), enc["w__" + k]) for k, v in base.state_dict().items()):
        bad.append("cfg10: веса энкодера %d seed %d не совпали побитно с 4/encB" % (year, seed))
    m = next(b for b in blocks if b["kind"] == "head" and b["d0"] >= CR.CHECK_FROM)
    z, _ = C4.z_of_encoder(C4.base_from(enc), pack, tab.index, year)                       # замороженный энкодер года — из сохранённых весов
    got, i = C4.fit_head(cols, Zg, z, m, seed), _block(part, m)
    for k, c in (("G", "G"), ("pi", "pi"), ("gk", "gk")):
        if not np.array_equal(got[k], part[c][i]):
            bad.append("cfg10: месячная переоценка %s seed %d (%s) не совпала побитно с частью" % (m["d0"].date(), seed, k))
    return bad


def frame_problems(df: pd.DataFrame, df6: pd.DataFrame) -> list:
    """Колонки, F > 0, G = 0 ⇒ F = H, π в симплексе, 2018 — строки конфигурации 6 побитно с пометкой `stageA_2018`, остальные — без пометки."""
    bad = [] if list(df.columns) == R4.COLS else ["cfg10: колонки ≠ объявленным"]
    if bad:
        return bad
    if not ((df["F"] > 0).all() and np.array_equal(df["F"][df["G"] == 0], df["H"][df["G"] == 0])):
        bad.append("cfg10: F ≤ 0 или G = 0, но F ≠ H")
    if not np.allclose(df.loc[df["fitted"] > 0, ["pi0", "pi1", "pi2"]].sum(axis=1), 1.0, rtol=0, atol=1e-12):
        bad.append("cfg10: π не в симплексе")
    a18 = np.asarray(pd.DatetimeIndex(df["d"]) < C4.FIRST_B)
    same = df.loc[a18, R4.COLS[:-1]].reset_index(drop=True).equals(df6.loc[df6["d"].isin(df.loc[a18, "d"]), R4.COLS[:-1]].reset_index(drop=True))
    if not (np.array_equal(df["stageA_2018"].to_numpy(float), a18.astype(float)) and same):
        bad.append("cfg10: 2018 не равно прогнозу 6 или нет пометки stageA_2018")
    return bad


def check(cp: dict, inp=None) -> list:
    """-> замечания (пусто = ок). Читает файлы, ничего не пишет."""
    cp4, bad = cp.get("4", {}), C4.declared_problems()
    rec, p = cp4.get("cfg10"), R4.path()
    if rec is None or not p.exists():
        return bad + ["конфигурация 10: нет в CHECKPOINT или нет файла"]
    df, df6 = pd.read_parquet(p), pd.read_parquet(R4.CFG6)
    if IO.sha_file(p) != rec["sha256_file"] or CR.content_hash(df) != rec["sha256_content"]:
        bad.append("cfg10.parquet: sha256 ≠ CHECKPOINT")
    bad += frame_problems(df, df6) + ([] if R2.same_dates(df) else ["cfg10: даты ≠ датам конфигурации 1"])
    tab, pack, seq, Zg = inp or R4.inputs()
    parts = R4.load_parts()
    if CR.content_hash(R4.assemble(tab, df6, parts)) != rec["sha256_content"]:
        bad.append("cfg10: среднее частей по seeds ≠ parquet")
    for (s, y), pt in parts.items():
        k = "%d_s%d" % (y, s)
        if STA.content_hash(pt) != rec["part_sha256"][k] or STA.content_hash(STA.load(R4.enc_path(y, s))) != rec["enc_sha256"][k]:
            bad.append("часть или веса энкодера %s: sha256 ≠ CHECKPOINT" % k)
    bad += recount(tab, pack, seq, Zg)
    return bad + order_problems(cp)


def check_run(ok: list, refused: list) -> int:
    before, cp = IO.outputs_digest(), CR.read_cp()
    bad = [] if refused == ["11"] else ["бегун: отказ должен быть ровно у 11 (нет cond_11), получено %s" % refused]
    bad += check(cp) if ok == ["10"] else ["бегун допустил %s, нужна только 10" % ok]
    if "eval" in cp.get("4", {}):                                      # оценка записана: таблицы, отчёт, ключи 4.eval / rules / cond11 / drift равны пересчёту
        from models_v2.s14 import v7_4_evrun as EVR
        bad += EVR.check_eval(cp)
    if IO.outputs_digest() != before:
        bad.append("--check изменил выходы (хэш до ≠ после)")
    print("\n".join(bad) if bad else "OK: G45 конфигурация 10 по sha256 CHECKPOINT (прогноз, части, веса энкодеров), F > 0, G = 0 ⇒ F = H, π в симплексе, 2018 = 6, годовая совместная и месячная переоценки побитно, 11 отказана; %sвыходы не изменены"
          % ("G46 таблицы оценки, дрейф, отчёт, 4.eval/rules/cond11 равны пересчёту; " if "eval" in cp.get("4", {}) else ""))
    return 1 if bad else 0
