"""HAR-J Hybrid V7 (Ж-232), SBER, h = 1. Лист 0: `python3 -W ignore -m models_v2.s14.run_v7 --stage 0 [--check]`.
Без `--check`: пишет CONTROLS_V7.csv, M10_PASSPORT.csv, SAMPLES.csv, REUSE_MAP.md, AUDIT0_V7.md, CHECKPOINT.json (ключ `0`) и последовательности X_d (окна 1 и 5 дней) в data/regen/harj_hybrid_v7/.
С `--check`: ничего не пишет — пересчитывает всё в памяти и сверяет с сохранённым; хэш выходов до и после обязан совпасть. Моделей не обучает; конфигураций 0."""
from __future__ import annotations

import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_audit as AU, v7_0_ctrl as C, v7_0_io as IO, v7_0_pass as PA, v7_0_reuse as RE, v7_0_samples as SA, v7_0_seq as SQ

MIN_FREE_GB = 40
WINDOWS = (1, 5)
TOL_ID = 1e-9


def build_all() -> dict:
    """Всё содержимое выходов листа 0 в памяти (без записи)."""
    bars, d = PA.load()
    ctrl = C.check_rows()
    pas, pinfo = PA.passport()
    smp, sinfo = SA.samples()
    reuse = RE.table()
    store = SQ.build_store(bars, d.index)
    return dict(ctrl=ctrl, passport=pas, pinfo=pinfo, samples=smp, sinfo=sinfo, reuse=reuse, reuse_md=RE.markdown(reuse), audit_md=AU.markdown(), store=store, days=d.index)


def sequences(store: dict, W: int) -> dict:
    n = len(store["days"])
    X, M, ok, nok = np.zeros((n, W, SQ.SLOTS, SQ.K), np.float32), np.zeros((n, W, SQ.SLOTS), bool), np.zeros((n, W), bool), np.zeros(n, bool)
    for i in range(n):
        x, m, o, k = SQ.x_of(store, i, W)
        X[i], M[i], ok[i], nok[i] = x, m, o, k
    return dict(X=X, mask=M, day_ok=ok, norm_ok=nok, days=store["days"].to_numpy("datetime64[ns]"), channels=np.array(SQ.CH))


def gate_status(a: dict) -> dict:
    ctrl_ok = bool((a["ctrl"]["status"] == "совпало").all())
    id_ok = a["pinfo"]["max_rel"] <= TOL_ID and a["pinfo"]["n_nan"] == 0
    return dict(G1_controls_equal=ctrl_ok, G3_identity_max_rel=a["pinfo"]["max_rel"], G3_identity_ok=bool(id_ok), G5_first_a=a["sinfo"]["first_a"], G5_first_b=a["sinfo"]["first_b"], n_configurations=0, trainings_on_sber=0)


def write_stage(a: dict) -> None:
    free = shutil.disk_usage(IO.ROOT).free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit("свободно %.1f ГБ < %d: запись отменена" % (free, MIN_FREE_GB))
    st = gate_status(a)
    if not (st["G1_controls_equal"] and st["G3_identity_ok"]):
        raise SystemExit("G1/G3 не сошлись — выходы не пишутся: %s" % st)
    IO.OUT.mkdir(parents=True, exist_ok=True)
    IO.write_csv(a["ctrl"], "CONTROLS_V7.csv", "G1: контроли без изменений")
    IO.write_csv(a["passport"], "M10_PASSPORT.csv", "G3: паспорт M10 SBER; тождество Σr² = RV")
    IO.write_csv(a["samples"], "SAMPLES.csv", "G5: число закрытых пар на даты переоценки; первая дата (а) %s, (б) %s" % (a["sinfo"]["first_a"], a["sinfo"]["first_b"]))
    (IO.REP / "REUSE_MAP.md").write_text(a["reuse_md"], encoding="utf-8")
    (IO.REP / "AUDIT0_V7.md").write_text(a["audit_md"], encoding="utf-8")
    np.savez_compressed(IO.OUT / "store_raw.npz", raw=a["store"]["raw"].astype(np.float32), present=a["store"]["present"], days=a["days"].to_numpy("datetime64[ns]"), channels=np.array(SQ.CH))
    for W in WINDOWS:
        np.savez_compressed(IO.OUT / ("X_w%d.npz" % W), **sequences(a["store"], W))
    sizes = {f.name: f.stat().st_size for f in sorted(IO.OUT.glob("*.npz"))}
    cp = {"0": dict(status=st, code=IO.code_hash(), inputs=IO.inputs_hash(), out_files=sizes, out_bytes=int(sum(sizes.values())))}
    (IO.REP / "CHECKPOINT.json").write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print("записано; выходы data/regen: %s байт" % sum(sizes.values()))


def same_frame(x: pd.DataFrame, y: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(x.reset_index(drop=True), y.reset_index(drop=True), check_dtype=False, rtol=1e-12, atol=1e-12)
        return True
    except AssertionError:
        return False


def check_stage(a: dict) -> list:
    """-> список замечаний (пусто = всё сошлось). Ничего не пишет."""
    bad = []
    st = gate_status(a)
    if not st["G1_controls_equal"]:
        bad.append("G1: контроли не равны CMP_SERIES/CMP_WINDOWS")
    if not st["G3_identity_ok"]:
        bad.append("G3: тождество Σr² = RV нарушено (max %.3g)" % st["G3_identity_max_rel"])
    for name, key in (("CONTROLS_V7.csv", "ctrl"), ("M10_PASSPORT.csv", "passport"), ("SAMPLES.csv", "samples")):
        if not (IO.REP / name).exists() or not same_frame(a[key], IO.read_csv(name)):
            bad.append("%s: нет или не совпал с пересчётом" % name)
    for name, key in (("REUSE_MAP.md", "reuse_md"), ("AUDIT0_V7.md", "audit_md")):
        p = IO.REP / name
        if not p.exists() or p.read_text(encoding="utf-8") != a[key]:
            bad.append("%s: нет или не совпал с пересчётом" % name)
    bad += check_sequences(a["store"])
    cpp = IO.REP / "CHECKPOINT.json"
    if not cpp.exists() or "0" not in json.loads(cpp.read_text(encoding="utf-8")):
        bad.append("CHECKPOINT.json: нет ключа 0")
    return bad


def check_sequences(store: dict) -> list:
    bad = []
    for W in WINDOWS:
        p = IO.OUT / ("X_w%d.npz" % W)
        if not p.exists():
            bad.append("%s: нет" % p.name)
            continue
        z = np.load(p)
        for i in (30, 500, len(store["days"]) // 2, len(store["days"]) - 1):
            x, m, o, k = SQ.x_of(store, i, W)
            if not (np.array_equal(z["X"][i], x.astype(np.float32)) and np.array_equal(z["mask"][i], m) and bool(z["norm_ok"][i]) == k):
                bad.append("%s: день %d не совпал с построителем" % (p.name, i))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["0", "1a", "1b", "2", "3", "4"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--config", default=None, help="id конфигурации реестра (стадии 1b–4); по умолчанию — все конфигурации стадии")
    args = ap.parse_args(argv)
    if args.stage != "0":                                   # лист 1a-M: формы, реестр, каркас конфигураций (`v7_1m_stage`; torch импортируется только здесь)
        from models_v2.s14 import v7_1m_stage as S1
        return S1.run(args.stage, args.check, args.config)
    before = IO.outputs_digest()
    a = build_all()
    if args.check:
        bad = check_stage(a)
        after = IO.outputs_digest()
        if before != after:
            bad.append("--check изменил выходы (хэш до ≠ после)")
        print("\n".join(bad) if bad else "OK: G1 sha/QLIKE равны CMP; G3 max отн. разн. %.3g; выходы не изменены (хэш %s)" % (a["pinfo"]["max_rel"], after[:16]))
        return 1 if bad else 0
    write_stage(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
