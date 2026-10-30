#!/usr/bin/env python3
"""К-5: прогон `vyr_etalony_finaly.py hxp|f1|f5` на теневом корне с новыми матрицами без дней ДСВД (docs/wf_sigma_910_dsvd.md, разд. 1, 5).
Теневой корень (FASTBT_MAIN=<shadow>): симлинки на основную копию, кроме новых матриц; выходы vyr1, test_locked/vyr, R2/cache/VYR — симлинки
в каталог out, вне основной копии и data/. FASTBT_MAIN обязателен и равен теневому корню, без него отказ.
  FASTBT_MAIN=<shadow> python3 scripts/dannye/wf_sigma_910_progon.py shadow --osnova <основная копия> --novye <каталог с data/regen/...> --shadow <shadow> --out <out>
  FASTBT_MAIN=<shadow> python3 scripts/dannye/wf_sigma_910_progon.py run --osnova ... --novye ... --shadow <shadow> --out <out> hxp f1 f5
Код scripts/analysis/ не правится; пути, зашитые в нём вне корня, не обходятся — доклад в отчёте."""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VYR_PY = REPO / "scripts" / "analysis" / "vyr_etalony_finaly.py"
NOVYE = ("features_pit_v3.parquet", "features_pit_v3.config.json", "features_pit_v3_rrv.parquet",
         "features_pit_v3_d7_h5.parquet", "features_pit_v3_d7_h22.parquet", "D13/intraday1850.parquet")
VYHODY = {"regen/vyr/vyr1": "vyr1", "regen/test_locked/vyr": "test_locked_vyr", "regen/R2/cache/VYR": "cache_VYR"}  # куда пишет vyr
MODES = ("hxp", "f1", "f5")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def korenj_progona(shadow, osnova):
    """FASTBT_MAIN обязателен и равен теневому корню (как в scripts/analysis, К-4); без переменной или на основную копию — отказ."""
    v = os.environ.get("FASTBT_MAIN")
    if not v:
        raise SystemExit("FASTBT_MAIN не задан: укажите теневой корень")
    if Path(v).resolve() != Path(shadow).resolve():
        raise SystemExit("FASTBT_MAIN != --shadow: корень прогона должен быть теневым")
    if Path(v).resolve() == Path(osnova).resolve():
        raise SystemExit("FASTBT_MAIN указывает на основную копию")
    return Path(v).resolve()


def _mirror(src, dst, replace=()):
    """dst — настоящий каталог; каждый элемент src — симлинк, кроме имён из replace (их создаёт вызывающий)."""
    dst.mkdir(parents=True, exist_ok=True)
    for e in sorted(src.iterdir()):
        if e.name in replace or (dst / e.name).is_symlink() or (dst / e.name).exists():
            continue
        (dst / e.name).symlink_to(e)


def build_shadow(main, novye, shadow, out):
    """Строит теневой корень. Новые матрицы — из novye/data/regen, выходы — в out (симлинки)."""
    main, novye, shadow, out = (Path(x).resolve() for x in (main, novye, shadow, out))
    if shadow == main or main in shadow.parents:
        raise SystemExit("теневой корень внутри основной копии запрещён")
    for k in VYHODY.values():
        (out / k).mkdir(parents=True, exist_ok=True)
    _mirror(main, shadow, replace=("data",))
    _mirror(main / "data", shadow / "data", replace=("regen",))
    d = shadow / "data" / "regen"
    top = {p.split("/")[0] for p in NOVYE} | {"vyr", "test_locked", "R2"}
    _mirror(main / "data" / "regen", d, replace=top)
    for e in NOVYE:
        if "/" not in e:
            (d / e).symlink_to(novye / "data" / "regen" / e)
    _mirror(main / "data" / "regen" / "D13", d / "D13", replace=("intraday1850.parquet",))
    (d / "D13" / "intraday1850.parquet").symlink_to(novye / "data" / "regen" / "D13" / "intraday1850.parquet")
    _mirror(main / "data" / "regen" / "vyr", d / "vyr", replace=("vyr1",))
    _mirror(main / "data" / "regen" / "test_locked", d / "test_locked", replace=("vyr",))
    _mirror(main / "data" / "regen" / "R2", d / "R2", replace=("cache",))
    _mirror(main / "data" / "regen" / "R2" / "cache", d / "R2" / "cache", replace=("VYR",))
    for rel, k in VYHODY.items():
        (shadow / "data" / rel).symlink_to(out / k)
    return shadow


def proverit_ten(main, novye, shadow, out):
    """Сверка входов по хэшам: пять матриц (+ конфиг, D13) теневого корня = новые; выходы — в out; остальное — основная копия."""
    main, novye, shadow, out = (Path(x).resolve() for x in (main, novye, shadow, out))
    err = []
    for e in NOVYE:
        s, n = shadow / "data" / "regen" / e, novye / "data" / "regen" / e
        if not s.exists() or sha(s) != sha(n):
            err.append(f"вход {e}: хэш теневого корня != новая матрица")
        if Path(os.path.realpath(s)) == main / "data" / "regen" / e:
            err.append(f"вход {e}: теневой корень указывает на основную копию (старая база)")
    for rel in VYHODY:
        t = Path(os.path.realpath(shadow / "data" / rel))
        if out not in t.parents and t != out:
            err.append(f"выход {rel} -> {t}: вне out")
        if main in t.parents:
            err.append(f"выход {rel} в основной копии")
    return err


def _vyr(out):
    import importlib.util
    sp = importlib.util.spec_from_file_location("vyr_etalony_finaly", VYR_PY)
    V = importlib.util.module_from_spec(sp)
    sys.modules["vyr_etalony_finaly"] = V
    sp.loader.exec_module(V)
    if Path(os.path.realpath(V.WORK)).parent != Path(out).resolve():
        raise SystemExit(f"WORK {V.WORK} вне out")
    return V


def f_v_obvyazke(out, mode, staraya, nseeds):
    """f1/f5 исходным cmd_f1/cmd_f5. staraya (G1'): СТАРАЯ матрица в исходном порядке, контроль R1 обязан быть Δ=0 — после него стоп (d0 не считается).
    Иначе (новые матрицы): контроль R1 не останавливает прогон — порядок строк и подвыборка бустинга дают эффект смены seed (толкование c11893bf),
    результат контроля пишется в control_<mode>.csv и в журнал. ↺ seed: nseeds из 10 (42..42+nseeds-1)."""
    V = _vyr(out)
    V.SEEDS = tuple(range(42, 42 + nseeds))
    V.WORK.mkdir(parents=True, exist_ok=True)
    if staraya:
        f = V.f5_setup

        def stop_d0(d0, *a, **k):
            if d0:
                raise SystemExit("G1': контроль R1 пройден, d0 не считается")
            return f(d0, *a, **k)
        V.f5_setup = stop_d0
    else:
        for nm in ("control", "control_q"):
            V.__dict__[nm] = _mjagkij(V.__dict__[nm])
    getattr(V, f"cmd_{mode}")(argparse.Namespace(workers=3))


def _mjagkij(fn):
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except SystemExit as e:
            print(f"(контроль R1 не остановил прогон на новых входах: {e})")
    return w


def hxp_kontrol_uzkij(out):
    """↺ hxp: контроль R1 исходного cmd_hxp сверяет весь ряд с D22/HXP_lnall (26.10) и на новых входах падает по определению: строки >= 2025
    другие (ДСВД), а календарь экспираций изменён 29.10 (MIX-3.22, ec9ea1bb) -> cal_dte строк 2022-Q1 и модели шагов >= 8 (обучение на 2022).
    Здесь контроль — тот же код R1 и тот же допуск, но на строках < 2022 и шагах 1–7; d0-часть — hxp_ln(d0=True) как есть."""
    import numpy as np
    import pandas as pd
    V = _vyr(out)
    r1 = V.hxp_ln(d0=False)
    ref = pd.read_parquet(V.REG / "D22" / "HXP_lnall.parquet")
    r1, ref = (x[pd.to_datetime(x.date) < "2022-01-01"] for x in (r1, ref))
    m = r1.merge(ref, on=["date", "ticker"], suffixes=("", "_r1"), validate="one_to_one")
    d = max(float(np.nanmax(np.abs(m[f"hxp_1_{k}"] - m[f"hxp_1_{k}_r1"]))) for k in range(1, 8))
    print(f"контроль HXP R1 (< 2022, шаги 1-7): max|dln| = {d:.2e}, строк {len(m)} из {len(ref)}/{len(r1)}")
    if not (d <= V.TOL and len(m) == len(ref) == len(r1)):
        raise SystemExit("СТОП: HAR-X+ R1 на строках < 2022, шаги 1-7, не повторён")
    V.put(V.hxp_ln(d0=True), V.WORK / "HXP_lnall.parquet")


def zapusk(shadow, mode, workers=3):
    env = dict(os.environ, FASTBT_MAIN=str(shadow), NUMBA_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(VYR_PY), mode, "--workers", str(min(workers, 3))], env=env).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("shadow", "run"))
    ap.add_argument("modes", nargs="*")
    for a in ("--osnova", "--novye", "--shadow", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--staraya", action="store_true", help="G1': novye = основная копия (старые матрицы), только контроль R1 f1/f5")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_intermixed_args()
    m = Path(a.osnova).resolve()
    korenj_progona(a.shadow, m)
    if a.cmd == "shadow":
        build_shadow(m, a.novye, a.shadow, a.out)
    err = proverit_ten(m, a.novye, a.shadow, a.out)
    if err and not (a.staraya and Path(a.novye).resolve() == m):
        raise SystemExit("СТОП: " + "; ".join(err))
    print("теневой корень: входы по хэшам = " + ("СТАРЫЕ (G1')" if a.staraya else "новые") + ", выходы в out")
    if len(a.modes) > 1:  # каждый режим — отдельный процесс: подмены l2_lib у h=1 и h=5 разные (docstring vyr_etalony_finaly)
        base = [x for x in sys.argv[1:] if x not in a.modes]
        for mode in a.modes:
            rc = subprocess.run([sys.executable, __file__, *base, mode]).returncode
            if rc:
                raise SystemExit(f"СТОП: {mode} rc={rc}")
        return
    for mode in (a.modes if a.cmd == "run" else []):
        if mode not in MODES:
            raise SystemExit(f"режим {mode} не из {MODES}")
        if mode == "hxp":
            hxp_kontrol_uzkij(a.out)
            continue
        try:
            f_v_obvyazke(a.out, mode, a.staraya, a.seeds)
        except SystemExit as e:
            if not (a.staraya and "d0" in str(e)):
                raise
            print(f"{mode}: {e}")


if __name__ == "__main__":
    main()
