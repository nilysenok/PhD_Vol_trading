"""Проверки S-Z-IMPL-0 (`z_feat`): --check (Z0.G1), --check-pit (Z0.G2). Только чтение: sha256 каталога выходов `Z/` до и после; запись `zmaps.npz` проверяется через память."""
from __future__ import annotations

import torch  # noqa: F401  (первым, класс № 152)

import hashlib
import io

import numpy as np
import pandas as pd

from models_v2.s14 import v7_1b_sta as STA
from models_v2.s17 import fwd_store as FS
from models_v2.s17 import fwd_z as FZ
from models_v2.s17.fwd_src import ARCHIVE_END, LAST_SESSION, Src
from models_v2.s18 import z_feat as F


def dir_hash(d=F.Z_DIR) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")) if d.exists() else []:
        h.update(str(p.relative_to(d)).encode() + (p.read_bytes() if p.is_file() else b""))
    return h.hexdigest()


def n_missing(m: dict, dm, years=F.YEARS, seeds=F.SEEDS) -> dict:
    """{(Y, s): число дат D_M ≤ 31.12.Y без конечного z}."""
    return {(y, s): int((~np.isfinite(m[y][s][np.asarray(dm < pd.Timestamp(y + 1, 1, 1))]).all(1)).sum()) for y in years for s in seeds}


def same(a, b) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def check_hashes() -> list:
    ok = sum(STA.content_hash(STA.load(STA.directory() / k)) == v for k, v in F.hashes45().items())
    print("content_hash = CHECKPOINT V7: %d из %d" % (ok, len(F.hashes45())))
    return [] if ok == 45 else ["content_hash %d из 45" % ok]


def check_missing(zm: dict, zs: dict, dm) -> list:
    bad = []
    for tag, m in (("z", zm), ("z сессии s − 252", zs)):
        miss = n_missing(m, dm)
        print("нет %s на датах D_M ≤ 31.12.Y: %d (по всем Y, s; Y = 2018…2026 × 5 seeds)" % (tag, sum(miss.values())))
        bad += ["нет %s: %s" % (tag, {k: v for k, v in miss.items() if v})] * (sum(miss.values()) > 0)
    f = np.asarray(dm > ARCHIVE_END)
    mf = sum(int((~np.isfinite(zm[2026][s][f]).all(1)).sum()) for s in F.SEEDS)
    print("2026 F (d > %s): дат %d; нет z (все seeds): %d" % (ARCHIVE_END.date(), f.sum(), mf))
    return bad + ["2026 F: %d дат или пропуски %d" % (f.sum(), mf)] * (f.sum() != 179 or mf != 0)


def check_direct(zm: dict, dm) -> list:
    """Независимый путь V7: `z_of_year` прямо на датах D_M (Y ≤ 2025) против z через сетку сессий SBER."""
    bad = [(y, s) for y in F.YEARS[:-1] for s in F.SEEDS if not same(zm[y][s], F.CFG.z_of_year(F.KIND, s, dm, y, load=F.load_checked))]
    print("z через сетку сессий = `v7_1b_cfg.z_of_year` на датах D_M (Y ≤ 2025, 40 энкодеров): %s" % ("побитно" if not bad else "РАСХОДЯТСЯ %s" % bad))
    return ["z ≠ z_of_year %s" % bad] * bool(bad)


def check_archive(frames: dict) -> list:
    bad = []
    for s in F.SEEDS:
        a = F.load_checked(STA.directory() / STA.name(F.KIND, 2026, s))
        ad = pd.DatetimeIndex(a["days"].astype("datetime64[ns]"))
        com = ad.intersection(frames[s].index)
        e = float(np.abs(frames[s].loc[com].to_numpy() - np.asarray(a["z"], float)[ad.get_indexer(com)]).max())
        print("seed %d: архив 2026 против build_z: общих дней %d, max|Δz| = %.1e" % (s, len(com), e))
        bad += ["архив 2026 seed %d: %d дней, %.1e" % (s, len(com), e)] * (len(com) != 2925 or e != 0.0)
    return bad


def check_shift_synth(sess, dm) -> list:
    zg = np.repeat(np.arange(300.0)[:, None], 4, axis=1)
    pos = np.array([0, 100, 251, 252, 253, 299])
    got, want = F.shift_rows(zg, pos), np.array([np.nan, np.nan, np.nan, 0.0, 1.0, 47.0])[:, None] * np.ones(4)
    p = sess.get_indexer(dm)
    n_row = int((sess[p[252:] - 252] == dm[:-252]).sum())
    print("сдвиг 252: известный ответ (позиции <252 — NaN, без переноса) %s; дата сессии s − 252 = дата строки i − 252: %d из %d (сдвиг по строкам D_M расходится на %d)" % ("ок" if same(got, want) else "ОШИБКА", n_row, len(dm) - 252, len(dm) - 252 - n_row))
    return [] if same(got, want) else ["сдвиг: известный ответ"]


def check_file(zm: dict, zs: dict, dm) -> list:
    buf = io.BytesIO()
    np.savez(buf, **F.pack(zm, zs, dm))
    buf.seek(0)
    z2, s2, d2 = F.unpack(buf)
    srcs = [("память → npz → память", z2, s2, d2)]
    if F.ZMAPS.exists():
        srcs.append(("файл zmaps.npz", *F.unpack(F.ZMAPS)))
    bad = []
    for tag, z_, s_, d_ in srcs:
        ok = all(same(zm[y][s], z_[y][s]) and same(zs[y][s], s_[y][s]) for y in F.YEARS for s in F.SEEDS) and bool((d_ == dm).all())
        print("z, zs: %s: %s" % (tag, "побитно" if ok else "РАСХОДЯТСЯ"))
        bad += ["z ≠ %s" % tag] * (not ok)
    if not F.ZMAPS.exists():
        print("файла zmaps.npz нет (запись после регистрации S-Z): сверка с файлом — только через память")
    return bad


def check() -> list:
    h0, sess, dm = dir_hash(), F.sessions(), F.dm_dates()
    frames = F.frames_2026()
    zg = F.zgrids(sess, frames)
    zm, zs = F.zmaps(zg, dm, sess), F.zshift(zg, dm, sess)
    print("D_M: %d дат; сетка сессий SBER: %d (%s…%s)" % (len(dm), len(sess), sess[0].date(), sess[-1].date()))
    bad = check_hashes() + check_missing(zm, zs, dm) + check_direct(zm, dm) + check_archive(frames) + check_shift_synth(sess, dm) + check_file(zm, zs, dm)
    bad += [] if dir_hash() == h0 else ["выходы Z/ изменены --check"]
    print("sha256 выходов Z/ до и после: %s" % ("равны" if dir_hash() == h0 else "РАЗЛИЧАЮТСЯ"))
    return bad


def check_pit() -> list:
    src, dm = Src(), F.dm_dates()
    print("замок открытия существует (Src.is_open, содержимое не читалось):", src.is_open())
    Fd = dm[dm > ARCHIVE_END]
    full, bad = F.frames_2026(src), []
    for dk in (Fd[0], Fd[len(Fd) // 2], Fd[-1]):
        st = FS.build_store(dk, src)
        xk = FS.build_xw5(st)
        e = max(float(np.abs(FZ.build_z(dk, src, s, st, xk).set_index("d").loc[dk].to_numpy() - full[s].loc[dk].to_numpy()).max()) for s in F.SEEDS)
        print("PIT: d %s — max|z(cut=d) − z(cut=%s)| по 5 seeds = %.1e" % (dk.date(), LAST_SESSION.date(), e))
        bad += ["PIT d %s: %.1e" % (dk.date(), e)] * (e != 0.0)
    return bad


def main(args: list) -> int:
    fn = {"--check": check, "--check-pit": check_pit}
    k = [a for a in args if a in fn]
    if not k:
        raise SystemExit("укажите --check или --check-pit (--write — после регистрации, отдельный лист)")
    bad = fn[k[0]]()
    print("%s: %s" % (k[0], "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0
