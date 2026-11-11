"""S-Z-IMPL-0 (Ж-258; `docs/otchety/harj_struct/S_Z_SPEC.md`, разд. 2.3, 3.3, 5, 8): склад z — внутридневное представление M10 (z энкодеров V7 Stage A, TCN-w5, 5 seeds, годы 2018…2026) на датах D_M и сдвинутое на 252 сессии.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Энкодер не обучается, PCA не пересчитывается; артефакты и `data/fwd/models/*` не перезаписываются. Новых библиотек нет.
Публичное API:
- `KIND` = "tcn", `YEARS` = 2018…2026, `SEEDS` = 0…4, `SHIFT` = 252, `V` = 1 (версия файла), `ZMAPS` = Z/zmaps.npz.
- `load_checked(path)`: артефакт Stage A с проверкой content_hash по CHECKPOINT V7 (`1b.stageA.hashes`); несовпадение — ValueError. `hashes45() -> {имя: хэш}`.
- `sessions() -> DatetimeIndex`: сетка сессий SBER (индекс `CM/feat.parquet`, 3124 сессии). `dm_dates()`: даты d окна D_M (2198).
- `frames_2026(src=None) -> {seed: DataFrame z0…z3 по d}`: `fwd_z.build_z(LAST_SESSION, Src(), seed)` — веса и PCA артефакта 2026, без обучения.
- `zgrids(sess=None, frames=None) -> {Y: {s: (n сессий, 4)}}`: z энкодера года Y (seed s) на сетке сессий; Y ≤ 2025 — `v7_1b_cfg.z_of_year` с проверкой хэша; Y = 2026 — `frames_2026`; NaN после 31.12.Y и где z нет.
- `take(zg, pos)`: строки сетки по позициям; позиция < 0 — NaN (без переноса). `shift_rows(zg, pos, shift=SHIFT)` = `take(zg, pos − shift)`.
- `zmaps(zg=None, dm=None, sess=None) -> {Y: {s: (2198, 4)}}`: z энкодера Y на датах D_M; NaN после 31.12.Y. `zshift(...)` — тот же словарь для z сессии pos(d) − 252 по сетке SBER (того же энкодера, без переноса: сессия < 0 или без z — NaN; после 31.12.Y — NaN, как у z).
- `pack(zm, zs, dm) -> dict` / `unpack(f) -> (zm, zs, dm)`: содержимое `zmaps.npz` (v = 1, content_hash 45 артефактов, sha256 `fwd_z.py`, кода и `feat.parquet`); `write(path=ZMAPS, ...)` — ТОЛЬКО после регистрации S-Z (отказ, если id S-RCKZ нет в `ALL_TRIALS.csv`) и без перезаписи отличающегося файла.
Запуск: `python3 -W ignore -m models_v2.s18.z_feat --check | --check-pit` (только чтение; код — `z_feat_chk.py`); `--write` — следующий лист."""
from __future__ import annotations

import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s14 import v7_1b_cfg as CFG
from models_v2.s14 import v7_1b_sta as STA
from models_v2.s16 import x1_reg as XR
from models_v2.s17 import fwd_store as FS
from models_v2.s17 import fwd_z as FZ
from models_v2.s17.fwd_src import LAST_SESSION, ROOT, Src
from models_v2.s18 import cm_feat as CF

KIND, YEARS, SEEDS, SHIFT, V = "tcn", STA.YEARS, STA.SEEDS, 252, 1
Z_DIR = ROOT / "data/regen/harj_struct/Z"
ZMAPS = Z_DIR / "zmaps.npz"
CKPT = ROOT / "docs/otchety/harj_hybrid_v7/CHECKPOINT.json"


def hashes45() -> dict:
    h = json.loads(CKPT.read_text(encoding="utf-8"))["1b"]["stageA"]["hashes"]
    return {STA.name(KIND, y, s): h[STA.name(KIND, y, s)] for y in YEARS for s in SEEDS}


def load_checked(path) -> dict:
    a, want = STA.load(path), hashes45()[Path(path).name]
    if STA.content_hash(a) != want:
        raise ValueError("%s: content_hash ≠ CHECKPOINT V7" % Path(path).name)
    return a


def sessions() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_parquet(CF.FEAT_PATH, columns=["rv"]).index)


def dm_dates() -> pd.DatetimeIndex:
    return CF.dm_dates()


def frames_2026(src: Src | None = None) -> dict:
    src = src or Src()
    store = FS.build_store(LAST_SESSION, src)
    xw = FS.build_xw5(store)
    return {s: FZ.build_z(LAST_SESSION, src, s, store, xw).set_index("d") for s in SEEDS}


def _grid_2026(f: pd.DataFrame, sess: pd.DatetimeIndex) -> np.ndarray:
    z, pos = np.full((len(sess), 4), np.nan), sess.get_indexer(f.index)
    z[pos[pos >= 0]] = f.to_numpy(float)[pos >= 0]
    return z


def zgrids(sess: pd.DatetimeIndex | None = None, frames: dict | None = None) -> dict:
    sess = sessions() if sess is None else sess
    frames = frames_2026() if frames is None else frames
    out = {}
    for y in YEARS:
        out[y] = {}
        for s in SEEDS:
            zg = _grid_2026(frames[s], sess) if y == 2026 else CFG.z_of_year(KIND, s, sess, y, load=load_checked)
            zg[np.asarray(sess >= pd.Timestamp(y + 1, 1, 1))] = np.nan
            out[y][s] = zg
    return out


def take(zg: np.ndarray, pos) -> np.ndarray:
    pos = np.asarray(pos)
    out, ok = np.full((len(pos), zg.shape[1]), np.nan), pos >= 0
    out[ok] = zg[pos[ok]]
    return out


def shift_rows(zg: np.ndarray, pos, shift: int = SHIFT) -> np.ndarray:
    return take(zg, np.asarray(pos) - shift)


def _maps(zg: dict | None, shift: int, dm, sess) -> dict:
    sess, dm = sessions() if sess is None else sess, dm_dates() if dm is None else dm
    zg, pos = zgrids(sess) if zg is None else zg, sess.get_indexer(dm)
    if (pos < 0).any():
        raise SystemExit("дата D_M не на сетке сессий SBER")
    out = {y: {s: shift_rows(zg[y][s], pos, shift) for s in SEEDS} for y in YEARS}
    for y in YEARS:
        for s in SEEDS:
            out[y][s][np.asarray(dm >= pd.Timestamp(y + 1, 1, 1))] = np.nan
    return out


def zmaps(zg: dict | None = None, dm=None, sess=None) -> dict:
    return _maps(zg, 0, dm, sess)


def zshift(zg: dict | None = None, dm=None, sess=None) -> dict:
    return _maps(zg, SHIFT, dm, sess)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pack(zm: dict, zs: dict, dm) -> dict:
    st = lambda m: np.stack([np.stack([m[y][s] for s in SEEDS]) for y in YEARS])
    h = hashes45()
    return dict(v=np.array(V), years=np.array(YEARS), seeds=np.array(SEEDS), d=np.asarray(dm, "datetime64[ns]"), z=st(zm), zs=st(zs), names=np.array(sorted(h)), hashes=np.array([h[k] for k in sorted(h)]),
                sha_fwd_z=np.array(_sha(FZ.__file__)), sha_code=np.array(_sha(__file__)), sha_feat=np.array(_sha(CF.FEAT_PATH)))


def unpack(f) -> tuple:
    a = np.load(f, allow_pickle=False)
    un = lambda k: {y: {s: a[k][i, j] for j, s in enumerate(a["seeds"].tolist())} for i, y in enumerate(a["years"].tolist())}
    return un("z"), un("zs"), pd.DatetimeIndex(a["d"])


def registered() -> bool:
    t = pd.read_csv(XR.TRIALS, comment="#", dtype=str, keep_default_na=False) if XR.TRIALS.exists() else pd.DataFrame(columns=XR.COLS)
    return "S-RCKZ" in set(t["id"])


def write(path: Path = ZMAPS, zm: dict | None = None, zs: dict | None = None) -> Path:
    if not registered():
        raise SystemExit("zmaps.npz: S-RCKZ нет в реестре — запись после регистрации S-Z (S-Z-IMPL-2)")
    dm, zg = dm_dates(), zgrids()
    arrs = pack(zmaps(zg, dm) if zm is None else zm, zshift(zg, dm) if zs is None else zs, dm)
    if path.exists():     # zip хранит время записи: сверяем содержимое массивов, не байты файла
        old = np.load(path, allow_pickle=False)
        if set(old.files) != set(arrs) or not all(np.array_equal(old[k], arrs[k], equal_nan=old[k].dtype.kind == "f") for k in arrs):
            raise SystemExit("%s существует и отличается — перезапись отказана" % path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp_" + path.name)
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrs)
    tmp.replace(path)
    return path


if __name__ == "__main__":
    from models_v2.s18 import z_feat_chk
    sys.exit(z_feat_chk.main(sys.argv[1:]))
