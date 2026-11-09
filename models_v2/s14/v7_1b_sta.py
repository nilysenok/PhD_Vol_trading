"""V7-1b-A: Stage A на данных SBER. Энкодеры TCN-w5 и CNN-w5, годы Y = 2018…2026, seeds 0–4, только бары сессий < 01.01.Y (`v7_1e_pre.train_encoder`, ε = 0,01·медиана положительных r²).
Артефакт (kind, Y, seed) — `data/regen/harj_hybrid_v7/stageA/<kind>_w5_<Y>_s<seed>.npz`: веса, z (4 главные компоненты по z дат < 01.01.Y) для дней до конца года Y, даты, PCA, мета.
Хэш артефакта — sha256 СОДЕРЖИМОГО массивов (zip-файл побайтно не воспроизводится: метки времени). Данные читаются только при вызове функций (не при импорте)."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO

KINDS, YEARS, SEEDS, WINDOW = ("tcn", "cnn"), tuple(range(2018, 2027)), tuple(range(5)), 5


def directory():
    return IO.OUT / "stageA"


def tasks() -> list:
    return [(k, y, s) for y in YEARS for s in SEEDS for k in KINDS]


def name(kind: str, year: int, seed: int, window: int = WINDOW) -> str:
    return "%s_w%d_%d_s%d.npz" % (kind, window, year, seed)


def content_hash(arrs: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def rows_to_year_end(pack: dict, year: int) -> np.ndarray:
    """Дни, для которых хранится z энкодера года Y: сессии до 31.12.Y включительно (прошлое для главных компонент и E0 + сам год Y)."""
    before = pd.DatetimeIndex(pack["days"]) < pd.Timestamp(year + 1, 1, 1)
    return np.flatnonzero(before & pack["norm_ok"] & pack["mask"][:, -1].any(1))


def compute(kind: str, year: int, seed: int, pack: dict, window: int = WINDOW, **kw) -> dict:
    """-> словарь массивов артефакта. kw — только для тестов на синтетике (max_epochs, min_rows). window = 1 — энкодер конфигурации 3 (лист 1b-C)."""
    from models_v2.s14 import v7_1e_pre as PRE, v7_1e_rep as REP
    m = PRE.train_encoder(kind, window, year, seed, pack, **kw)
    rows = rows_to_year_end(pack, year)
    z, p = REP.z_pit(m, pack, rows, year)
    a = {"w__" + k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    a.update(z=z, days=np.asarray(pack["days"])[rows].astype("datetime64[ns]").astype(np.int64), rows=rows)
    a.update({"pca_" + k: np.asarray(p[k], float) for k in ("mu", "sd", "comps", "ssd")})
    mt = m.meta
    a["meta"] = np.array([mt["eps"], mt["n_train"], mt["n_val"], mt["epochs"], mt["best_val"], mt["n_params"]], float)
    return a


def save(arrs: dict, path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp_" + path.name)
    np.savez(tmp, **arrs)
    os.replace(tmp, path)
    return content_hash(arrs)


def load(path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def file_hash(kind: str, year: int, seed: int) -> str:
    return content_hash(load(directory() / name(kind, year, seed)))


def train_missing(shard: int = 0, n_shards: int = 1, log=print) -> int:
    """Считает недостающие артефакты задач shard::n_shards (возобновляемо). -> число посчитанных."""
    from models_v2.s14 import v7_1e_pre as PRE
    pack, done = PRE.pack_from_npz(WINDOW, IO.OUT), 0
    for i, (k, y, s) in enumerate(tasks()):
        if i % n_shards != shard or (directory() / name(k, y, s)).exists():
            continue
        save(compute(k, y, s, pack), directory() / name(k, y, s))
        done += 1
        log("%s %d seed %d готово" % (k, y, s))
    return done


def missing() -> list:
    return [name(*t) for t in tasks() if not (directory() / name(*t)).exists()]


if __name__ == "__main__":
    import sys
    sh, n = int(sys.argv[1]), int(sys.argv[2])
    print("shard %d/%d: посчитано %d" % (sh, n, train_missing(sh, n)))
