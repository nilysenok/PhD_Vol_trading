"""Эталон эквивалентности на диске: npz с записями ekviv.py (строки — как байты utf-8, hex-словарь — json)."""
import json

import numpy as np

MASSIVY = ("p", "t", "pz", "sd", "vi")
TEKST = ("sig", "disk")


def sohranit(path: str, rez: dict, meta: dict) -> None:
    arr = {"__meta__": np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)}
    for imya, r in rez.items():
        for k in TEKST:
            arr[f"{imya}__{k}"] = np.frombuffer(r[k].encode(), dtype=np.uint8)
        arr[f"{imya}__hex"] = np.frombuffer(json.dumps(r["hex"], sort_keys=True).encode(), dtype=np.uint8)
        arr.update({f"{imya}__{k}": r[k] for k in MASSIVY})
    with open(path, "wb") as f:
        np.savez(f, **arr)


def zagruzit(path: str) -> tuple:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z["__meta__"].tobytes().decode())
        imena = sorted({k.rsplit("__", 1)[0] for k in z.files if k != "__meta__"})
        rez = {}
        for m in imena:
            r = {k: z[f"{m}__{k}"].tobytes().decode() for k in TEKST}
            r["hex"] = json.loads(z[f"{m}__hex"].tobytes().decode())
            r.update({k: z[f"{m}__{k}"] for k in MASSIVY})
            rez[m] = r
    return rez, meta


def tochki_iz(rez: dict) -> list:
    return sorted((m.split("__")[1], int(m.split("__")[2])) for m in rez if m.startswith("tochka__"))
