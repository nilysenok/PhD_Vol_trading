"""P3-17, G17 (У-22, R11): LOTO на 17 — для каждой из 17 бумаг i: обучение панели на ОСТАЛЬНЫХ 16 (E11 в обучении нет, У-19), выбор по году (глубина, c, λ, κ, a, b, w) — по 16, индивидуальные части (K1, Ḡ, α) — из прошлого i; прогноз на i.
Не новая логика: боевой конвейер P17 (`hp_p17_hyb.Run17`) с обучающим составом `hp_p17.TRAIN` минус i (крючки `train_exclude` / `sel_mask` читают его при вызове; подмена — контекст `train_without`, восстанавливается в `finally`).
Поддержаны все конфигурации Run17 (A1…D2): проходы кэшируются под ключом с составом исключённых (Д5), так что A1 и D2 одного модуля делят проходы A1. Результаты на 28 (`loto*.parquet`) не трогаются: файлы `loto_p17_*`."""
from __future__ import annotations

import contextlib
import os
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_hyb_core as C, hp_loto as L, hp_p17 as X, hp_p17_hyb as H17, hp_paths as P

PROGRAM17 = "Ж-220-p17-LOTO"


@contextlib.contextmanager
def train_without(tk: str):
    """Обучающий состав 17 минус tk на время вызова (восстанавливается в finally; класс № 43/52/110)."""
    if tk not in X.TRAIN:
        raise ValueError("%s не из 17: LOTO определён только для утверждённых бумаг" % tk)
    old = X.TRAIN
    X.TRAIN = [t for t in old if t != tk]
    try:
        yield
    finally:
        X.TRAIN = old


def loto_ticker17(module: str, names: list, tk: str, nthread: int = 4, log=None, panel: pd.DataFrame | None = None, hp: dict | None = None) -> pd.DataFrame:
    """Прогноз σ̂² конфигураций `names` на строках tk моделью без tk (16 бумаг). -> config, ticker, date, t_plus, sigma2, a1_depth, a1_c, a1_lam, a1_g (строки с конечным прогнозом)."""
    out = []
    with train_without(tk):
        r = H17.Run17(module, nthread, log, panel=panel, hp=hp)
        order = (["A1"] if any(n[0] in "BD" or n == "A1" for n in names) else []) + [n for n in names if n != "A1"]
        for n in order:
            r.calc(n)
        m = (r.panel["ticker"] == tk).to_numpy()
        for n in names:
            ok = m & np.isfinite(r.sig[n])
            d = r.panel.loc[ok, ["ticker", "date", "t_plus"]].assign(config="%s_%s" % (module, n), sigma2=r.sig[n][ok])
            out.append(d.assign(**{"a1_" + k: r.a1[k][ok] for k in ("depth", "c", "lam", "g")}) if r.a1 else d)
    return pd.concat(out, ignore_index=True)


def _part(module: str, names: list, tk: str, key: str):
    return P.OUT / ("loto_p17_part_%s_%s_%s_%s.parquet" % (module, "-".join(names), tk, key))


def part_key(module: str, names: list, tk: str, panel: pd.DataFrame, sha: str, hp=None) -> str:
    """Ключ кэша бумаги: хэш входов панели, состав обучения без tk, имена, HP, код движка (Д5)."""
    base = panel["sigma2_k1"].to_numpy(float)
    extra = "loto17|%s|%s|%s|%s" % (tk, ",".join(t for t in X.TRAIN if t != tk), "+".join(names), C.data_sha(panel, C.Z28, base))
    return C.run_key(module, "loto17", 0, "qlike", C.Z28, hp, sha, extra)


def run_module(module: str, names: list, nthread: int = 4, log=print, reverse: bool = False, panel: pd.DataFrame | None = None, hp: dict | None = None) -> None:
    """Стадия `s17 --part loto --module m`: по бумаге (в прямом или обратном порядке — два процесса делят работу); `loto_p17_<module>.parquet` — когда готовы все 17; строки реестра."""
    pnl = pd.read_parquet(X.ENGINE_P17).reset_index(drop=True) if panel is None else panel      # panel задана только в тестах (синтетика)
    sha = P.sha16(X.ENGINE_P17) if panel is None else "synthetic"
    for tk in sorted(X.TRAIN, reverse=reverse):
        path = _part(module, names, tk, part_key(module, names, tk, pnl, sha, hp))
        if path.exists():
            continue
        t0 = time.time()
        df = loto_ticker17(module, names, tk, nthread, None, panel, hp)
        tmp = path.with_suffix(".tmp%d" % os.getpid())
        df.to_parquet(tmp)
        os.replace(tmp, path)
        log("LOTO17 %s %s %s: %.0f с" % (module, "+".join(names), tk, time.time() - t0))
    done = [_part(module, names, tk, part_key(module, names, tk, pnl, sha, hp)) for tk in X.TRAIN]
    if all(p.exists() for p in done):
        pd.concat([pd.read_parquet(p) for p in done], ignore_index=True).to_parquet(P.OUT / ("loto_p17_%s.parquet" % module))
        for n in names:
            log("LOTO17 %s %s: реестр %s" % (module, n, "добавлена" if L.register(module, n, PROGRAM17, 16, "Ж-220") else "уже была"))
