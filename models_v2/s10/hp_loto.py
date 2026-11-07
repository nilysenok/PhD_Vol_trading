"""P3, G12 (R11, У-13): leave-one-ticker-out для конфигураций из `statistics/SELECTION.json` (`loto`): A1-XGB, A1-LGBM и позиция тройки.
Для каждой бумаги i — отдельное обучение панели БЕЗ i (все 28 минус i, месячное переобучение; `E.walk_forward(exclude=(i,))`); правила выбора года Y (глубина, c, λ) — по закрытым прогнозам Y−1 по ОСТАЛЬНЫМ 27 бумагам;
индивидуальные части (K1 бумаги i) — её прошлое (`base`). Поддерживаются конфигурации A1 (qlike) и A2 (l2log) на признаках Z28; другие — отказ. Это оценка, не конфигурация бюджета (в Холм входит)."""
from __future__ import annotations

import csv
import hashlib
import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_engine as E, hp_hyb_core as C, hp_hyb_out as O, hp_paths as P

LOSS = {"A1": "qlike", "A2": "l2log"}
PROGRAM = "Ж-217-LOTO"       # префикс строк реестра: не попадает в `used` бюджета 24 (префикс «Ж-217:»)


def loto_ticker(module: str, name: str, panel: pd.DataFrame, base, tk: str, hp=None, nthread: int = 3, log=None, extra_exclude=()) -> pd.DataFrame:
    """Прогноз σ̂² на строках бумаги tk моделью, обученной без tk; выбор года Y по 27 прочим. -> ticker, date, t_plus, depth, c, lam, g, sigma2.
    extra_exclude (P3-17, Ж-220): бумаги вне обучения и вне выбора года (E11) — обучение на 17 без tk, выбор по 16; по умолчанию пусто (как на 28)."""
    loss, base = LOSS[name], np.asarray(base, float)
    ex = (tk,) + tuple(extra_exclude)
    runs = {d: E.walk_forward(panel, C.Z28, base, loss, module, d, exclude=ex, hp=hp, nthread=nthread, log=log) for d in E.DEPTHS}
    oth = ~np.isin(panel["ticker"].to_numpy(), list(ex))
    po, bo = panel[oth].reset_index(drop=True), base[oth]
    ro = {d: r[oth].reset_index(drop=True) for d, r in runs.items()}
    me, yr = (panel["ticker"] == tk).to_numpy(), pd.DatetimeIndex(panel["date"]).year.to_numpy()
    sig, dep, cc, ll, gg = (np.full(len(panel), np.nan) for _ in range(5))
    for Y in sorted(set(yr[yr >= E.FIRST_YEAR])):
        sel = E.choose(po, bo, ro, Y, loss)
        m, r = me & (yr == Y), runs[sel["depth"]]
        g, s2 = r["g"].to_numpy()[m], r["s2g"].to_numpy()[m]
        sig[m] = E.apply_corr(base[m], g, s2, sel["c"], sel["lam"], loss)
        dep[m], cc[m], ll[m], gg[m] = sel["depth"], sel["c"], sel["lam"], g
    out = panel[["ticker", "date", "t_plus"]].assign(depth=dep, c=cc, lam=ll, g=gg, sigma2=sig)
    return out[me & np.isfinite(sig)].reset_index(drop=True)


def _part_path(module: str, name: str, tk: str, key: str):
    return P.OUT / ("loto_part_%s_%s_%s_%s.parquet" % (module, name, tk, key))


def compute(module: str, name: str, panel: pd.DataFrame, base, sha: str, hp=None, nthread: int = 3, log=print) -> pd.DataFrame:
    """Все бумаги панели для (модуль, конфигурация); результат по бумаге кэшируется (ключ — хэш входов и кода, Д5)."""
    ds, parts = C.data_sha(panel, C.Z28, base), []
    for tk in sorted(panel["ticker"].unique()):
        key = C.run_key(module, "loto_" + name, 0, LOSS[name], C.Z28, hp, sha, ds + tk)
        path = _part_path(module, name, tk, key)
        if not path.exists():
            t0 = time.time()
            loto_ticker(module, name, panel, base, tk, hp, nthread).to_parquet(path)
            log("LOTO %s %s %s: %.0f с" % (module, name, tk, time.time() - t0))
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True)


def register(module: str, name: str, program: str = PROGRAM, n_tickers: int = 28, source: str = "Ж-217") -> bool:
    """Строка реестра одного LOTO-прогона (по одной; повтор не дублируется). `used` бюджета не меняется (R11). program / n_tickers / source — P3-17 (`Ж-220-p17-LOTO`, 17 бумаг, «Ж-220»)."""
    with O.lock():
        pre = "%s:%s:%s:" % (program, module, name)
        if any(r.startswith(pre) for r in O.REG.read_text(encoding="utf-8").splitlines()):
            return False
        js = json.dumps({"name": name, "module": module, "kind": "loto", "trial": True, "arch": "leave_one_ticker_out", "n_tickers": n_tickers}, ensure_ascii=False, sort_keys=True)
        with open(O.REG, "a", newline="", encoding="utf-8") as f:
            if not O.REG.read_bytes().endswith(b"\n"):
                f.write("\n")
            csv.writer(f, lineterminator="\n").writerow([pre + hashlib.sha256(js.encode()).hexdigest()[:12], source, "harj_panel_hybrid", "leave_one_ticker_out", js, "trial"])
        return True


def run_module(module: str, names: list, nthread: int = 3, log=print) -> None:
    """Стадия `loto --module m`: считает прогоны модуля из списка; `loto_<module>.parquet` (длинный: config) и строки реестра."""
    bad = O.check_hashes()
    if bad:
        raise SystemExit("стоп: входы не совпали: %s" % bad)
    panel = pd.read_parquet(P.OUT / "engine_panel.parquet").reset_index(drop=True)
    sha, base, outs = P.sha16(P.OUT / "engine_panel.parquet"), panel["sigma2_k1"].to_numpy(float), []
    for name in names:
        outs.append(compute(module, name, panel, base, sha, None, nthread, log).assign(config="%s_%s" % (module, name)))
        log("LOTO %s %s: реестр %s" % (module, name, "добавлена" if register(module, name) else "уже была"))
        pd.concat(outs, ignore_index=True).to_parquet(P.OUT / ("loto_%s.parquet" % module))
