"""P3-17, G19 (Ж-220, У-25): LOTO на 17 с поперечными признаками по 16. Для каждой из 17 бумаг i: панель строится заново — m_med, m_disp, m_share_up, c_rank, c_dev_med дня по 16 бумагам (17 \\ {i}),
признаки бумаги i (и E11) — ранг и отклонение ОТНОСИТЕЛЬНО 16 (как E11 относительно 17, У-19); обучение и выбор по году — по 16; прогноз на i. Дальше — боевой путь `hp_s17_loto.loto_ticker17` без изменений.
Не новые конфигурации: те же 3 прогона (xgb_A1, lgbm_A1, lgbm_D2). Прежний LOTO (`loto_p17_*`, поперечные по 17 — признаки i несли её саму) не трогается: файлы `loto16_p17_*`; строки реестра `Ж-220-p17-LOTO16:*`."""
from __future__ import annotations

import os
import time

import pandas as pd

from models_v2.s10 import hp_ctrl as CT, hp_hyb_core as C, hp_loto as L, hp_p17 as X, hp_paths as P, hp_s17_loto as L17

PROGRAM16 = "Ж-220-p17-LOTO16"
VERSION = "loto16-v1"


def cross_without(tk: str) -> list:
    """Состав поперечных признаков прогона без tk (У-25): 17 \\ {tk}. Крючок: мутация подставляет сюда 17 (исключённая бумага в поперечных признаках)."""
    return [t for t in X.TRAIN if t != tk]


def load_world() -> tuple:
    """(daily, IMOEX, K1) боевых входов; K1 не зависит от состава поперечных признаков — считается один раз на процесс."""
    daily = {t: pd.read_parquet(P.OUT / ("daily_%s.parquet" % t)) for t in P.TICKERS}
    return daily, pd.read_parquet(P.OUT / "daily_IMOEX.parquet"), CT.k1_ext(daily)


def panel16(tk: str, daily: dict, imo: pd.DataFrame, k1: dict | None = None) -> pd.DataFrame:
    """Панель движка для прогона без tk: все колонки как `hp_p17.build_panel`, поперечные — по `cross_without(tk)`."""
    return X.build_panel(daily, imo, cross=cross_without(tk), k1=k1)


def loto_ticker16(module: str, names: list, tk: str, daily: dict, imo: pd.DataFrame, nthread: int = 4, log=None, hp: dict | None = None, k1: dict | None = None) -> pd.DataFrame:
    """Прогноз σ̂² конфигураций `names` на строках tk: панель с поперечными по 16, обучение и выбор по 16 (`loto_ticker17`)."""
    return L17.loto_ticker17(module, names, tk, nthread, log, panel16(tk, daily, imo, k1), hp)


def _inputs_sha() -> str:
    """Хэш входов и кода (Д5): SHA ежедневных рядов, панели 17, код составления панели и этого файла."""
    import hashlib
    h = hashlib.sha256()
    for f in [P.OUT / ("daily_%s.parquet" % t) for t in P.TICKERS] + [P.OUT / "daily_IMOEX.parquet", X.ENGINE_P17, X.__file__, __file__]:
        h.update(P.sha16(f).encode())
    return h.hexdigest()[:16]


def part_path(module: str, names: list, tk: str, key: str):
    return P.OUT / ("loto16_p17_part_%s_%s_%s_%s.parquet" % (module, "-".join(names), tk, key))


def part_key(module: str, names: list, tk: str, sha: str, hp=None) -> str:
    extra = "%s|%s|cross=%s|train=%s|%s" % (VERSION, tk, ",".join(cross_without(tk)), ",".join(t for t in X.TRAIN if t != tk), "+".join(names))
    return C.run_key(module, "loto16", 0, "qlike", C.Z28, hp, sha, extra)


def run_module(module: str, names: list, nthread: int = 4, log=print, reverse: bool = False, hp: dict | None = None) -> None:
    """Стадия `s17 --part loto16 [loto16-rev] --module m`: по бумаге (в прямом или обратном порядке — два процесса делят работу); `loto16_p17_<module>.parquet` — когда готовы все 17; строки реестра."""
    sha, world = _inputs_sha(), None
    for tk in sorted(X.TRAIN, reverse=reverse):
        path = part_path(module, names, tk, part_key(module, names, tk, sha, hp))
        if path.exists():
            continue
        if world is None:
            world = load_world()
        t0 = time.time()
        df = loto_ticker16(module, names, tk, world[0], world[1], nthread, None, hp, world[2])
        tmp = path.with_suffix(".tmp%d" % os.getpid())
        df.to_parquet(tmp)
        os.replace(tmp, path)
        log("LOTO16 %s %s %s: %.0f с" % (module, "+".join(names), tk, time.time() - t0))
    done = [part_path(module, names, tk, part_key(module, names, tk, sha, hp)) for tk in X.TRAIN]
    if all(p.exists() for p in done):
        pd.concat([pd.read_parquet(p) for p in done], ignore_index=True).to_parquet(P.OUT / ("loto16_p17_%s.parquet" % module))
        for n in names:
            log("LOTO16 %s %s: реестр %s" % (module, n, "добавлена" if L.register(module, n, PROGRAM16, 16, "Ж-220") else "уже была"))
