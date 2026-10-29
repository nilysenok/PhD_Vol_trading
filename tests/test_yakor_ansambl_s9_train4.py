"""A_zashch (ж-124 п. 1): ансамбль s9 (ApproachA, ежегодный top-10 walk-forward с голосованием) на train4 воспроизводит
эталон — позиции из кэша s9 (data/fastbt/s9/cache_latest_train4.txt, sha1 кэша пишется в сообщение) — с допуском 1e-12.
Подмножество (объявлено): все 17 тикеров × 6 Sk, годы 2016–2019 (подмножества нет; мутация — AFLT/S3).
Мутация: ансамбль подменён одной фиксированной точкой (первая точка top-10 2016 года на все годы — аналог A_fix) —
позиции расходятся, тест обязан это видеть. Модульных подмен нет (mutations.md № 43)."""
import dataclasses
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastbt import s9  # noqa: E402

from fastbt.data import TICKERS as TICKERY  # noqa: E402

SID, TOL, TICKER = "S3", 1e-12, "AFLT"


def _kesh() -> tuple[dict, str]:
    ptr = s9.OUT / "cache_latest_train4.txt"
    if not ptr.exists():
        pytest.skip(f"нет {ptr}: кэш s9 train4 не собран")
    p = s9.OUT / ptr.read_text().strip()
    return pickle.loads(p.read_bytes()), hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def _ansambl(fiksirovat: bool, ticker: str = TICKER, sid: str = SID) -> dict:
    """{Y: позиции} по тому же коду, что s9._job (ApproachA → turnover.apply). fiksirovat — мутация A_fix-типа."""
    from fastbt import dev, masks, turnover
    from fastbt import design as dsg
    from fastbt.approach_a import ApproachA
    from fastbt.data import load_ticker
    from fastbt.unit import make_unit
    with s9.period_ctx("train4"):
        d, _ = dev.load_design(s9.MAIN_YAML)
        cfg = s9.period_cfg(dev.make_config(d, s9.SCOPE))
        dsg.apply(cfg.design)
        td = load_ticker(ticker, cfg.corp_actions)
        u = make_unit(ticker, sid, td)
        aa = ApproachA(u, cfg)
        a = aa.run()
        if fiksirovat:
            tochka = a[min(a)][0][:1]
            a = {Y: (tochka, aa.ensemble(tochka)) for Y in a}
        masks._CACHE.clear()
        return {Y: v[1].astype(np.float64) for Y, v in turnover.apply(a, u, cfg, "A").items()}


def _maks_delta(poz: dict, eta: dict) -> float:
    assert set(poz) == set(eta), (sorted(poz), sorted(eta))
    return max(float(np.abs(poz[Y] - eta[Y]).max()) for Y in eta)


@pytest.mark.parametrize("ticker", TICKERY)
@pytest.mark.parametrize("sid", s9.SIDS)
def test_ansambl_s9_train4_raven_kesh(ticker, sid):
    kesh, h = _kesh()
    eta = kesh[ticker]["A"][sid]
    assert sorted(eta) == [2016, 2017, 2018, 2019] and any(np.abs(v).sum() > 0 for v in eta.values())
    d = _maks_delta(_ansambl(False, ticker, sid), eta)
    assert d <= TOL, f"ансамбль s9 ≠ эталон кэша {h}: макс |Δ| = {d:.2e}"


def test_mutaciya_fiksirovannaya_tochka_vmesto_ansamblya_lovitsya():
    kesh, h = _kesh()
    eta = kesh[TICKER]["A"][SID]
    mut = _ansambl(True)
    d = _maks_delta(mut, eta)
    assert d > TOL, f"A_fix-подмена не отличима от эталона {h}: тест её не ловит"
    assert sum(int((mut[Y] != eta[Y]).sum()) for Y in eta) > 0
