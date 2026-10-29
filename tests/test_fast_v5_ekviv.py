"""П4 сетки v5 (Ж-127): оси Л5 — единицы уровней ed_s/ed_t ∈ {sigma, atr, pct} и Chandelier (Params.chand) — в быстром слое.
Эталон — медленный путь `exec_e.run`; быстрый `fast_etap1.run` обязан совпасть ПОБАЙТНО (допуск 0, proverka.md п. 3).
1) синтетика: 600 точек × 3 ленты (на каждое значение оси ≥ 200 точек; дни pusto, дивиденды, close Н6, тик);
2) реальные 5-мин бары train4 (2016–2019, AFLT/SBER/LKOH; FASTBT_BARS1M — 1-минутки D25), 300 точек × 3 тикера
   (на ось ed_s, ed_t, chand — ≥ 200 точек; σ — std 20 дней по close, сигнал — блоки −1/0/1, дивиденды — случайные);
3) #488: режим, неизвестный ядру (поле Params вне FAST_POLYA, новое поле, лишний вход run), уходит в эталон;
   мутация «прежний список исключений» обязана ловиться;
4) мутации медленного пути (monkeypatch): ATR дня d в Chandelier, единица стопа = σ̂ при atr, храповик снят — сверка расходится.
FASTBT_MAIN=$PWD FASTBT_BARS1M=<основная копия>/data/regen/D25/bars1m FASTBT_BARS5M_CACHE=<scratch> python3 -m pytest tests/test_fast_v5_ekviv.py -q
"""
import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from fastbt import exec_e, exec_e_run, fast_etap1
from fastbt.exec_e_types import Day, Params
from test_fast_etap1_smoke import _days, _same

N_PT, N_LENT, N_REAL = 600, 3, 300
EDS = ("atr", "pct", "sigma")
TIKERY = ("AFLT", "SBER", "LKOH")


def _ch(rng, *v):
    return v[int(rng.integers(len(v)))]


def tochka(i: int, tick=None, osi=None) -> Params:
    """Точка i: ed_s = EDS[i % 3], ed_t = EDS[i // 3 % 3], chand — при нечётном i // 9 (на значение оси ≥ 200 точек из 600);
    osi = (ed_s, ed_t, chand вкл) — задать оси явно."""
    rng = np.random.default_rng(20261029 + i)
    es, et, ch = osi or (EDS[i % 3], EDS[i // 3 % 3], (i // 9) % 2 == 1)
    sb = (0.01, 0.02, 0.04) if es == "pct" else (1.0, 2.0)
    ta = (0.01, 0.03) if et == "pct" else (1.0, 2.0, 3.0)
    b = _ch(rng, *sb) if ch or rng.random() < 0.8 else None
    R = _ch(rng, None, 1.5) if b is not None else None
    k = _ch(rng, None, 0.25, 0.5, 1.0)
    check = "bary" if ch else _ch(rng, "bary", "bary", "zakrytie")
    return Params(k=k, f=_ch(rng, "propusk", "zakrytie_d1", "povtor"), a=None if R else _ch(rng, None, *ta), b=b,
                  tau=_ch(rng, None, 3, 5), s=_ch(rng, "fiks", "ezhedn"), comm=0.0004, exit_signal=_ch(rng, True, False),
                  side=_ch(rng, 1, -1), tick=_ch(rng, None, 0.0, tick), stop_touch=_ch(rng, True, False), R=R,
                  h=4.0 if R else None, k_tr=_ch(rng, None, sb[0] * 1.5) if b is not None else None,
                  be=_ch(rng, False, True) if b is not None else False, check=check, reentry=_ch(rng, True, False),
                  vyhod_lim=bool(k is not None and check == "bary" and rng.random() < 0.3),
                  ed_s=es, ed_t=et, chand=_ch(rng, 1.0, 2.0, 3.0) if ch else None)


def sintetika(seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 100))
    days = _days(rng, n)
    sig = np.repeat(rng.choice([-1, 0, 1], n // 3 + 1), 3)[:n].astype(float)
    sigma = rng.uniform(0.005, 0.03, n)
    sigma[rng.random(n) < 0.05] = np.nan
    close = np.array([d.c[-1] * (1 + rng.normal(0, 0.002)) if not d.pusto else 1.0 for d in days])
    div = np.where(rng.random(n) < 0.05, rng.uniform(0.5, 3.0, n), 0.0)
    return days, sig, sigma, (close if rng.random() < 0.7 else None), (div if rng.random() < 0.5 else None)


def sverit(days, sig, sigma, p, close, div, run_ref=None) -> bool:
    """True — эталон и быстрый слой равны побайтно (или оба отказали тем же текстом)."""
    ref = run_ref or exec_e.run
    try:
        a = ref("T", days, sig, sigma, p, close=close, div=div)
    except ValueError as e:
        with pytest.raises(ValueError) as ei:
            fast_etap1.run("T", days, sig, sigma, p, close=close, div=div)
        return str(ei.value) == str(e)
    b = fast_etap1.run("T", days, sig, sigma, p, close=close, div=div)
    try:
        _same(a, b)
    except AssertionError:
        return False
    return True


def test_sintetika_bystryj_ravno_etalonu_po_osyam():
    plokho, deistvuet = [], {"atr_s": 0, "pct_s": 0, "atr_t": 0, "pct_t": 0, "chand": 0}
    for i in range(N_PT):
        p = tochka(i, tick=0.01)
        assert not fast_etap1._slow_only(p, None), p           # ось перенесена: точка идёт быстрым путём
        for t in range(N_LENT):
            days, sig, sigma, close, div = sintetika(10_000 * t + i)
            if not sverit(days, sig, sigma, p, close, div):
                plokho.append((i, t))
            _deistvie(deistvuet, days, sig, sigma, p, close, div)
    assert not plokho, plokho[:10]
    assert all(v >= 30 for v in deistvuet.values()), deistvuet      # ось реально меняет ленту (не «ничего не делает»)


def _deistvie(cnt, days, sig, sigma, p, close, div):
    """Счётчик: точка с осью ≠ умолчанию даёт иную ленту, чем та же точка в умолчании оси."""
    for key, rep in (("atr_s", dict(ed_s="sigma")), ("pct_s", dict(ed_s="sigma")), ("atr_t", dict(ed_t="sigma")),
                     ("pct_t", dict(ed_t="sigma")), ("chand", dict(chand=None))):
        on = (p.ed_s == key[:3]) if key.endswith("_s") else (p.ed_t == key[:3]) if key.endswith("_t") else p.chand is not None
        if not on:
            continue
        try:
            a = exec_e.run("T", days, sig, sigma, p, close=close, div=div)
            b = exec_e.run("T", days, sig, sigma, dataclasses.replace(p, **rep), close=close, div=div)
        except ValueError:
            continue
        cnt[key] += a.gross.tobytes() != b.gross.tobytes()


# --- реальные бары train4 (≤ 2019) ---
OSN_BARS1M = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D25/bars1m")   # основная копия, только чтение


def _realnye(t: str) -> tuple:
    from fastbt import bars5m
    src = Path(os.environ.get("FASTBT_BARS1M", bars5m.SRC if bars5m.SRC.exists() else OSN_BARS1M))
    if not (src / f"{t}_2016.csv").exists():
        pytest.fail(f"нет 1-минуток D25 {src} — задать FASTBT_BARS1M (без skip, Ж-081)")
    b = bars5m.load_bars5m(t, "2016-01-01", "2019-12-31", "s_predotkrytiem", src=src)
    b = b[~b["pusto"]]
    days = [Day(*g[["open", "high", "low", "close"]].to_numpy(float).T, False)
            for _, g in b.groupby(b["date"].astype("datetime64[ns]").dt.normalize())]
    c = np.array([d.c[-1] for d in days])
    r = np.r_[np.nan, np.diff(np.log(c))]
    sigma = np.array([np.nanstd(r[max(1, i - 19):i + 1], ddof=1) if i >= 20 else np.nan for i in range(len(c))])
    return days, sigma


@pytest.fixture(scope="module")
def realnye():
    return {t: _realnye(t) for t in TIKERY}


def osi_real(i: int) -> tuple:
    """300 точек: ed_s ≠ sigma во всех (atr/pct по 150), ed_t ≠ sigma в 200, chand — в 200: на ось ≥ 200 точек × 3 тикера."""
    return ("atr", "pct")[i % 2], ("pct", "atr", "sigma")[i // 2 % 3], i % 3 != 2


_REAL: dict = {}


def _real_tochka(i: int) -> list:
    """Точка i на всех тикерах: [(i, тикер, быстрый путь?, равно эталону?)]."""
    rng = np.random.default_rng(7 + i)
    p = tochka(i, tick=0.01, osi=osi_real(i))
    out = []
    for t, (days, sigma) in _REAL.items():
        n = len(days)
        sig = np.repeat(rng.choice([-1, 0, 1], n // 5 + 1), 5)[:n].astype(float)
        div = np.where(rng.random(n) < 0.004, rng.uniform(0.5, 3.0, n), 0.0)
        out.append((i, t, not fast_etap1._slow_only(p, None), sverit(days, sig, sigma, p, None, div)))
    return out


def test_realnye_bary_train4_bystryj_ravno_etalonu(realnye):
    """Эталон на 4 годах — 0,2 с на вызов: точки считаются в 4 процессах fork (бары общие, копия при записи)."""
    import multiprocessing as mp
    _REAL.update(realnye)
    with mp.get_context("fork").Pool(4) as pool:
        rows = [r for part in pool.map(_real_tochka, range(N_REAL), chunksize=10) for r in part]
    assert len(rows) == N_REAL * len(TIKERY)
    assert all(r[2] for r in rows), "точка осей Л5 ушла на медленный путь"
    plokho = [(i, t) for i, t, _, ok in rows if not ok]
    assert not plokho, plokho[:10]
    assert sum(osi_real(i)[2] for i in range(N_REAL)) >= 200 and sum(osi_real(i)[1] != "sigma" for i in range(N_REAL)) >= 200


# --- #488: неизвестный режим — только эталоном ---
def neizvestnye_ne_v_etalon(slow_only) -> list:
    """Список режимов, которые slow_only отдал бы ядру, хотя ядро их не знает (пусто — правило выполнено)."""
    bad, base = [], Params(b=1.0)
    for f in dataclasses.fields(Params):
        if f.name not in fast_etap1.FAST_POLYA and not slow_only(dataclasses.replace(base, **{f.name: object()}), None):
            bad.append(f.name)
    Nov = dataclasses.make_dataclass("Nov", [("nova_os", str, dataclasses.field(default="net"))], bases=(Params,), frozen=True)
    if not slow_only(Nov(b=1.0, nova_os="vkl"), None):
        bad.append("novoe_pole")
    for kw in (dict(ed_s="kvantil"), dict(f="novyj"), dict(s="nedelya")):
        if not slow_only(dataclasses.replace(base, **kw), None):
            bad.append(str(kw))
    if not slow_only(base, None, np.zeros(3)):
        bad.append("lishnij_vhod")
    return bad


def test_neizvestnyj_rezhim_uhodit_v_etalon():
    assert neizvestnye_ne_v_etalon(fast_etap1._slow_only) == []
    assert not fast_etap1._slow_only(Params(b=1.0, ed_s="atr", ed_t="pct", chand=2.0), None)


def test_mutaciya_488_spisok_isklyuchenij_lovitsya():
    """Мутация: прежний `_slow_only` (перечень известных новых осей вместо списка полей ядра) — новое поле проходит в ядро."""
    def staryj(p, ed, *extra):
        return ed is not None or p.tejk != "net" or p.vhod_tip != "podtverzhdenie" or p.atr_m is not None or p.chast_1r \
            or exec_e._new_axes(p, ed)
    assert neizvestnye_ne_v_etalon(staryj), "мутант не пойман"


# --- мутации медленного пути: сверка обязана разойтись ---
def _mut_atr_dnya_d(mp):
    mp.setattr(exec_e_run, "atr_for_day", lambda atr, ix, j: atr[ix[j]] if atr is not None else float("nan"))


def _mut_ed_kak_sigma(mp):
    orig = exec_e.unit_series
    mp.setattr(exec_e, "unit_series", lambda days, close, kind, sigma, atr: None if kind == "atr" else orig(days, close, kind, sigma, atr))


def _mut_bez_hrapovika(mp):
    mp.setattr(exec_e_run, "ratchet", lambda sd, S, prev: S)


@pytest.mark.parametrize("mut", [_mut_atr_dnya_d, _mut_ed_kak_sigma, _mut_bez_hrapovika])
def test_mut_yadro_lovitsya(mut, monkeypatch):
    mut(monkeypatch)
    razosh = 0
    for i in range(N_PT):
        p = tochka(i, tick=0.01)
        if p.chand is None and p.ed_s != "atr":
            continue
        days, sig, sigma, close, div = sintetika(i)
        razosh += not sverit(days, sig, sigma, p, close, div)
        if razosh >= 3:
            break
    assert razosh >= 3, f"мутант {mut.__name__} не пойман"
