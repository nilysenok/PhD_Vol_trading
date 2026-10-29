"""Р-4 (29.10): независимость дневного excess от stop_nach у тренда S3/S4 на стоп-заявке при trail = net (находка #679, V6-12b).

Расследование (причина названа, код не правился):
  (а) начальный стоп на стоп-заявке СТАВИТСЯ (RunStop._levels берёт Run2._levels и добавляет канал xl/xs) — отвергнута;
  (б) канал ближе стопа σ → выход "kanal" и независимость законна, но только когда уровень канала строже всех трёх стопов — ручная лента ниже;
  (в) stop_nach доходит до Params.b (eval_v5_osi._vyhod: k_s → pt4["k_s"]) — отвергнута (число выходов "stop" растёт при уменьшении σ);
  (г) быстрый слой даёт те же сделки, что эталон (лента ниже, оба пути) — отвергнута;
  (д) ПРИЧИНА: размер сделки по рангу (m_sily ≠ "1") берёт ранг на d_sig, а у тренда d_sig — любой вечер постановки пары, не «день старта»
      сигнала; ранг определён только на днях старта (EvaluatorV5._ranki) → вес 0 → excess ≡ 0 при любом stop_nach. При rang_tip = tikernyj
      ранга нет вовсе (нужно ≥ 20 прошлых стартов тикера за 252 дня): на S3, точка Соболя № 199, 0 из 465 сделок с рангом, x ≡ 0.
Мутации: канал выключен в RunStop (xl = nan) — лента «канал строже» падает; stop_nach не доходит до b — лента «стоп раньше канала» падает.
python3 -m pytest tests/test_stop_nach_trail_net.py -q"""
import numpy as np
import pytest

from fastbt import exec_e, fast_exec
from fastbt.exec_e_types import Day, Params

N0, NAN = 15, float("nan")


def _day(o, h, l, c):
    return Day(np.array(o, float), np.array(h, float), np.array(l, float), np.array(c, float), False)


def _flat():
    return _day([100, 100], [101, 101], [99, 99], [100, 100])


def _trade(b, xl, f):
    """День 15: buy-stop 102 исполнен по 102 (σ = 0,01); день 16: open 102, low 99. xl — уровень канала (заявка выхода дня 15)."""
    p = Params(vhod_tip="stop_zayavka", stop_touch=True, exit_signal=False, comm=0.0, b=b)
    days = [_flat() for _ in range(N0)] + [_day([102], [103], [101.2], [102.5]), _day([102], [102], [99], [100])]
    n = len(days)
    u = {k: np.full(n, NAN) for k in ("up", "dn", "xl", "xs")}
    u["up"][N0 - 1], u["dn"][N0 - 1], u["xl"][N0] = 102.0, 98.0, xl
    return f("T", days, np.zeros(n), np.full(n, 0.01), p, urovni=u).trades[0]


BOTH = [exec_e.run, fast_exec.run_v5]


@pytest.mark.parametrize("f", BOTH)
def test_stop_ranshe_kanala_stop_nach_menyaet_vyhod(f):
    """Канал далеко (90): σ-стоп 1σ / 2σ срабатывает, 4σ (97,92) нет — выходы различаются; эталон = быстрое ядро."""
    t1, t2, t4 = (_trade(b, 90.0, f) for b in (1.0, 2.0, 4.0))
    assert (t1.reason, t2.reason, t4.reason) == ("stop", "stop", None)
    assert t1.exit_px == pytest.approx(102 * 0.99) and t2.exit_px == pytest.approx(102 * 0.98)


@pytest.mark.parametrize("f", BOTH)
def test_kanal_strozhe_vsekh_stopov_nezavisimost_zakonna(f):
    """Канал 101,5 строже всех трёх стопов (100,98 / 99,96 / 97,92): выход "kanal" по 101,5 при любом stop_nach — законная независимость."""
    ts = [_trade(b, 101.5, f) for b in (1.0, 2.0, 4.0)]
    assert {(t.reason, t.exit_px) for t in ts} == {("kanal", 101.5)}


# --- реальные данные train4 (2016–2019), S3; пропуск без дневного ряда ---
def _est_dannye() -> bool:
    from fastbt import daily_1850
    return daily_1850.out_path().exists()


real = pytest.mark.skipif(not _est_dannye(), reason="нет дневного ряда 18:50 в FASTBT_MAIN")
_S: dict = {}


def _sreda():
    if not _S:
        from fastbt import canon_v5, progon_sobol as ps
        rz = ps.Rezhim("v5", "train4", "seed", False, canon_v5.load_kub("S3")["fdr"])
        _S.update(sr=ps.fabrika_real("S3", rz), rz=rz, P=dict(ps.tochki_v5(ps.load_kub("S3", rz), 0, 200, rz)))
    return _S


def _posch(i, **izm):
    """Точка Соболя № i S3 с изменёнными осями → (excess портфеля, сделки)."""
    from fastbt import progon_sobol as ps
    s = _sreda()
    row, x, _ = ps.schitat_tochku(s["sr"], "S3", i, {**s["P"][i], **izm}, "r4", False, False, rz=s["rz"])
    assert row["status"] == "ok", row["prichina"]
    return np.asarray(x), [t for r in s["sr"].evr._last[4] for t in r.trades]


@real
def test_m_sily_1_stop_nach_menyaet_ryad_i_chislo_stopov():
    """Точка № 199 (trail = net), размер без ранга: ряды при σ 1/2/4 различны, выходов "stop" строго меньше при росте σ."""
    res = [_posch(199, m_sily="1", stop_nach=f"sigma:{b}") for b in ("1.0", "2.0", "4.0")]
    assert not np.array_equal(res[0][0], res[1][0]) and not np.array_equal(res[1][0], res[2][0])
    n = [sum(t.reason == "stop" for t in tr) for _, tr in res]
    assert n[0] > n[1] > n[2] > 0


@real
def test_trend_rang_obshchiy_ne_obnulyaet_excess():
    """Ф-ранг: вес — по рангу дня старта текущего сигнала (eval_v5.rang_starta); до правки excess ≡ 0 при сделках."""
    x, trades = _posch(199, m_sily="kvadr", rang_tip="obshchiy")
    assert len(trades) > 0 and np.any(x != 0)


@real
@pytest.mark.xfail(strict=True, reason="Ф-ранг: tikernyj на точке 199 ранга нет вовсе (< 20 стартов тикера за 252 дня) — вес 0; "
                                       "а вход тренда ранговым запретом не гейтится (уровни, не sig). Нужно решение дирижёра")
def test_trend_rang_tikernyj_ne_obnulyaet_excess():
    x, trades = _posch(199, m_sily="kvadr", rang_tip="tikernyj")
    assert len(trades) > 0 and np.any(x != 0)
