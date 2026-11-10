"""Загрузка M10 SBER после 27.02.2026 (Ж-247 п. 3): тесты на синтетике, в сеть не ходят.
Фикстура — страницы в формате ISS candles (columns/data; формат сверен с ответом ISS 11.11.2026), значения синтетические."""
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/dannye"))
import fwd_sber_m10 as L  # noqa: E402
import fwd_sber_m10_check as C  # noqa: E402

ISS_COLS = ["open", "close", "high", "low", "value", "volume", "begin", "end"]


def bars(start, n, price=300.0, volume=1000, step_min=10):
    t0 = pd.Timestamp(start)
    return [[price, price, price, price, price * volume, volume, str(t0 + pd.Timedelta(minutes=step_min * i)),
             str(t0 + pd.Timedelta(minutes=step_min * (i + 1)))] for i in range(n)]


def page(rows):
    return json.dumps({"candles": {"columns": ISS_COLS, "data": rows}})


def make_get(rows, size=L.PAGE):
    def get(url):
        start = int(url.split("start=")[1].split("&")[0])
        return page(rows[start:start + size])
    return get


def test_pagination_collects_all_pages():
    rows = bars("2026-03-02 00:00", 2 * L.PAGE + 120)
    pages = L.fetch_pages("2026-02-20", "2026-03-05", make_get(rows))
    assert [s for s, _ in pages] == [0, L.PAGE, 2 * L.PAGE]
    df = L.parse_pages([t for _, t in pages])
    assert len(df) == len(rows) and df["begin"].is_unique and str(df["volume"].dtype) == "int64"
    assert str(df["begin"].dtype) == "datetime64[ms]"


def test_pagination_exact_multiple_asks_empty_last_page():
    rows = bars("2026-03-02 00:00", L.PAGE)
    pages = L.fetch_pages("a", "b", make_get(rows))
    assert [s for s, _ in pages] == [0, L.PAGE] and len(L.parse_pages([t for _, t in pages])) == L.PAGE


def test_boundary_bar_of_0227_not_in_fwd_and_mutation_caught():
    df = L.parse_pages([page(bars("2026-02-27 23:30", 1) + bars("2026-02-27 23:40", 1) + bars("2026-03-02 10:00", 2))])
    fwd, ov = L.split(df)
    assert len(C.border_bad(fwd)) == 0 and fwd["begin"].min() == pd.Timestamp("2026-03-02 10:00")
    assert len(ov) == 2
    mutant = df[df["begin"] >= pd.Timestamp("2026-02-27")]  # граница «≤» вместо «>»
    assert len(C.border_bad(mutant)) == 2


def test_duplicate_begin_caught():
    rows = bars("2026-03-02 10:00", 3)
    df = L.parse_pages([page(rows + [rows[1]])])
    assert C.dup_count(df) == 1
    assert C.dup_count(L.parse_pages([page(rows)])) == 0


def test_volume_in_lots_caught_by_units_check():
    good = L.parse_pages([page(bars("2026-03-02 10:00", 5))])
    lots = good.assign(volume=good["volume"] // 10)  # лоты по 10 акций: value/volume/close = 10
    assert len(C.units_bad_days(good)) == 0
    assert len(C.units_bad_days(lots)) == 1


def test_today_unfinished_session_not_taken():
    df = L.parse_pages([page(bars("2026-11-10 23:30", 1) + bars("2026-11-11 00:00", 2) + bars("2026-11-12 10:00", 1))])
    out = L.complete_days_only(df, dt.date(2026, 11, 11))
    assert out["begin"].dt.normalize().unique().tolist() == [pd.Timestamp("2026-11-10")]
    leak = df[df["begin"].dt.normalize() <= pd.Timestamp("2026-11-11")]  # мутация: сегодня взят
    assert len(leak) > len(out)


def test_session_labels_and_proxy_days():
    df = L.parse_pages([page(bars("2026-03-02 09:50", 3) + bars("2026-03-02 19:00", 1) + bars("2026-03-07 10:00", 1))])
    assert C.session_of(df["begin"]).tolist() == ["utro", "osnov", "osnov", "vecher", "vyhodnoy"]
    assert C.proxy_days(df) == {pd.Timestamp("2026-03-02")}
