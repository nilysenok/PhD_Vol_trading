"""Загрузка M10 16 бумаг + IMOEX после 27.02.2026 (Ж-248 п. 1): тесты на синтетике, в сеть не ходят.
Фикстура — страницы в формате ISS candles (формат сверен с ответом ISS 11.11.2026), значения синтетические."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/dannye"))
import fwd_mkt_m10 as L  # noqa: E402
import fwd_mkt_m10_check as C  # noqa: E402

ISS_COLS = ["open", "close", "high", "low", "value", "volume", "begin", "end"]


def bars(start, n, price=300.0, volume=1000):
    t0 = pd.Timestamp(start)
    return [[price, price, price, price, price * volume, volume, str(t0 + pd.Timedelta(minutes=10 * i)),
             str(t0 + pd.Timedelta(minutes=10 * (i + 1)))] for i in range(n)]


def page(rows):
    return json.dumps({"candles": {"columns": ISS_COLS, "data": rows}})


def make_get(rows, drop_start=None):
    def get(url):
        start = int(url.split("start=")[1].split("&")[0])
        return page([] if start == drop_start else rows[start:start + L.PAGE])
    return get


def test_pagination_collects_all_pages_for_each_instrument_kind():
    rows = bars("2026-03-02 10:00", 2 * L.PAGE + 120)
    for secid in ("AFLT", "IMOEX"):
        pages = L.fetch_pages(secid, L.FROM, L.TILL, make_get(rows))
        assert [s for s, _ in pages] == [0, L.PAGE, 2 * L.PAGE] and L.pages_ok(pages)
        df = L.parse_pages([t for _, t in pages])
        assert len(df) == len(rows) and df["begin"].is_unique and str(df["volume"].dtype) == "int64"
        assert str(df["begin"].dtype) == "datetime64[ms]"


def test_urls_papers_tqbr_and_index_markets_index():
    assert "/shares/boards/TQBR/securities/AFLT/candles.json?interval=10" in L.url_template("AFLT")
    assert "/index/securities/IMOEX/candles.json?interval=10" in L.url_template("IMOEX")


def test_lost_page_caught():
    rows = bars("2026-03-02 10:00", 3 * L.PAGE + 10)
    good = L.fetch_pages("AFLT", "a", "b", make_get(rows))
    assert L.pages_ok(good)
    assert not L.pages_ok([good[0], good[2], good[3]])  # потеряна страница start=500
    assert not L.pages_ok([good[0], (L.PAGE, page(rows[L.PAGE:L.PAGE + 100])), good[2]])  # короткая страница в середине
    lost = L.fetch_pages("AFLT", "a", "b", make_get(rows, drop_start=L.PAGE))  # сервер отдал пустую страницу вместо второй
    assert len(L.parse_pages([t for _, t in lost])) < len(rows)
    assert not L.pages_ok(lost + [(2 * L.PAGE, page(rows[2 * L.PAGE:3 * L.PAGE]))])


def test_boundary_bar_of_0227_not_in_fwd_and_mutation_caught():
    df = L.parse_pages([page(bars("2026-02-27 23:30", 1) + bars("2026-02-27 23:40", 1) + bars("2026-03-02 10:00", 2))])
    fwd, ov = L.S.split(df)
    assert len(C.C.border_bad(fwd)) == 0 and fwd["begin"].min() == pd.Timestamp("2026-03-02 10:00") and len(ov) == 2
    mutant = df[df["begin"] >= pd.Timestamp("2026-02-27")]  # граница «≤» вместо «>»
    assert len(C.C.border_bad(mutant)) == 2


def test_duplicate_begin_caught():
    rows = bars("2026-03-02 10:00", 3)
    assert C.C.dup_count(L.parse_pages([page(rows + [rows[1]])])) == 1
    assert C.C.dup_count(L.parse_pages([page(rows)])) == 0


def test_volume_in_lots_caught_by_units_check():
    good = L.parse_pages([page(bars("2026-03-02 10:00", 5))])
    assert len(C.C.units_bad_days(good)) == 0
    assert len(C.C.units_bad_days(good.assign(volume=good["volume"] // 10))) == 1


def test_bar_after_last_day_clipped_and_mutation_caught():
    df = L.parse_pages([page(bars("2026-11-10 23:40", 1) + bars("2026-11-11 00:00", 2) + bars("2026-11-12 10:00", 1))])
    ok = L.clip_last(df)
    assert ok["begin"].max() == pd.Timestamp("2026-11-10 23:40") and len(C.after_last(ok)) == 0
    assert len(C.after_last(df)) == 3  # без обрезки проверка ловит


def test_instrument_outside_list_refused():
    for bad in ("SBER", "GAZP", "RGSS", "imoex", ""):
        with pytest.raises(ValueError):
            L.url_template(bad)
        with pytest.raises(ValueError):
            L.fetch_pages(bad, "a", "b", make_get([]))
    with pytest.raises(ValueError):
        L.run(out=Path("/nonexistent_never_written"), get=make_get([]), secids=("AFLT", "GAZP"))
    assert len(L.INSTRUMENTS) == 17 and "IMOEX" in L.INSTRUMENTS and "SBER" not in L.INSTRUMENTS


def test_index_null_volume_becomes_zero_like_archive():
    rows = bars("2026-03-02 10:00", 2)
    rows[1][5] = None
    df = L.parse_pages([page(rows)])
    assert df["volume"].tolist() == [1000, 0]


def test_availability_main_session_only_and_mutation_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT", tmp_path)
    sber = L.parse_pages([page(bars("2026-03-02 10:00", 54))])  # последний main-бар SBER 18:40
    ok = L.parse_pages([page(bars("2026-03-02 10:00", 54) + bars("2026-03-02 19:00", 3))])  # вечер не считается основной
    bad = L.parse_pages([page(bars("2026-03-02 10:00", 54))])
    bad = bad.assign(begin=bad["begin"] + pd.Timedelta(minutes=-10))  # на 10 минут раньше: не нарушает
    early_sber = L.parse_pages([page(bars("2026-03-02 10:00", 52))])  # SBER закрылся в 18:30, инструмент торгует до 18:40
    C.ROWS.clear()
    C.availability({"AFLT": ok, "MTSS": bad}, sber)
    assert [r["status"] for r in C.ROWS if "дней SBER" in r["item"]] == ["OK", "OK"]
    C.ROWS.clear()
    C.availability({"AFLT": ok}, early_sber)  # бар основной сессии после закрытия SBER — нарушение ловится
    assert [r["status"] for r in C.ROWS if "дней SBER" in r["item"]] == ["FAIL"]
    C.ROWS.clear()
