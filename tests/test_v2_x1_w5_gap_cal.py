"""W5-B, метка gap^PIT (W5_SPEC 11.2–11.5; mutations.md № 1, 35): известный ответ на синтетическом календаре — пятница → понедельник даёт 1; праздник среды, объявленный до d 18:40, — 1,
после — 0; неподтверждённый будний день — 0; торговая суббота, объявленная до d, даёт пятнице 0, неподтверждённая — 1; метка не зависит от реализованного t⁺ (gap^real и длина Δ_d в метку не входят);
три пометки 11.5 в шапке выхода — дословно из спецификации. Реальные файлы только читаются (autouse-хэши)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w5_gap as G
from models_v2.s16 import x1_w5_gap_cal as CAL

REAL = (CAL.GAP_PIT, CAL.CAL, G.REG.TRIALS, G.REG.CKPT)
MARKS_SPEC = ("состояние gap выбрано из 24 после анализа истории окна M", "способ оценки β выбран после просмотра 24.02.2022 (Д-2)",
              "исследовательский результат на раскрытой истории, не новое независимое подтверждение (Ж-250 п. 6)")


def real_state() -> dict:
    s = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}
    s["W5"] = sorted(str(p) for p in G.OUT.rglob("*")) if G.OUT.exists() else []
    return s


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before


def cal_of(rows: list) -> pd.DataFrame:
    """rows: (date, kind, announced_at | None, confirmed)."""
    return pd.DataFrame(dict(date=pd.to_datetime([r[0] for r in rows]), kind=[r[1] for r in rows], reason="тест",
                             announced_at=pd.to_datetime([r[2] for r in rows]), source_url="u", source_kind="t", confirmed=[r[3] for r in rows]))


WED, TUE = "2024-05-08", pd.Timestamp("2024-05-07")                                 # d = вторник, d + 1 = среда
SAT, FRI = "2024-04-27", pd.Timestamp("2024-04-26")                                 # d = пятница, d + 1 = суббота
EMPTY = cal_of([])


def one(d, rows, canon=frozenset()) -> int:
    return int(CAL.gap_pit([pd.Timestamp(d)], cal_of(rows), set(canon))["gap"].iloc[0])


def test_friday_to_monday_is_one_and_ordinary_days_are_zero() -> None:
    g = CAL.gap_pit(pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-06", "2024-01-07"]), EMPTY, set())
    assert g["gap"].tolist() == [1, 0, 0, 0, 0, 1, 0]                               # пт → 1; пн…чт → 0; суббота → воскресенье 1; воскресенье → понедельник 0


@pytest.mark.parametrize("announced,confirmed,want", [
    ("2024-01-01 10:00", True, 1),                                                  # объявлено заранее
    ("2024-05-07 18:40", True, 1),                                                  # ровно в 18:40 — известно
    ("2024-05-07 18:41", True, 0),                                                  # после d 18:40
    ("2024-05-07 19:00", True, 0),
    ("2024-05-08 09:00", True, 0),                                                  # объявлено в день неторгового дня
    ("2024-05-07 23:59:59", True, 0),                                                # дата без времени (загрузчик ставит конец дня): со следующего дня
    ("2024-05-06 23:59:59", True, 1),                                               # дата накануне: известна
    ("2024-01-01 10:00", False, 0),                                                 # неподтверждённый будний день — неизвестен, gap = 0
    (None, True, 0),                                                                # без даты объявления
])
def test_weekday_holiday_known_answer(announced, confirmed, want) -> None:
    assert one(TUE, [(WED, "weekday_closed", announced, confirmed)]) == want


def test_unconfirmed_weekday_marks_unconfirmed_date_in_source_columns() -> None:
    g = CAL.gap_pit([TUE], cal_of([(WED, "weekday_closed", "2024-01-01 10:00", False)]), set())
    assert g["gap"].iloc[0] == 0 and g["unconfirmed_date"].iloc[0] == pd.Timestamp(WED)
    g2 = CAL.gap_pit([TUE], cal_of([(WED, "weekday_closed", "2024-01-01 10:00", True)]), set())
    assert g2["gap"].iloc[0] == 1 and pd.isna(g2["unconfirmed_date"].iloc[0])


@pytest.mark.parametrize("announced,confirmed,canon,want", [
    ("2024-01-10 12:00", True, {pd.Timestamp(SAT)}, 0),                             # торговая суббота объявлена до d → пятница не перед перерывом
    ("2024-04-26 18:40", True, {pd.Timestamp(SAT)}, 0),
    ("2024-04-26 18:41", True, {pd.Timestamp(SAT)}, 1),                             # объявлена после d 18:40 → считается неторговой
    ("2024-01-10 12:00", False, {pd.Timestamp(SAT)}, 1),                            # неподтверждённая → неторговая, gap = 1
    ("2024-01-10 12:00", True, set(), 1),                                           # строка не каноническая (сессия выходного дня): день торговым не делает
])
def test_trading_saturday_known_answer(announced, confirmed, canon, want) -> None:
    assert one(FRI, [(SAT, "weekend_trading", announced, confirmed)], canon) == want


def test_saturday_without_row_is_one_and_working_saturday_to_monday_is_one() -> None:
    assert one(FRI, []) == 1
    sat_work = cal_of([(SAT, "weekend_trading", "2024-01-10 12:00", True)])
    assert int(CAL.gap_pit([pd.Timestamp(SAT)], sat_work, {pd.Timestamp(SAT)})["gap"].iloc[0]) == 1      # рабочая суббота → воскресенье → понедельник: d + 1 = воскресенье → 1 (11.4)


def test_label_depends_only_on_d_and_calendar_not_on_realised_t_plus_or_delta() -> None:
    """11.1/11.2: t⁺ неизвестен на выпуске. Подмена t⁺ не меняет gap^PIT; gap^real меняется; длина Δ в метку не входит (метка ∈ {0, 1})."""
    cal = cal_of([(WED, "weekday_closed", "2024-05-07 19:00", True)])               # объявлено поздно: на d неизвестно
    d = pd.to_datetime(["2024-05-07", "2024-05-03"])
    x1 = pd.DataFrame(dict(d=d, t_plus=pd.to_datetime(["2024-05-09", "2024-05-06"])))      # t⁺ = через среду; пятница → понедельник
    x2 = pd.DataFrame(dict(d=d, t_plus=pd.to_datetime(["2024-05-08", "2024-05-06"])))      # t⁺ = среда (если бы торговали)
    t1, t2 = CAL.pit_table(x1, cal, set()), CAL.pit_table(x2, cal, set())
    assert t1["gap_pit"].tolist() == t2["gap_pit"].tolist() == [0, 1]
    assert t1["gap_real"].tolist() == [1, 1] and t2["gap_real"].tolist() == [0, 1]
    assert t1["agree"].tolist() == [0, 1] and "только по факту" in t1["note"].iloc[0]
    long = CAL.pit_table(pd.DataFrame(dict(d=pd.to_datetime(["2024-12-27"]), t_plus=pd.to_datetime(["2025-01-09"]))), EMPTY, set())       # Δ = 12 дней
    assert long["delta_days"].iloc[0] == 12 and long["gap_pit"].iloc[0] == 1 and set(long["gap_pit"]) <= {0, 1}


def labels_ok(pit_table_fn) -> bool:
    """Детектор: метка бинарна, одинакова при любом t⁺ и равна известному ответу. Применяется к боевой `pit_table` и к мутантам."""
    cal = cal_of([(WED, "weekday_closed", "2024-05-07 19:00", True)])
    d = pd.to_datetime(["2024-05-07", "2024-05-03"])
    a = pit_table_fn(pd.DataFrame(dict(d=d, t_plus=pd.to_datetime(["2024-05-09", "2024-05-06"]))), cal, set())["gap_pit"]
    b = pit_table_fn(pd.DataFrame(dict(d=d, t_plus=pd.to_datetime(["2024-05-08", "2024-05-06"]))), cal, set())["gap_pit"]
    return a.tolist() == b.tolist() == [0, 1] and set(a) <= {0, 1}


def test_detector_passes_real_pit_table_and_catches_gap_real_and_delta_length_mutants() -> None:
    assert labels_ok(CAL.pit_table)

    def mut_real(x, cal, canon):                                                    # мутация: gap^real вместо gap^PIT
        t = CAL.pit_table(x, cal, canon)
        t["gap_pit"] = t["gap_real"]
        return t

    def mut_delta(x, cal, canon):                                                   # мутация: длина Δ_d в метке
        t = CAL.pit_table(x, cal, canon)
        t["gap_pit"] = t["delta_days"]
        return t

    assert not labels_ok(mut_real) and not labels_ok(mut_delta)


def test_real_table_label_is_binary_equals_calendar_and_models_use_only_pit(monkeypatch) -> None:
    r = G.compute()
    tbl, g = r["tbl"], r["series"]["W5-G"]
    assert len(tbl) == 2198 and set(tbl["gap_pit"]) <= {0, 1} and g["gap"].tolist() == tbl["gap_pit"].tolist()
    again = CAL.gap_pit(r["x"]["d"], r["cal"], r["canon"])["gap"].to_numpy(int)
    assert again.tolist() == tbl["gap_pit"].tolist()                                # метка пересчитывается из d и календаря
    assert (tbl.loc[tbl["gap_pit"] == 1, "source"] != "").all()                     # источник у каждой пары с gap = 1 (11.4)
    assert (tbl["agree"] == (tbl["gap_pit"] == tbl["gap_real"]).astype(int)).all()
    orig = CAL.pit_table

    def mut(x, cal, canon):                                                         # мутация: модель получает gap^real
        t = orig(x, cal, canon)
        t["gap_pit"] = t["gap_real"]
        return t

    monkeypatch.setattr(CAL, "pit_table", mut)
    assert G.compute()["series"]["W5-G"]["gap"].tolist() == tbl["gap_real"].tolist() != g["gap"].tolist() or int((tbl["agree"] == 0).sum()) == 0   # подмена видна, если расхождения есть


def test_header_has_three_marks_from_spec_verbatim() -> None:
    assert tuple(m.lstrip("# ") for m in CAL.MARKS) == MARKS_SPEC
    r = G.compute()
    head = CAL.header(r["tbl"], r["cal"], r["canon"]).splitlines()
    assert [h.lstrip("# ") for h in head[:3]] == list(MARKS_SPEC)
    assert np.all([h.startswith("#") for h in head])
    if CAL.GAP_PIT.exists():
        assert CAL.GAP_PIT.read_text(encoding="utf-8").splitlines()[:3] == head[:3]


def test_calendar_loader_refuses_duplicate_wrong_kind_and_wrong_weekday(tmp_path) -> None:
    head = "date,kind,reason,announced_at,source_url,source_kind,confirmed\n"
    for name, body in (("dup", "2024-05-08,weekday_closed,r,2024-01-01 10:00,u,t,yes\n2024-05-08,weekday_closed,r,2024-01-01 10:00,u,t,yes\n"),
                       ("kind", "2024-05-08,holiday,r,2024-01-01 10:00,u,t,yes\n"), ("wd", "2024-04-27,weekday_closed,r,2024-01-01 10:00,u,t,yes\n"),
                       ("wd2", "2024-05-08,weekend_trading,r,2024-01-01 10:00,u,t,yes\n")):
        p = tmp_path / (name + ".csv")
        p.write_text(head + body, encoding="utf-8")
        with pytest.raises(SystemExit):
            CAL.load_calendar(p)
    ok = tmp_path / "ok.csv"
    ok.write_text(head + "2024-05-08,weekday_closed,r,2024-05-07,u,t,yes\n2024-04-27,weekend_trading,r,,u,t,no\n", encoding="utf-8")
    c = CAL.load_calendar(ok)
    assert c.loc[c["date"] == pd.Timestamp(WED), "announced_at"].iloc[0] == pd.Timestamp("2024-05-07 23:59:59")      # только дата — конец дня
    assert pd.isna(c.loc[c["date"] == pd.Timestamp(SAT), "announced_at"].iloc[0]) and not bool(c.loc[c["date"] == pd.Timestamp(SAT), "confirmed"].iloc[0])
    assert one(TUE, [(WED, "weekday_closed", c["announced_at"].iloc[1], True)]) == 0         # дата без времени → до d 18:40 не известна
