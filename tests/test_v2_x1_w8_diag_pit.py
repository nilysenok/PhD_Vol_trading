"""W8-A, PIT признаков диагностики (mutations.md № 21, 35, 44, 142; W8_SPEC 1.0, 2): z_d = ln RV^IMOEX дня d — только бары основной сессии с началом 10:00…18:30 дня d (бар 18:40 и бары d + 1 не входят);
PIT-терцили — пороги строго по дням до d (день d и будущее не входят; ≥ 60 прошлых); плановая длина перерыва L_d — по календарю, известному на d 18:40, реализованный t⁺ в неё не входит; объявление
календаря после d не отдаётся на d. Известные ответы на синтетике + мутации исходника (копия модуля, боевой файл не меняется) и monkeypatch (№ 43). Реальные файлы только читаются."""
from __future__ import annotations

import hashlib
import inspect
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w8_diag_core as K
from models_v2.s16 import x1_w8_diag_feat as FE
from models_v2.s16 import x1_w8_reg as R


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (R.TRIALS, R.CKPT) if p.exists()}
        s.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(R.REP.glob("W8_*")) if p.is_file()})
        s["W8"] = sorted(str(p) for p in R.W8.rglob("*")) if R.W8.exists() else []
        return s
    before = st()
    yield
    assert st() == before


def mutant(mod, old: str, new: str) -> types.ModuleType:
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    m = types.ModuleType(mod.__name__ + "_mutant")
    m.__file__ = mod.__file__
    exec(compile(src.replace(old, new), mod.__file__, "exec"), m.__dict__)
    return m


# ---------- PIT-терцили (№ 21/44) ----------

def oracle_terc(v) -> list:
    out = []
    for i in range(len(v)):
        past = [a for a in v[:i] if np.isfinite(a)]
        if len(past) < 60 or not np.isfinite(v[i]):
            out.append("na")
            continue
        q1, q2 = np.percentile(past, 100 / 3), np.percentile(past, 200 / 3)
        out.append("T1" if v[i] <= q1 else "T2" if v[i] <= q2 else "T3")
    return out


def assert_terc_known(f) -> None:
    v = np.r_[np.arange(60.0), 19.7, 19.6, 39.4, 39.3, np.nan, 5.0, np.full(40, 500.0)]  # q1 = 59/3 = 19,667; q2 = 118/3 = 39,333 по первым 60; хвост 500 — «будущее»
    g = list(f(v))
    assert g[:60] == ["na"] * 60                                                          # меньше 60 прошлых — na (в том числе i = 59)
    assert g[60] == "T2"                                                                  # 19,7 > q1 (день d в порог НЕ входит: с ним q1 = 19,7 → T1)
    assert g[61] == "T1"                                                                  # 19,6 ≤ q1 (с учётом 19,7 в прошлом порог не меняется заметно)
    assert g[62] == "T3" and g[64] == "na" and g[65] in ("T1", "T2", "T3")                # 39,4 > q2; NaN — na; после NaN считается по конечным прошлым
    assert g == oracle_terc(v)


def test_terc_past_known_answer_and_matches_independent_oracle() -> None:
    assert_terc_known(K.terc_past)
    rng = np.random.default_rng(3)
    v = rng.normal(size=300)
    v[[70, 71, 150]] = np.nan
    assert list(K.terc_past(v)) == oracle_terc(v) and K.terc_past(v[:60]).tolist() == ["na"] * 60
    assert K.MINPAST == 60 and K.terc_past(v, minpast=100)[101] == "na" and K.terc_past(v, minpast=100)[102] != "na"      # 2 NaN в прошлом: 100 конечных наступает на строке 102


@pytest.mark.parametrize("old,new", [
    ("past = v[:i][np.isfinite(v[:i])]", "past = v[:i + 1][np.isfinite(v[:i + 1])]"),    # порог с днём d (№ 21/44)
    ("past = v[:i][np.isfinite(v[:i])]", "past = v[np.isfinite(v)]"),                    # порог по всей выборке
    ("past = v[:i][np.isfinite(v[:i])]", "past = v[:i]"),                                # NaN в прошлом не отброшены
])
def test_terc_mutations_are_caught(old, new) -> None:
    with pytest.raises(AssertionError):
        assert_terc_known(mutant(K, old, new).terc_past)


# ---------- бары IMOEX дня d (№ 142) ----------

def day_bars(day: str, first: float, upto: str = "18:30", extra=()) -> pd.DataFrame:
    """Бары каждые 10 минут от 10:00 до `upto` включительно; цена каждого бара — детерминированная; `extra` — доп. (время, open, close)."""
    t = pd.date_range(day + " 10:00", day + " " + upto, freq="10min")
    c = first * np.exp(0.0007 * np.sin(np.arange(len(t)) * 1.3) + 0.0003 * np.arange(len(t)) / len(t))
    o = np.r_[first * 0.996, c[:-1]]                                                      # первый бар открывается с гэпом: ln(close/open) ≠ 0
    b = pd.DataFrame(dict(begin=t, open=o, close=c))
    if not extra:
        return b
    return pd.concat([b, pd.DataFrame([(pd.Timestamp(day + " " + h), a, z) for h, a, z in extra], columns=["begin", "open", "close"])], ignore_index=True)


def expected_day(b: pd.DataFrame) -> tuple:
    b = b[(b["begin"].dt.time >= pd.Timestamp("10:00").time()) & (b["begin"].dt.time <= pd.Timestamp("18:30").time())].sort_values("begin")
    c, o = b["close"].to_numpy(), b["open"].to_numpy()
    lr = np.r_[np.log(c[0] / o[0]), np.diff(np.log(c))]
    return float((lr ** 2).sum()), float(c[-1])


ARCH = pd.concat([day_bars("2024-03-04", 3000.0, extra=[("09:50", 2000.0, 2500.0), ("18:40", 3000.0, 3300.0), ("18:50", 3300.0, 3400.0), ("19:00", 3400.0, 3500.0), ("23:40", 3500.0, 3600.0)]),
                  day_bars("2024-03-05", 3010.0, extra=[("18:40", 3010.0, 2700.0)])], ignore_index=True)
FWD = pd.concat([day_bars("2024-03-05", 9999.0), day_bars("2024-03-06", 3020.0)], ignore_index=True)      # 03-05 в продолжении — дубль дня архива (отбрасывается); 03-06 — новый


def fake_read(monkeypatch, arch=ARCH, fwd=FWD) -> None:
    def rp(path, columns=None, **kw):
        return (arch if "index_10m" in str(path) else fwd).copy()
    monkeypatch.setattr(FE.pd, "read_parquet", rp)


def assert_imoex_known(f) -> None:
    im = f()
    d1, d2, d3 = (pd.Timestamp(s) for s in ("2024-03-04", "2024-03-05", "2024-03-06"))
    assert list(im.index) == [d1, d2, d3]
    for d, b in ((d1, ARCH[ARCH["begin"].dt.normalize() == d1]), (d2, ARCH[ARCH["begin"].dt.normalize() == d2]), (d3, FWD[FWD["begin"].dt.normalize() == d3])):
        rv, close = expected_day(b)
        assert im.at[d, "rv"] == pytest.approx(rv, rel=1e-12, abs=0) and im.at[d, "close"] == close
    assert im.at[d2, "r"] == pytest.approx(np.log(im.at[d2, "close"] / im.at[d1, "close"]), abs=1e-15) and np.isnan(im.at[d1, "r"])


def test_imoex_day_known_answer_session_window_and_continuation(monkeypatch) -> None:
    fake_read(monkeypatch)
    assert_imoex_known(FE.imoex_day)
    rv = FE.imoex_day()["rv"]
    assert rv.iloc[0] < 1e-4 and rv.iloc[0] > 0                                           # бары 09:50, 18:40…23:40 (скачок 10 %) не вошли: иначе rv ≥ 0,009


@pytest.mark.parametrize("mod,old,new", [
    (FE, "(tod <= CUT_BAR)", "(tod <= CUT_BAR + pd.Timedelta(minutes=10))"),              # бар с началом 18:40 в z_d
    (FE, "(tod >= pd.Timedelta(hours=10))", "(tod >= pd.Timedelta(hours=9))"),             # бар до основной сессии
    (FE, "(tod <= CUT_BAR)", "(tod <= pd.Timedelta(hours=23, minutes=50))"),               # вечерняя сессия
    (FE, "n[n[\"begin\"] > a[\"begin\"].max()]", "n"),                                      # продолжение не обрезано: дубль дня архива
    (FE, "b[\"day\"] = b[\"begin\"].dt.normalize()", "b[\"day\"] = (b[\"begin\"] - pd.Timedelta(days=1)).dt.normalize()"),      # бары дня d отнесены ко дню d − 1 (и d + 1 попадает в d)
    (FE, "first = b.groupby(\"day\").head(1).index", "first = b.groupby(\"day\").head(0).index"),    # первый бар дня без ln(close/open)
])
def test_imoex_day_mutations_are_caught(monkeypatch, mod, old, new) -> None:
    fake_read(monkeypatch)
    with pytest.raises(AssertionError):
        assert_imoex_known(mutant(mod, old, new).imoex_day)


def test_imoex_day_ignores_bars_of_next_day_and_1840(monkeypatch) -> None:
    """Порча баров дня d + 1 и баров d после 18:30 не меняет rv, close, r дня d."""
    fake_read(monkeypatch)
    base = FE.imoex_day()
    a2 = ARCH.copy()
    minutes = a2["begin"].dt.hour * 60 + a2["begin"].dt.minute
    d1, d2 = pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")
    late = (a2["begin"].dt.normalize() == d1) & (minutes > 18 * 60 + 30)
    nxt = a2["begin"].dt.normalize() == d2
    assert late.sum() == 4 and nxt.sum() > 50
    a2.loc[late, ["open", "close"]] *= 5.0
    a2.loc[nxt, "close"] *= 5.0
    fake_read(monkeypatch, arch=a2)
    alt = FE.imoex_day()
    assert alt.loc[d1].equals(base.loc[d1]) and alt.at[d2, "rv"] != base.at[d2, "rv"]       # день d не изменился, порча действует (день d + 1 изменился)


# ---------- признаки на реальных рядах: PIT ----------

@pytest.fixture(scope="module")
def real():
    x = K.load()
    return x, FE.imoex_day()


def groups(monkeypatch, x, im) -> dict:
    monkeypatch.setattr(FE, "imoex_day", lambda: im)
    return FE.features(x)[0]


def corrupted(im: pd.DataFrame, D) -> pd.DataFrame:
    c = im.copy()
    later = c.index > D
    c.loc[later, ["rv", "r"]] = c.loc[later, ["rv", "r"]] * 50.0                         # будущее IMOEX испорчено
    return c


def leaks(monkeypatch, x, im, shift: bool = False) -> bool:
    ds = [pd.Timestamp(s) for s in ("2019-06-28", "2020-03-31", "2021-01-29", "2021-09-30", "2022-06-30", "2023-12-28")]
    f = (lambda e: e.shift(-1)) if shift else (lambda e: e)                              # shift: мутация «в день d отдан день d + 1»
    ref = groups(monkeypatch, x, f(im))
    for D in ds:
        alt = groups(monkeypatch, x, f(corrupted(im, D)))
        m = (x["d"] <= D).to_numpy()
        if any(not np.array_equal(np.asarray(ref[k])[m], np.asarray(alt[k])[m]) for k in ("W8D7_imoex_absr", "W8D8_imoex_rv")):
            return True
    return False


def test_features_use_only_days_up_to_d_and_next_day_mutation_is_caught(monkeypatch, real) -> None:
    x, im = real
    assert leaks(monkeypatch, x, im) is False                                             # порча IMOEX после D не меняет группы дней ≤ D
    assert leaks(monkeypatch, x, im, shift=True) is True                                  # мутация № 142: день d получает бары d + 1


def test_features_imoex_groups_equal_independent_oracle_and_threshold_with_day_d_is_caught(monkeypatch, real) -> None:
    x, im = real
    f = groups(monkeypatch, x, im)
    irv = im["rv"].reindex(pd.DatetimeIndex(x["d"])).to_numpy()
    ir = im["r"].reindex(pd.DatetimeIndex(x["d"])).to_numpy()
    assert list(f["W8D8_imoex_rv"]) == oracle_terc(np.log(np.where(irv > 0, irv, np.nan)))        # z_d = ln RV^IMOEX дня d, пороги по дням < d
    assert list(f["W8D7_imoex_absr"]) == oracle_terc(np.abs(ir))
    bad = mutant(K, "past = v[:i][np.isfinite(v[:i])]", "past = v[:i + 1][np.isfinite(v[:i + 1])]").terc_past
    monkeypatch.setattr(FE, "terc_past", bad)                                             # мутация: порог с днём d
    g = FE.features(x)[0]
    assert any(not np.array_equal(f[k], g[k]) for k in ("W8D4_lvl", "W8D5_shock", "W8D7_imoex_absr", "W8D8_imoex_rv"))


# ---------- плановая длина перерыва (№ 35) ----------

def cal_of(rows: list) -> pd.DataFrame:
    """rows: (date, kind, announced_at | None, confirmed)."""
    return pd.DataFrame(dict(date=pd.to_datetime([r[0] for r in rows]), kind=[r[1] for r in rows], reason="тест",
                             announced_at=pd.to_datetime([pd.Timestamp(r[2]) if r[2] else pd.NaT for r in rows]), source_url="u", source_kind="t", confirmed=[r[3] for r in rows]))


def plan(d, rows, canon=()):
    L, fm = K.planned([pd.Timestamp(d)], cal_of(rows), {pd.Timestamp(c) for c in canon})
    return int(L[0]), int(fm[0])


def assert_planned_known(planned) -> None:
    def p(d, rows, canon=()):
        L, fm = planned([pd.Timestamp(d)], cal_of(rows), {pd.Timestamp(c) for c in canon})
        return int(L[0]), int(fm[0])
    assert p("2024-01-05", []) == (2, 0) and p("2024-01-08", []) == (0, 0)                # пятница → понедельник: L = 2, будних неторговых нет
    assert p("2024-05-07", [("2024-05-08", "weekday_closed", "2024-01-01", True)]) == (1, 5)       # будний праздник, объявлен заранее
    assert p("2024-05-07", [("2024-05-08", "weekday_closed", "2024-05-07 18:40", True)]) == (1, 5)        # ровно 18:40 — известно
    assert p("2024-05-07", [("2024-05-08", "weekday_closed", "2024-05-07 18:41", True)]) == (0, 0)        # объявлен после d 18:40 — не известен (№ 35)
    assert p("2024-05-07", [("2024-05-08", "weekday_closed", "2024-05-08 09:00", True)]) == (0, 0)
    assert p("2024-05-07", [("2024-05-08", "weekday_closed", "2024-01-01", False)]) == (0, 0)             # неподтверждённый
    rows = [("2024-05-01", "weekday_closed", "2024-01-01", True), ("2024-05-02", "weekday_closed", "2024-01-01", True), ("2024-05-03", "weekday_closed", "2024-01-01", True)]
    assert p("2024-04-30", rows) == (5, 5)                                                # ср, чт, пт + сб, вс; месяц первого буднего неторгового — май
    assert p("2024-04-30", rows[:2] + [("2024-05-03", "weekday_closed", "2024-04-30 20:00", True)]) == (2, 5)      # пятница объявлена после d: для d она торговая, цепочка (ср, чт) обрывается → L = 2
    sat = ("2024-04-27", "weekend_trading", "2024-04-01", True)
    assert p("2024-04-26", [sat], canon=["2024-04-27"]) == (0, 0)                         # рабочая суббота объявлена и каноническая: пятница → суббота без перерыва
    assert p("2024-04-26", [("2024-04-27", "weekend_trading", "2024-04-26 18:41", True)], canon=["2024-04-27"]) == (2, 0)      # объявлена после d: суббота неторговая (№ 35)
    assert p("2024-04-26", [sat], canon=[]) == (2, 0)                                     # не каноническая — день торговым не делает
    assert p("2024-04-27", [sat], canon=["2024-04-27"]) == (1, 0)                         # d = рабочая суббота → воскресенье


def test_planned_known_answer_on_real_code() -> None:
    assert_planned_known(K.planned)


def test_planned_mutations_are_caught(monkeypatch) -> None:
    monkeypatch.setattr(K.GC, "known", lambda r, d: bool(r["confirmed"]))                # мутация № 35: объявление после d отдано на d
    with pytest.raises(AssertionError):
        assert_planned_known(K.planned)
    monkeypatch.undo()
    monkeypatch.setattr(K.GC, "known", lambda r, d: True)                                # мутация: подтверждение не требуется
    with pytest.raises(AssertionError):
        assert_planned_known(K.planned)


def test_planned_does_not_take_realized_t_plus_and_refuses_over_30_days() -> None:
    assert list(inspect.signature(K.planned).parameters) == ["d_index", "cal", "canon"]   # реализованного t⁺ среди входов нет
    closed = [((pd.Timestamp("2024-06-03") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"), "weekday_closed", "2024-01-01", True) for i in range(40) if (pd.Timestamp("2024-06-03") + pd.Timedelta(days=i)).dayofweek < 5]
    with pytest.raises(SystemExit):
        plan("2024-06-02", closed)


def test_gap_type_known_answer() -> None:
    d = pd.to_datetime(["2024-01-05", "2024-05-07", "2024-04-30", "2024-04-27", "2024-01-08", "2024-01-03", "2024-01-04"])
    L = np.array([2, 1, 5, 1, 0, 2, 2])
    assert K.gap_type(d, L).tolist() == ["weekend", "hol1", "long", "worksat", "none", "other", "other"]     # среда/четверг с L = 2 — «иное»
    assert K.gap_type(pd.to_datetime(["2024-05-10"]), np.array([3])).tolist() == ["long"]


def test_real_load_types_gap_equals_planned_and_one_pair_differs_from_realized(real) -> None:
    x, _ = real
    assert len(x) == 2198 and ((x["L"] >= 1).astype(int) == x["gap"]).all()               # L ≥ 1 ⇔ gap^PIT во всех 2198 парах
    assert x.loc[x["gap"] == 1, "type"].value_counts().to_dict() == {"weekend": 415, "long": 33, "hol1": 27, "worksat": 7}
    assert (x["delta_real"] > 0).astype(int).ne(x["gap"]).sum() == 1                      # по реализованному t⁺ метка отличалась бы ровно в одной паре (не вход модели)
    m = x["delta_real"] > 0
    assert (x.loc[m, "L"] == x.loc[m, "delta_real"]).all()                                # плановая L = реализованной Δ_d везде, где перерыв был
