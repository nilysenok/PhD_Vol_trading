#!/usr/bin/env python3
"""Загрузка M10 16 бумаг пула и IMOEX после 27.02.2026 (Ж-248 п. 1): только данные, без признаков, прогнозов и статистик.

Пишет ТОЛЬКО в data/fwd/mkt_m10/ (архив data/iss_check/ только читает: sha256 до и после — в манифесте).
Источник — публичный MOEX ISS candles (без ключа): бумаги TQBR, IMOEX — markets/index (как в архиве, moex_iss/src/index10.rs).
Одна сессия сети, пауза >= 0,3 с, повтор <= 3. Период: from 2026-02-20 (перекрытие) … till 2026-11-10 включительно.
  python3 scripts/dannye/fwd_mkt_m10.py        # загрузка (отказывает, если MANIFEST.json уже есть)
  python3 scripts/dannye/fwd_mkt_m10.py --self # самопроверка без сети
"""
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fwd_sber_m10 as S  # noqa: E402  (переиспользуем get_text, sha256, split, git_head, константы границы)

ROOT, OUT = S.ROOT, S.ROOT / "data/fwd/mkt_m10"
PAPERS = ("AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX", "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "TATN", "VTBR")
INSTRUMENTS = PAPERS + ("IMOEX",)
BARS_DIR, IMOEX_ARCHIVE = ROOT / "data/iss_check/bars_10m_full", ROOT / "data/iss_check/index_10m/IMOEX.parquet"
FROM, TILL, LAST_DAY = "2026-02-20", "2026-11-10", pd.Timestamp("2026-11-10")
LAST_BAR = pd.Timestamp("2026-11-10 23:59")
PAGE, COLS = S.PAGE, S.COLS
BASE = "https://iss.moex.com/iss/engines/stock/markets/"


def archive_path(secid: str) -> Path:
    return IMOEX_ARCHIVE if secid == "IMOEX" else BARS_DIR / f"{secid}.parquet"


def check_secid(secid: str) -> str:
    if secid not in INSTRUMENTS:
        raise ValueError(f"инструмент вне списка 16 бумаг + IMOEX: {secid!r}")
    return secid


def url_template(secid: str) -> str:
    check_secid(secid)
    path = "index/securities/IMOEX" if secid == "IMOEX" else f"shares/boards/TQBR/securities/{secid}"
    return BASE + path + "/candles.json?interval=10&from={from}&till={till}&start={start}"


def fetch_pages(secid: str, frm: str, till: str, get=S.get_text, page: int = PAGE) -> list:
    """Все страницы свечей: [(start, сырой текст)]; конец — пустая или короткая страница."""
    tpl, out, start = url_template(secid), [], 0
    while True:
        txt = get(tpl.format(**{"from": frm, "till": till, "start": start}) + "&iss.meta=off")
        out.append((start, txt))
        if len(json.loads(txt)["candles"]["data"]) < page:
            return out
        start += page


def pages_ok(pages: list, page: int = PAGE) -> bool:
    """Страницы идут подряд (start = 0, page, 2·page…), все кроме последней полные — потери страницы нет."""
    n = [len(json.loads(t)["candles"]["data"]) for _, t in pages]
    return [s for s, _ in pages] == [i * page for i in range(len(pages))] and all(x == page for x in n[:-1]) and n[-1] < page


def parse_pages(texts: list) -> pd.DataFrame:
    """Сырые тексты -> бары в типах архива (begin datetime64[ms], volume int64; null объёма индекса -> 0, как в архиве)."""
    rows = []
    for t in texts:
        c = json.loads(t)["candles"]
        ix = {n: i for i, n in enumerate(c["columns"])}
        rows += [[r[ix[k]] for k in COLS] for r in c["data"]]
    df = pd.DataFrame(rows, columns=COLS)
    df["begin"] = pd.to_datetime(df["begin"]).astype("datetime64[ms]")
    for k in ("open", "high", "low", "close", "value"):
        df[k] = df[k].astype("float64")
    df["volume"] = pd.to_numeric(df["volume"]).fillna(0).astype("int64")
    return df.sort_values("begin").reset_index(drop=True)


def clip_last(df: pd.DataFrame) -> pd.DataFrame:
    """Бары позже 2026-11-10 23:59 не берутся (последняя полностью завершённая сессия)."""
    return df[df["begin"] <= LAST_BAR].reset_index(drop=True)


def run(out: Path = OUT, get=S.get_text, secids=INSTRUMENTS) -> dict:
    if (out / "MANIFEST.json").exists():
        raise SystemExit("MANIFEST.json уже есть: повторная загрузка не перезаписывает")
    for s in secids:
        check_secid(s)
    sha_before = {s: S.sha256(archive_path(s)) for s in secids}
    man = {"source": {"url_template": {s: url_template(s) for s in secids}, "interval": 10, "page": PAGE},
           "request_from": FROM, "request_till": TILL, "instruments": {}}
    for secid in secids:
        t0 = dt.datetime.now().isoformat(timespec="seconds")
        pages = fetch_pages(secid, FROM, TILL, get)
        t1 = dt.datetime.now().isoformat(timespec="seconds")
        raw = out / "raw" / secid
        raw.mkdir(parents=True, exist_ok=True)
        raw_sha = {}
        for start, txt in pages:
            p = raw / f"{FROM}_{TILL}_{start}.json"
            p.write_text(txt, encoding="utf-8")
            raw_sha[p.name] = S.sha256(p)
        df = clip_last(parse_pages([t for _, t in pages]))
        fwd, ov = S.split(df)
        names = (f"{secid}_m10_fwd.parquet", f"{secid}_m10_overlap.parquet")
        fwd.to_parquet(out / names[0], index=False)
        ov.to_parquet(out / names[1], index=False)
        man["instruments"][secid] = {
            "request_time_start": t0, "request_time_end": t1, "pages": len(pages), "pages_contiguous": pages_ok(pages),
            "raw_sha256": raw_sha, "out_sha256": {n: S.sha256(out / n) for n in names},
            "fwd_first": str(fwd["begin"].min()), "fwd_last": str(fwd["begin"].max()), "fwd_rows": len(fwd),
            "overlap_first": str(ov["begin"].min()), "overlap_last": str(ov["begin"].max()), "overlap_rows": len(ov),
            "fwd_days": int(fwd["begin"].dt.normalize().nunique()),
        }
    man.update(archive_sha256_before=sha_before, archive_sha256_after={s: S.sha256(archive_path(s)) for s in secids},
               loader_sha256=S.sha256(__file__), git_head=S.git_head())
    (out / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def _self():
    def page(n):
        d = [["1", "1", "1", "1", "1", "2026-03-02 10:00:00", "1"]] * n
        return json.dumps({"candles": {"columns": ["open", "close", "high", "low", "value", "begin", "volume"], "data": d}})
    pg = {0: page(PAGE), PAGE: page(3)}
    got = fetch_pages("AFLT", "a", "b", lambda u: pg[int(u.split("start=")[1].split("&")[0])])
    assert [s for s, _ in got] == [0, PAGE] and pages_ok(got)
    try:
        url_template("SBER")
    except ValueError:
        print("self ok")


if __name__ == "__main__":
    if "--self" in sys.argv:
        _self()
    else:
        print(json.dumps(run(), ensure_ascii=False, indent=1)[:600])
