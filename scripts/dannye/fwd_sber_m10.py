#!/usr/bin/env python3
"""Загрузка M10 SBER (TQBR) после 27.02.2026 (Ж-247 п. 3): только данные, без прогнозов и статистик RV нового периода.

Пишет ТОЛЬКО в data/fwd/sber_m10/ (архив data/iss_check/ только читает: sha256 до и после — в манифесте).
Источник — публичный MOEX ISS candles (без ключа). Одна сессия сети, пауза >= 0,3 с, повтор <= 3.
  python3 scripts/dannye/fwd_sber_m10.py        # загрузка (отказывает, если MANIFEST.json уже есть)
Самопроверка без сети: python3 scripts/dannye/fwd_sber_m10.py --self
"""
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/iss_check/bars_10m_full/SBER.parquet"
OUT = ROOT / "data/fwd/sber_m10"
URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/SBER/candles.json"
URL_TEMPLATE = URL + "?interval=10&from={from}&till={till}&start={start}"
FROM = "2026-02-20"
OVERLAP_FROM, FWD_AFTER = pd.Timestamp("2026-02-20"), pd.Timestamp("2026-02-27 23:59")
PAGE, PAUSE, TRIES = 500, 0.3, 3
COLS = ["begin", "open", "high", "low", "close", "volume", "value"]


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def get_text(url: str) -> str:
    last = None
    for k in range(TRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                txt = r.read().decode("utf-8")
            time.sleep(PAUSE)
            return txt
        except Exception as e:  # обрыв/429/5xx: пауза и повтор
            last = e
            time.sleep(2 ** k)
    raise RuntimeError(f"GET не удался после {TRIES} попыток: {url}: {last}")


def fetch_pages(frm: str, till: str, get=get_text, page: int = PAGE) -> list:
    """Все страницы свечей: [(start, сырой текст JSON)]; страница короче `page` — конец."""
    out, start = [], 0
    while True:
        txt = get(URL_TEMPLATE.format(**{"from": frm, "till": till, "start": start}) + "&iss.meta=off")
        out.append((start, txt))
        if len(json.loads(txt)["candles"]["data"]) < page:
            return out
        start += page


def parse_pages(texts: list) -> pd.DataFrame:
    """Сырые тексты страниц -> бары в типах архива (begin datetime64[ms], volume int64)."""
    rows = []
    for t in texts:
        c = json.loads(t)["candles"]
        ix = {n: i for i, n in enumerate(c["columns"])}
        rows += [[r[ix[k]] for k in COLS] for r in c["data"]]
    df = pd.DataFrame(rows, columns=COLS)
    df["begin"] = pd.to_datetime(df["begin"]).astype("datetime64[ms]")
    for k in ("open", "high", "low", "close", "value"):
        df[k] = df[k].astype("float64")
    df["volume"] = df["volume"].astype("int64")
    return df.sort_values("begin").reset_index(drop=True)


def complete_days_only(df: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """Сегодняшний (и более поздний) день не берётся: сессия не завершена."""
    return df[df["begin"].dt.normalize() < pd.Timestamp(today)].reset_index(drop=True)


def split(df: pd.DataFrame):
    """(fwd, overlap): fwd — строго после 2026-02-27 23:59; overlap — 20…27.02."""
    fwd = df[df["begin"] > FWD_AFTER].reset_index(drop=True)
    ov = df[(df["begin"] >= OVERLAP_FROM) & (df["begin"] <= FWD_AFTER)].reset_index(drop=True)
    return fwd, ov


def git_head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def run(today: dt.date, out: Path = OUT, get=get_text) -> dict:
    if (out / "MANIFEST.json").exists():
        raise SystemExit("MANIFEST.json уже есть: повторная загрузка не перезаписывает")
    sha_before = sha256(ARCHIVE)
    till = (today - dt.timedelta(days=1)).isoformat()
    t0 = dt.datetime.now().isoformat(timespec="seconds")
    pages = fetch_pages(FROM, till, get)
    t1 = dt.datetime.now().isoformat(timespec="seconds")
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    raw_sha = {}
    for start, txt in pages:
        p = raw / f"{FROM}_{till}_{start}.json"
        p.write_text(txt, encoding="utf-8")
        raw_sha[p.name] = sha256(p)
    df = complete_days_only(parse_pages([t for _, t in pages]), today)
    fwd, ov = split(df)
    fwd.to_parquet(out / "SBER_m10_fwd.parquet", index=False)
    ov.to_parquet(out / "SBER_m10_overlap.parquet", index=False)
    man = {
        "source": {"url_template": URL_TEMPLATE, "board": "TQBR", "secid": "SBER", "interval": 10, "page": PAGE},
        "request_from": FROM, "request_till": till, "today_mac": today.isoformat(),
        "request_time_start": t0, "request_time_end": t1, "pages": len(pages), "raw_sha256": raw_sha,
        "out_sha256": {n: sha256(out / n) for n in ("SBER_m10_fwd.parquet", "SBER_m10_overlap.parquet")},
        "fwd_first": str(fwd["begin"].min()), "fwd_last": str(fwd["begin"].max()), "fwd_rows": len(fwd),
        "overlap_first": str(ov["begin"].min()), "overlap_last": str(ov["begin"].max()), "overlap_rows": len(ov),
        "archive_sha256_before": sha_before, "archive_sha256_after": sha256(ARCHIVE),
        "loader_sha256": sha256(__file__), "git_head": git_head(),
    }
    (out / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def _self():
    def page(n0, n):
        d = [["1", "1", "1", "1", "1", "2026-03-02 10:00:00", "x"]] * n
        return json.dumps({"candles": {"columns": ["open", "close", "high", "low", "value", "begin", "volume"], "data": d}})
    pg = {0: page(0, PAGE), PAGE: page(PAGE, 3)}
    got = fetch_pages("a", "b", lambda u: pg[int(u.split("start=")[1].split("&")[0])])
    assert [s for s, _ in got] == [0, PAGE]
    print("self ok")


if __name__ == "__main__":
    if "--self" in sys.argv:
        _self()
    else:
        print(json.dumps(run(dt.date.today()), ensure_ascii=False, indent=1)[:400])
