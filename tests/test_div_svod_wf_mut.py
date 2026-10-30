"""Мутация (д) PR #730: усечение свода ≤ 2021 оставляет строку, объявленную в 2022 (проверяется только первая дата).
Свод усекается по обеим датам (record_date, announce_pit): строка с реестром ≤ 2021 и объявлением 2022+ не едет в пакет."""
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts/server"))
import usech_2021  # noqa: E402

COLS = ["ticker", "record_date", "dps_rub", "announce_pit", "godna", "prichina"]


def _usech(tmp_path, rows):
    src, dst = tmp_path / "s.csv", tmp_path / "o.csv"
    with open(src, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLS)
        w.writerows(rows)
    st = usech_2021.usech_csv(src, dst, ("record_date", "announce_pit"))
    return st, list(csv.DictReader(open(dst, encoding="utf-8")))


def test_obyavleno_v_2022_otbrasyvaetsya_dazhe_pri_reestre_2021(tmp_path):
    st, out = _usech(tmp_path, [["AAA", "2021-12-30", "5", "2022-01-05", "1", ""],
                                ["BBB", "2021-12-30", "5", "2021-12-01", "1", ""],
                                ["CCC", "2022-02-01", "5", "2022-01-05", "1", ""]])
    assert [r["ticker"] for r in out] == ["BBB"] and st.otbrosheno == 2


def test_granica_31_dekabrya_2021_vklyuchitelna(tmp_path):
    _, out = _usech(tmp_path, [["AAA", "2021-12-31", "5", "2021-12-31", "1", ""],
                               ["BBB", "2022-01-01", "5", "2021-12-31", "1", ""]])
    assert [r["ticker"] for r in out] == ["AAA"]
