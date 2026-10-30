"""Свод Д8 на период walk-forward (2022 – 2026-02): годность по хостам source_url, усечение ≤ 2021 даёт прежний свод.
Синтетика; реальный свод читается только файлом. Мутация: «строки 2022+ годны без подтверждения»."""
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts/server"))
import usech_2021  # noqa: E402

_spec = importlib.util.spec_from_file_location("div_svod_wf", REPO / "scripts" / "div_svod.py")
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)

HDR = ["ticker", "record_date", "last_cum_date", "announce_date", "dps_rub", "source_url", "accessed"]
SHA_2021 = "60ca0d822af653d7c8349645301dbfddb31c6d6ee2cb935a368565dacf158074"
PERV = "https://www.novatek.ru/ru/investors/dividends/payments/ (ГОСА) ; https://www.dohod.ru/ik/analytics/dividend/nvtk"
AGR = "https://www.dohod.ru/ik/analytics/dividend/nvtk ; https://investmint.ru/nvtk/divs/"


def _zapis(papka, name, rows):
    with open(papka / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(rows)


def _svod(tmp_path, url, ann="2023-03-01", rec="2023-04-20", dps="10", name="2026-10-29_dividendy_2025_2026.csv"):
    _zapis(tmp_path, name, [["NVTK", rec, "2023-04-19", ann, dps, url, "2026-10-29"]])    # дата не важна: фильтра по году нет
    return ds.svod(tmp_path)


def test_2023_pervichnoe_raskrytie_godna(tmp_path):
    (r,) = _svod(tmp_path, PERV)
    assert (r["godna"], r["prichina"], r["announce_pit"]) == (1, "", "2023-03-01")


def test_2023_odin_agregator_ne_godna(tmp_path):
    (r,) = _svod(tmp_path, "https://www.dohod.ru/ik/analytics/dividend/nvtk")
    assert (r["godna"], r["prichina"]) == (0, "tolko_agregatory")


def test_dva_agregatora_ne_nezavisimy(tmp_path):
    (r,) = _svod(tmp_path, AGR)
    assert r["godna"] == 0


def test_novost_plyus_agregator_godna(tmp_path):
    (r,) = _svod(tmp_path, "https://www.interfax.ru/business/1 ; https://www.dohod.ru/x")
    assert r["godna"] == 1


SV = ["ticker", "record_date", "announce_date", "dps_rub", "src_tablica", "source2_url", "record_date_2", "announce_date_2",
      "dps_rub_2", "tip_istochnika", "verified", "primechanie"]


def _sverka(tmp_path, verified, name="2026-10-30_div_sverka_wf_P.csv", src="2022_2024"):
    with open(tmp_path / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SV)
        w.writerow(["NVTK", "2023-04-20", "2023-03-01", "10", src, "", "2023-04-20", "2023-03-05", "10", "vtoroy", verified, ""])


def test_2022_2024_bez_sverki_ne_godna_dazhe_s_pervichnoj_ssylkoj(tmp_path):
    (r,) = _svod(tmp_path, PERV, name="2026-10-29_dividendy_2022_2024.csv")
    assert (r["godna"], r["prichina"]) == (0, "net_sverki_wf")


def test_sverka_wf_podhvatyvaetsya_i_zamenyaet_syruyu_stroku(tmp_path):
    _zapis(tmp_path, "2026-10-29_dividendy_2022_2024.csv", [["NVTK", "2023-04-20", "x", "2023-03-01", "10", AGR, "x"]])
    _sverka(tmp_path, "yes")
    (r,) = ds.svod(tmp_path)
    assert (r["godna"], r["announce_pit"]) == (1, "2023-03-05")      # более поздняя из дат
    _sverka(tmp_path, "net_istochnika")
    (r,) = ds.svod(tmp_path)
    assert r["godna"] == 0


def test_sverka_2025_2026_zamenyaet_pravilo_po_hostam(tmp_path):
    _zapis(tmp_path, "2026-10-29_dividendy_2025_2026.csv", [["NVTK", "2023-04-20", "x", "2023-03-01", "10", PERV, "x"]])
    _sverka(tmp_path, "net_istochnika", name="2026-10-30_div_sverka_wf_R.csv", src="2025_2026")
    (r,) = ds.svod(tmp_path)
    assert r["godna"] == 0


def test_pustaya_ssylka_ne_godna(tmp_path):
    (r,) = _svod(tmp_path, "")
    assert (r["godna"], r["prichina"]) == (0, "net_istochnika")


def test_net_daty_obyavleniya_ne_godna(tmp_path):
    (r,) = _svod(tmp_path, PERV, ann="")
    assert (r["godna"], r["prichina"]) == (0, "net_daty_obyavleniya")


def test_dva_platezha_v_odin_reestr_ne_skleivayutsya(tmp_path):
    _zapis(tmp_path, "2026-10-29_dividendy_2022_2024.csv",
           [["LKOH", "2022-12-21", "2022-12-20", "2022-10-28", "537", PERV, "x"],
            ["LKOH", "2022-12-21", "2022-12-20", "2022-11-02", "256", PERV, "x"]])
    assert sorted(float(r["dps_rub"]) for r in ds.svod(tmp_path)) == [256.0, 537.0]


def test_vyplat_net_ne_idet_v_svod(tmp_path):
    assert _svod(tmp_path, PERV, dps="0") == []


def test_realnyj_svod_usechenie_2021_dayot_prezhnij(tmp_path):
    cp = tmp_path / "svod.csv"
    usech_2021.usech_csv(REPO / "docs/issledovaniya/div_sverka_svod.csv", cp, ("record_date", "announce_pit"))
    assert hashlib.sha256(cp.read_bytes()).hexdigest() == SHA_2021


def test_realnyj_svod_pokryvaet_do_2026_02_i_godnye_po_godam():
    rows = list(csv.DictReader(open(REPO / "docs/issledovaniya/div_sverka_svod.csv", encoding="utf-8")))
    god = {y: sum(int(r["godna"]) for r in rows if r["record_date"][:4] == y) for y in map(str, range(2022, 2027))}
    assert max(r["record_date"] for r in rows) <= "2026-02-28" and "2026-01-12" in {r["record_date"] for r in rows}
    assert [god[y] for y in sorted(god)] == [12, 16, 21, 16, 2]


def test_mutaciya_2022_plyus_godny_bez_podtverzhdeniya_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "godna_wf", lambda url: "")
    (r,) = _svod(tmp_path, "https://www.dohod.ru/ik/analytics/dividend/nvtk")
    assert r["godna"] == 1                            # мутация действует: подложенное правило пропускает агрегатор
    monkeypatch.undo()
    (r,) = _svod(tmp_path, "https://www.dohod.ru/ik/analytics/dividend/nvtk")
    with pytest.raises(AssertionError):               # честное правило её ловит
        assert r["godna"] == 1
