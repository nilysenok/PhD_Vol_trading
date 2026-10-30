"""Усечение календарей: retrieved_at (дата выгрузки) не обнуляется, формат строк сохраняется (лист «мелочи wf»).
Мутация: правило снова очищает retrieved_at -> тест падает."""
import datetime as dt
import importlib.util
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parent.parent / "scripts" / "server" / "usech_2021.py"
_s = importlib.util.spec_from_file_location("usech_2021_cbr", _P)
U = importlib.util.module_from_spec(_s)
_s.loader.exec_module(U)

HEAD = "date,event,source_url,retrieved_at,type,known_from,vneplan"
ROWS = [
    "2021-05-03,\"CBR board, quoted\",https://x/,2026-10-25,факт,2021-04-01,0",
    "2021-12-31,CBR meeting,https://x/,2026-10-25,план,2021-12-30,0",
    "2026-02-27,CBR meeting,https://x/,2026-10-25,план,2026-01-01,0",
    "2026-03-20,CBR meeting,https://x/,2026-10-25,план,2026-01-01,0",
]


def _koren(tmp_path: Path) -> Path:
    p = tmp_path / "k" / "data/calendar/cbr.csv"
    p.parent.mkdir(parents=True)
    p.write_text("\n".join([HEAD, *ROWS]) + "\n", encoding="utf-8")
    return tmp_path / "k"


@pytest.mark.parametrize("cut,n_keep", [(dt.date(2021, 12, 31), 2), (dt.date(2026, 2, 28), 3)])
def test_cbr_retrieved_at_i_format_sohranyayutsya(tmp_path, monkeypatch, cut, n_keep):
    monkeypatch.setattr(U, "CUT", cut)
    k = _koren(tmp_path)
    U.usech(k, tmp_path / "o", "calendar/cbr.csv")
    got = (tmp_path / "o/data/calendar/cbr.csv").read_text(encoding="utf-8").splitlines()
    assert got == [HEAD, *ROWS[:n_keep]]                    # побайтно: retrieved_at на месте, строки как в исходнике
    assert all(",2026-10-25," in r for r in got[1:])


def test_pravilo_kalendarey_ne_ochishchaet_retrieved_at():
    kal = [p for p in U.PRAVILA if p[0].startswith("data/calendar/")]
    assert len(kal) >= 6 and all("retrieved_at" not in p[3] for p in kal)
