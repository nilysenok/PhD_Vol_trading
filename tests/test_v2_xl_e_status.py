"""Волна E: статус гибрида после R8, но без допуска R9 (admitted = False) — не ACCEPT; допущенный и REJECT не меняются."""
import pandas as pd

from models_v2.s6.e import st_e4

NA = "R8 пройдено; не допущен (R9, критерий C)"


def frames(admitted):
    ar = pd.DataFrame({"id": ["A1", "A2", "B1q", "XJ"], "role": ["гибрид", "гибрид", "гибрид", "самостоятельная"], "accept": [True, True, False, True],
                       "verdict": ["ACCEPT", "ACCEPT", "REJECT", "ACCEPT"], "verdict_full": ["ACCEPT", "ACCEPT", "REJECT", "ACCEPT"]})
    sel = pd.DataFrame({"id": ["A1", "A2", "B1q"], "admitted": admitted})
    return ar, sel


def test_not_admitted_hybrid_is_not_accept():
    out = st_e4.relabel_not_admitted(*frames([False, True, False]))
    v = out.set_index("id")
    assert v.loc["A1", "verdict"] == NA and v.loc["A1", "verdict_full"] == NA and v.loc["A1", "accept"]
    assert v.loc["A2", "verdict"] == "ACCEPT"          # допущен — остаётся
    assert v.loc["B1q", "verdict"] == "REJECT"          # не прошёл R8 — остаётся REJECT
    assert v.loc["XJ", "verdict"] == "ACCEPT"           # самостоятельная — вне R9
