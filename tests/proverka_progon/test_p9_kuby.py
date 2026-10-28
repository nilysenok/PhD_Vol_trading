"""П-9: значения кубов второго прохода (Ж-091 п. 1.5). Красные, пока ветка `progon-L21-kuby` не слита в базу: значения коммитятся отдельно
перед вторым проходом и в PR Л-21 (#459) не входят — поэтому тест вынесен из test_p9_zayavki.py.
tejk_signal.setka (S1, S2, S5, S6) = [net, k_srednemu, pol_puti, rodnoy_uroven], yakor net.
Запуск: python3 -m pytest tests/proverka_progon/test_p9_kuby.py -q"""
from pathlib import Path

import pytest
import yaml

KUB = Path("fastbt/designs/progon")
OSI4 = ["net", "k_srednemu", "pol_puti", "rodnoy_uroven"]


def _find(o, key):
    if isinstance(o, dict):
        return [o[key]] if key in o else [x for v in o.values() for x in _find(v, key)]
    return [x for v in o for x in _find(v, key)] if isinstance(o, list) else []


@pytest.mark.parametrize("sid", ["S1", "S2", "S5", "S6"])
def test_kub_rodnoy_v_osi(sid):
    f = _find(yaml.safe_load((KUB / f"kub_{sid}.yaml").read_text(encoding="utf-8")), "tejk_signal")
    assert len(f) == 1 and list(f[0]["setka"]) == OSI4 and f[0]["yakor"] == "net"
