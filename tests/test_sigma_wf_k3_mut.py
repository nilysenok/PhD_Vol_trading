"""К-3, тестировщик #800: мутации, которых нет в tests/test_sigma_wf_k3.py. Синтетика без сети и без данных.

Мутации: «ночь от воскресенья» (бары ДСВД выкинуты из дня, но остались в «закрытии предыдущего дня»),
«выкинута только суббота 01.11.2025», «граница 2025 → 2024» (набор ДСВД обрезан с 2024), «конец ровно 2026-02-03 вырезан»,
«chitat не режет строки позже konec».
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "scripts/dannye"), str(R / "scripts/analysis")]
import wf_intraday_bez_dsvd as K  # noqa: E402
import test_sigma_wf_k3 as A  # noqa: E402


def _sobrat_noch_ot_voskresenya():
    """Мутант: intraday_one видит бары ДСВД (закрытие воскресенья), а строки ДСВД потом выбрасываются."""
    r = K.T.intraday_one(A.bary(), "SBER")
    return r[~r.date.isin(A.DSVD)].set_index("date")


def test_mutaciya_noch_ot_voskresenya_lovitsya():
    r = _sobrat_noch_ot_voskresenya()
    assert "2025-03-02" not in [x.strftime("%Y-%m-%d") for x in r.index]  # по дате мутант чист
    with pytest.raises(AssertionError):
        A.proverit(r)


def test_mutaciya_tolko_subbota_20251101_lovitsya():
    def bez_subboty(c, ds, k):
        d = pd.to_datetime(c["begin"]).dt.normalize()
        return c[(d <= k) & ~d.isin(ds) & (d != pd.Timestamp("2025-11-01"))]
    with pytest.raises(AssertionError):
        A.proverit(A.sobrat(filtr=bez_subboty))


def test_mutaciya_dsvd_s_2024_lovitsya():
    """Набор ДСВД расширен выходными с 2024: рабочая суббота 27.04.2024 пропадает."""
    def shire(c, ds, k):
        d = pd.to_datetime(c["begin"]).dt.normalize()
        return c[(d <= k) & ~d.isin(ds) & ~((d >= "2024-01-01") & (d.dt.dayofweek == 5) & (d != "2025-11-01"))]
    with pytest.raises(AssertionError):
        A.proverit(A.sobrat(filtr=shire))


def test_mutaciya_konec_strogo_menshe_lovitsya():
    def strogo(c, ds, k):
        d = pd.to_datetime(c["begin"]).dt.normalize()
        return c[(d < k) & ~d.isin(ds)]
    r = A.sobrat(filtr=strogo)
    assert r.index.max() < pd.Timestamp("2026-02-03")  # 03.02 пропал
    assert pd.Timestamp("2026-02-03") in A.sobrat().index  # а в исходном есть


def test_chitat_rezhet_pozzhe_konec(tmp_path, monkeypatch):
    b = A.bary()
    b["begin"] = b["begin"].astype(str)
    b.to_parquet(tmp_path / "SBER.parquet")
    monkeypatch.setattr(K.T, "BARS", tmp_path)
    out = K.chitat(["SBER"])["SBER"]
    assert pd.to_datetime(out["begin"]).max() < pd.Timestamp("2026-02-04")
    assert pd.to_datetime(out["begin"]).dt.normalize().max() == pd.Timestamp("2026-02-03")
    # мутант: konec сдвинут на день — строки 04.02 протекают
    assert pd.to_datetime(K.chitat(["SBER"], pd.Timestamp("2026-02-04"))["SBER"]["begin"]).max() >= pd.Timestamp("2026-02-04")
