"""Шапки таблиц реверса р.1 (proverka.md п. 4): у каждой таблицы — «# коммит: <sha1>» и «# вход: <путь> sha256=<хэш>» на каждый вход.
Писатели таблиц (b, е, в, ж-затвор, a_cd/v_a_zashch/автокорреляция через r1_a_cd.shapka) вызываются на синтетике: данных Z2 не нужно, входы — временные файлы.
Мутация: хэш убран из строки входа / путь заменён именем — проверка шапки падает. Модульных подмен нет (mutations.md № 43): env и зона — monkeypatch."""
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import r1_a_cd as CD  # noqa: E402
import r1_b  # noqa: E402
import r1_e  # noqa: E402
import r1_v  # noqa: E402
import r1_zh_zashch as ZH  # noqa: E402
import shapka_lib as SL  # noqa: E402
from fastbt import sigma_z2, zona  # noqa: E402

DF = pd.DataFrame({"a": [1.0, 2.0]})


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def sreda(tmp_path, monkeypatch):
    """Z2-среда из временных файлов; дивиденды — NET (таблицы дивидендов репозитория сюда не нужны)."""
    dev, z2, dn = tmp_path / "dev", tmp_path / "z2", tmp_path / "dn"
    for k in (dev, z2, dn):
        k.mkdir()
    vh = []
    for i, h in enumerate(sigma_z2.HORIZONS):
        for p in (sigma_z2.dev_path(h, dev), sigma_z2.z2_path(h, z2)):
            p.write_bytes(f"{p.name}-{i}".encode())
            vh.append(p)
    daily, imoex = dn / "ohlcv.parquet", dn / "IMOEX.parquet"
    daily.write_bytes(b"daily"), imoex.write_bytes(b"imoex")
    monkeypatch.setenv("FASTBT_DEV_SIGMA", str(dev))
    monkeypatch.setenv("FASTBT_Z2_R2FINAL", str(z2))
    monkeypatch.setenv("FASTBT_Z2_DAILY", str(daily))
    monkeypatch.setenv("FASTBT_Z2_IMOEX", str(imoex))
    monkeypatch.setenv("FASTBT_Z2_DIV", "NET")
    monkeypatch.setattr(zona, "_zone", "test1")
    return vh + [daily, imoex]


def proverit_shapku(text: str, vhody: list[Path]) -> None:
    sh = [s for s in text.splitlines() if s.startswith("#")]
    assert any(re.fullmatch(r"# коммит: [0-9a-f]{40}", s) for s in sh), "нет строки «# коммит: <sha1>»"
    for p in vhody:
        want = f"# вход: {SL.put(p)} sha256={_sha(p)}"
        assert want in sh, f"нет строки входа с путём и хэшем: {want}"
    for s in (s for s in sh if s.startswith("# вход:")):
        assert re.fullmatch(r"# вход: .+ sha256=[0-9a-f]{64}", s), s


def test_b_pishet_kommit_i_vhody(sreda, tmp_path):
    out = tmp_path / "b.csv"
    r1_b.zapisat(out, DF, "test1", "A_zashch")
    proverit_shapku(out.read_text(encoding="utf-8"), sreda)
    assert pd.read_csv(out, comment="#").equals(DF)                      # числа не тронуты, шапка читается с comment="#"


def test_e_pishet_kommit_i_vhody(sreda, tmp_path):
    r1_e.zapis("test1", {"po_fazam": DF, "fazy_doli": DF}, tmp_path)
    for n in ("e_po_fazam_test1.csv", "e_fazy_doli_test1.csv"):
        proverit_shapku((tmp_path / n).read_text(encoding="utf-8"), sreda)


def test_v_pishet_kommit_i_vhody(sreda, tmp_path):
    out = tmp_path / "v.csv"
    r1_v.zapisat(out, DF, "test1")
    proverit_shapku(out.read_text(encoding="utf-8"), sreda)
    out4 = tmp_path / "v4.csv"
    r1_v.zapisat(out4, DF, "train4")
    assert not out4.read_text(encoding="utf-8").startswith("#")          # train4 — без шапки, как раньше (сверка с #487)


def test_a_cd_shapka_dlya_a_cd_v_a_zashch_i_avtokorr(sreda):
    proverit_shapku(CD.shapka("test1"), sreda)


def test_zh_zashch_obe_tablicy_i_dva_odnoimennyh_sigma(sreda, tmp_path):
    sd = tmp_path / "ch4_sigma_z2"
    sd.mkdir()
    for n in ZH.FILES.values():
        (sd / f"{n}.parquet").write_bytes(f"Z2-{n}".encode())
    ZH.zapisat(tmp_path / "o", DF, DF, "test1", sd, 20)
    z2_hyb = sd / "sigma_hyb_h1_dev.parquet"
    dev_hyb = sigma_z2.dev_path(1)
    assert z2_hyb.name == dev_hyb.name and _sha(z2_hyb) != _sha(dev_hyb)    # два разных файла с одним именем (dev и Z2)
    for n in ("zh_sigma_zatvor_a_zashch_test1.csv", "zh_sigma_zatvor_a_zashch_k2_test1.csv"):
        t = (tmp_path / "o" / n).read_text(encoding="utf-8")
        proverit_shapku(t, sreda + [sd / f"{x}.parquet" for x in ZH.FILES.values()])
        assert SL.put(z2_hyb) != SL.put(dev_hyb) and SL.vhod(z2_hyb).strip() in t and SL.vhod(dev_hyb).strip() in t


def test_mutatsiya_bez_hasha_lovitsya(sreda, tmp_path, monkeypatch):
    monkeypatch.setattr(SL, "vhod", lambda p: f"# вход: {SL.put(p)}\n")            # писатель «забыл» хэш
    out = tmp_path / "b.csv"
    r1_b.zapisat(out, DF, "test1", "A_zashch")
    with pytest.raises(AssertionError):
        proverit_shapku(out.read_text(encoding="utf-8"), sreda)


def test_mutatsiya_bez_kommita_lovitsya():
    with pytest.raises(AssertionError):
        proverit_shapku("# реверс раунд 1\nportfel\n", [])


def test_arhiv_a_fix_sovpadaet_s_manifestom():
    """Опубликованная версия A_fix (_do_a_zashch) — эталон воспроизведения; подмена файла ловится по sha256 из MANIFEST.tsv."""
    d = ROOT / "docs" / "tables" / "revers_r1"
    rows = [r.split("\t") for r in (d / "_do_a_zashch" / "MANIFEST.tsv").read_text(encoding="utf-8").splitlines() if r.strip()]
    rows = [r for r in rows if r[0].startswith("_do_a_zashch/")]
    assert len(rows) == 4
    for rel, h, *_ in rows:
        assert _sha(d / rel) == h, rel
