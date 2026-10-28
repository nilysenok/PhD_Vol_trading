"""Возобновление прерванной резки: исходник уже в _ishodnye, части записаны — перепроверка без перезаписи."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import suzhenie_obshee as ob
import suzhenie_rezka as rz


def odinakovy(a, b):
    """Схема (без метаданных), значения и порядок; разбиение на чанки не важно (sha_tabl склеивает чанки)."""
    return a.schema.equals(b.schema, check_metadata=False) and ob.sha_tabl(a) == ob.sha_tabl(b)


def nesovpadenie(t, p):
    """None, если файл части p совпадает с таблицей t (пустая часть -> файла нет)."""
    if not len(t):
        return None if not p.exists() else f"лишний файл части: {p.name}"
    if not p.exists():
        return f"нет файла части: {p.name}"
    return None if odinakovy(pq.read_table(p), t) else f"часть не совпала: {p.name}"


def proverit(koren, rel, dirs, itog):
    """True при нарушении (status=FAIL). Ничего не пишет: исходник из _ishodnye режется в памяти и сверяется."""
    z2, z3, ish = dirs
    dst, src = Path(koren) / ish / rel, Path(koren) / rel
    m = ob.meta(dst, rel)
    klass, cut, h = ob.classify(rel, m, ob.dates_of(dst, m["date_col"]) if m["date_col"] else [])
    itog[klass] += 1
    base = dict(path=rel, klass=klass, h=h, cut=cut)
    if klass != "в":
        ob.vyvod(**base, status="FAIL", reason="исходник в _ishodnye не класса в")
        return True
    gr = ob.T1_HI if h == 0 else cut
    col = m["date_col"]
    parts = rz.razrezat(dst, col, gr)
    rz.proveri(pq.read_table(dst), parts, col, ob.dates_of(dst, col), h, gr)
    pre, t1, h25 = parts
    p1, p3 = rz.chasti_puti(rel, Path(koren) / z2, Path(koren) / z3)
    reason = (nesovpadenie(t1, p1) or nesovpadenie(h25, p3)
              or (None if src.exists() and odinakovy(pq.read_table(src), pa.concat_tables([pre, t1]))
                  else "усечённая копия не совпала с pre+t1"))
    cnt = dict(n_pre=len(pre), n_t1=len(t1), n_h25=len(h25))
    if reason:
        ob.vyvod(**base, **cnt, status="FAIL", reason=reason)
        return True
    ob.vyvod(**base, **cnt, status="uzhe_razrezan_provereno")
    return False
