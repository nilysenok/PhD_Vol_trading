"""Ш3 и Ш4: резка смешанных файлов по дате и манифест после резки (Ж-112 п. 1)."""
import hashlib
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

import suzhenie_obshee as ob


def perenos(src, dst):
    """Исходник переносится целиком, не удаляется."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def filtry(schema, col, cut):
    typ = schema.field(col).type
    if not (pa.types.is_timestamp(typ) or pa.types.is_date(typ)):
        raise ValueError(f"тип столбца даты не поддержан: {typ}")

    def sk(d):
        if pa.types.is_date(typ):
            return pa.scalar(d, type=typ)
        return pa.scalar(datetime(d.year, d.month, d.day, tzinfo=ob.MSK if typ.tz else None), type=typ)

    f, lo, mid = pds.field(col), sk(ob.T1_LO), sk(cut + timedelta(days=1))
    return f < lo, (f >= lo) & (f < mid), f >= mid


def razrezat(src, col, cut):
    d = pds.dataset(str(src))
    return tuple(d.to_table(filter=x) for x in filtry(d.schema, col, cut))


def put_z2(parent):
    """Z2 открыт для чтения: в пути не должно быть сегмента test_locked.
    data/regen/test_locked/final3 -> regen/final3; data/models_cache/test_locked/boost -> models_cache/boost.
    Корни различаются сегментами до test_locked, поэтому коллизий нет; пути без test_locked не меняются."""
    parts = parent.parts
    if "test_locked" not in parts:
        return parent
    i = parts.index("test_locked")
    pre = parts[1:i] if parts[:1] == ("data",) else parts[:i]
    return Path(*pre, *parts[i + 1:])


def chasti_puti(rel, z2, z3):
    p = Path(rel)
    base = re.sub(r"_test", "", p.stem)
    return Path(z2) / put_z2(p.parent) / f"{base}__t1.parquet", Path(z3) / p.parent / f"{base}__h25.parquet"


def proveri(orig, parts, col, dates, h, cut):
    """Проверки Ш3 п. 3 до записи; нарушение — исключение, ничего не тронуто."""
    pre, t1, h25 = parts
    if sum(len(p) for p in parts) != orig.num_rows:
        raise ValueError("сумма строк частей не равна строкам исходника")
    for p in parts:
        if not p.schema.equals(orig.schema, check_metadata=False):
            raise ValueError("схема части не совпадает")
    if len(t1):
        mx = max(ob.to_dates(t1.column(col).to_pandas()))
        if mx > ob.T1_HI or not ob.okno_ok(dates, mx, h):
            raise ValueError("Z2: дата или окно цели за 31.12.2024")
    # при h > 0 в Z3 уходят и поздние дни 2024, чья цель в 2025; граница Z3 — cut, не 01.01.2025
    if len(h25) and min(ob.to_dates(h25.column(col).to_pandas())) <= cut:
        raise ValueError("Z3: строка не позже границы Z2")
    if h == 0 and len(h25) and min(ob.to_dates(h25.column(col).to_pandas())) < ob.H25_LO:
        raise ValueError("Z3: строка раньше 01.01.2025")


def proverka_chastey(parts, col, h, cal=None):
    """Ж-118: (причина нарушения | None, min дата закрытой части, число строк 2024 в ней)."""
    ds = [set(ob.to_dates(p.column(col).to_pandas())) if len(p) else set() for p in parts]
    pre, t1, h25 = ds
    mn, n24 = (min(h25) if h25 else None), 0
    if h25:
        n24 = sum(d <= ob.T1_HI for d in ob.to_dates(parts[2].column(col).to_pandas()))
    if (pre & t1) or (pre & h25) or (t1 & h25):
        return "части пересекаются по датам", mn, n24
    op = pre | t1
    if op and not (max(op) <= ob.T1_HI and ob.okno_ok([], max(op), h, cal)):
        return "открытая часть: дата или цель за 31.12.2024", mn, n24
    if h25 and h == 0 and mn < ob.H25_LO:
        return "закрытая часть: min дата раньше 01.01.2025 при h=0", mn, n24
    if h25 and h > 0:
        c = [d for d in (cal or ob.kalendar()) if d <= ob.T1_HI]
        cut = ob.granitsa([], h, cal)
        if mn != c[c.index(cut) + 1]:
            return "закрытая часть: min дата не следующий торговый день после cut", mn, n24
    return None, mn, n24


def rezat_file(koren, rel, m, dirs, primenit, itog):
    """Возвращает True при нарушении (status=FAIL); itog — счётчики классов для ITOG."""
    z2, z3, ish = dirs
    src = Path(koren) / rel
    dates = ob.dates_of(src, m["date_col"])
    klass, cut, h = ob.classify(rel, m, dates)
    itog[klass] += 1
    base = dict(path=rel, klass=klass, h=h, cut=cut)
    if klass != "в":
        ob.vyvod(**base, status="ne_rezhetsya")
        return False
    orig = pq.read_table(src)
    parts = razrezat(src, m["date_col"], cut)
    proveri(orig, parts, m["date_col"], dates, h, cut)
    pre, t1, h25 = parts
    cnt = dict(n_pre=len(pre), n_t1=len(t1), n_h25=len(h25))
    reason, mn, n24 = proverka_chastey(parts, m["date_col"], h)
    if reason:
        ob.vyvod(**base, **cnt, min_h25=mn, n_h25_2024=n24, status="FAIL", reason=reason)
        return True
    cnt.update(min_h25=mn, n_h25_2024=n24)
    p1, p3 = chasti_puti(rel, Path(koren) / z2, Path(koren) / z3)
    dst = Path(koren) / ish / rel
    if not primenit:
        ob.vyvod(**base, **cnt, status="dry-run")
        return
    if dst.exists() or p1.exists() or p3.exists():
        raise FileExistsError("целевой файл уже есть, перезапись запрещена")
    for p, t in ((p1, t1), (p3, h25)):
        if len(t):
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, p)
    perenos(src, dst)
    if not dst.exists() or ob.sha_file(dst) != m["sha256"]:
        raise RuntimeError("исходник не найден в _ishodnye или sha не совпал")
    pq.write_table(pa.concat_tables([pre, t1]), src)
    flt = filtry(orig.schema, m["date_col"], cut)[0]
    if ob.sha_tabl(pds.dataset(str(src)).to_table(filter=flt)) != ob.sha_tabl(pre):
        raise RuntimeError("срез <= 2021 усечённой копии не совпал с исходником")
    ob.vyvod(**base, **cnt, status="razrezan")


def rezat(koren, vkhod, dirs, primenit):
    """Число нарушений. Пробный прогон: нарушение -> status=FAIL и дальше; с --primenit исключение, как раньше."""
    itog, fail, files = dict.fromkeys("абвгд", 0), 0, ob.find_files(koren, vkhod, skip=dirs)
    for rel in files:
        m = ob.meta(Path(koren) / rel, rel)
        if not (m["date_col"] and m["nrows"]):
            itog["г"] += 1
            continue
        try:
            fail += bool(rezat_file(koren, rel, m, dirs, primenit, itog))
        except ValueError as e:
            if primenit:
                raise
            ob.vyvod(path=rel, status="FAIL", reason=str(e))
            fail += 1
    print(f"ITOG fajlov={len(files)} razrezano={itog['в']} celikom_otkryto={itog['а']} zakryto={itog['б']} "
          f"bez_daty={itog['г'] + itog['д']} fail={fail}")
    return fail


def posle(koren, vkhod, dirs, do_tsv, out, primenit):
    """Ш4: манифест после, сохранение «до» = части «после», хэш hold-out."""
    z2, z3, ish = dirs
    rows = [ob.meta(Path(koren) / r, r) for r in ob.find_files(koren, list(vkhod) + [z2, z3, ish])]
    for r in rows:  # Ж-119: строки конца 2024 с целью в 2025 лежат в закрытой части — их число в манифесте
        r["n_h25_2024"] = None
        if r["path"].startswith(z3 + "/") and r["date_col"] and r["nrows"]:
            col = pq.read_table(Path(koren) / r["path"], columns=[r["date_col"]]).column(0).to_pandas()
            r["n_h25_2024"] = sum(d <= ob.T1_HI for d in ob.to_dates(col))
    by = {r["path"]: r for r in rows}
    bad = 0
    for d in ob.chitai_tsv(do_tsv):
        p1, p3 = chasti_puti(d["path"], z2, z3)
        same = by.get(d["path"], {}).get("sha256") == d["sha256"]
        moved = by.get(f"{ish}/{d['path']}", {}).get("sha256") == d["sha256"]
        if same:
            ok = True
        elif moved and d["nrows"]:
            n = sum(int(by[x.as_posix()]["nrows"] or 0) for x in (Path(d["path"]), p3) if x.as_posix() in by)
            ok = n == int(d["nrows"])
        else:
            ok = moved
        bad += not ok
        ob.vyvod(path=d["path"], proveno=int(ok))
    z3rows = sorted(f"{r['path']}\t{r['size']}\t{r['sha256']}" for r in rows
                    if r["path"].startswith((z3 + "/", ish + "/")))
    hs = hashlib.sha256("\n".join(z3rows).encode()).hexdigest()
    if primenit:
        ob.pishi_tsv(Path(koren) / out, rows, ob.MANIFEST_COLS + ("n_h25_2024",))
    ob.vyvod(holdout_sha=hs, proveno=len(by) - bad, status="OK" if not bad else "FAIL")
    return bad
