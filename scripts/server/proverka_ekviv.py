#!/usr/bin/env python3
"""Проверка эквивалентности на сервере (Ж-131 п. 3, Ж-133 п. 4): пересчёт набора эталона Мака и сравнение в два уровня (ekviv_sravn.py).

Дискретное (ключи, хэши, kod, vhody, позиции/сделки/индексы) — строго; непрерывное (Шарп, доходности, ряды) — с допуском DOPUSK
(переменная окружения, по умолчанию 0 = побайтно; якорь — 1e-12 по Ж-131). Печатает: OK / FAIL с первым расхождением и его
величиной; отдельно max |Δ| по непрерывным величинам — даже при OK. Выход: 0 — OK (пишется --otmetka), 1 — FAIL дискретного
(стоп), 3 — FAIL только непрерывного (прислать max |Δ| дирижёру, счёт не запускать). Метрики не печатаются.
    python3 proverka_ekviv.py --etalony etalony_<коммит>.npz --otmetka proverka/OK --kod-sha <..> --dannye-sha <..> [--procs N]
    python3 proverka_ekviv.py ... --moi <npz>     # сравнить с готовым файлом вместо пересчёта (тест, повторный разбор)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ekviv  # noqa: E402
import ekviv_io  # noqa: E402
import ekviv_sravn  # noqa: E402
import ekviv_v6  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--etalony", required=True)
    ap.add_argument("--otmetka", required=True)
    ap.add_argument("--kod-sha", required=True)
    ap.add_argument("--dannye-sha", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--moi", help="готовый npz вместо пересчёта")
    ap.add_argument("--rezhim", default="1", help="метка режима (1 — как есть, 2 — без AVX512/SVML) в отметке и вердикте")
    a = ap.parse_args(argv)
    dopusk = float(os.environ.get("DOPUSK", "1e-12"))      # Ж-137 п. 3: якорь и точки ≤ 1e-12, дискретное — строго
    otm = Path(a.otmetka)
    otm.parent.mkdir(parents=True, exist_ok=True)
    otm.unlink(missing_ok=True)
    eta, meta_eta = ekviv_io.zagruzit(a.etalony)
    tochki = ekviv_io.tochki_iz(eta)
    print(f"proverka[режим {a.rezhim}]: эталон {Path(a.etalony).name} ({meta_eta.get('machine')}, numba {meta_eta.get('numba')}, "
          f"numpy {meta_eta.get('numpy')}, numba_threads {meta_eta.get('numba_threads')}, layer {meta_eta.get('numba_layer')}): "
          f"якорей {sum(m.startswith('yakor') for m in eta)}, точек {len(tochki)}; v6: якорей {sum(m.startswith('v6yakor') for m in eta)}, "
          f"точек {len(ekviv_v6.iz_etalona(eta)[0])}, быстрый слой {len(ekviv_v6.iz_etalona(eta)[1])}; DOPUSK {dopusk:g}", flush=True)
    t0 = time.time()
    if a.moi:
        moi, meta_moi = ekviv_io.zagruzit(a.moi)
    else:
        moi, meta_moi = ekviv.poschitat(tochki, a.procs)
        if ekviv_v6.est_v6(eta):                  # эталон с набором v6: тот же допуск 1e-12, дискретное строго (Ж-137 п. 4)
            t6, b6 = ekviv_v6.iz_etalona(eta)
            rez6, m6, kod6 = ekviv_v6.poschitat(t6, b6, a.procs)
            ekviv_v6.dobavit(moi, meta_moi, rez6, m6, kod6)
        meta_moi.update(ekviv.sreda_meta())
    r = ekviv_sravn.sravnit(eta, meta_eta, moi, meta_moi, dopusk)
    disk, pv = r["diskretnoe"], r["nepreryvnoe_pervoe"]
    nf = ekviv.net_fajla(moi)
    if nf:
        disk = [f"{nf[0]}: оценщик не нашёл файл — данных не хватает ({len(nf)} записей)"] + disk
        r["kod"] = 1
    (otm.parent / f"verdikt_r{a.rezhim}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(
        dict(rezhim=a.rezhim, dopusk=dopusk, diskretnoe=disk[:20], nepreryvnoe_pervoe=pv, nepreryvnoe_chislo=r["nepreryvnoe_chislo"],
             max_yakor=r["max_yakor"], max_tochki=r["max_tochki"], etalon=meta_eta, zdes=meta_moi, sekund=round(time.time() - t0, 1),
             vremya=time.strftime("%F %T %Z")), ensure_ascii=False, indent=1, default=repr))
    print(f"proverka[режим {a.rezhim}]: max |Δ| непрерывных: якорь {r['max_yakor']:.3e} (допуск {ekviv_sravn.TOL_YAKOR:g}), "
          f"точки {r['max_tochki']:.3e} (допуск {dopusk:g}); дискретных расхождений {len(disk)}, "
          f"непрерывных вне допуска {r['nepreryvnoe_chislo']}; вердикт-файл рядом с отметкой", flush=True)
    if disk:
        print(f"FAIL дискретное [режим {a.rezhim}]: {disk[0]}")
        return 1
    if pv:
        print(f"FAIL непрерывное [режим {a.rezhim}]: {pv[0]}: {pv[1]}, |Δ| = {pv[2]:.3e} > {pv[3]:g}")
        return 3
    bb = ekviv_sravn.bystr_protiv_etalona(moi)       # быстрый слой v6 = эталонный путь здесь же (одинаковый сбой на обеих машинах не пройдёт)
    if bb:
        print(f"FAIL быстрый слой v6 ≠ эталонный путь [режим {a.rezhim}]: {len(bb)} точек, первая {bb[0][0]}: {bb[0][2]}")
        return 1 if any(k == 1 for _, k, _ in bb) else 3
    eh = hashlib.sha256(Path(a.etalony).read_bytes()).hexdigest()
    otm.write_text(f"kod={a.kod_sha}\ndannye={a.dannye_sha}\netalony={eh}\nrezhim={a.rezhim}\n")
    print(f"OK [режим {a.rezhim}]: дискретное совпало; {len(tochki)} точек + якорь 6 Sk; max |Δ| якорь {r['max_yakor']:.1e}, "
          f"точки {r['max_tochki']:.1e}; {time.time() - t0:.0f} с; отметка {otm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
