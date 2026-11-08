"""HAR-J Hybrid V5, лист 1b (Ж-226), диагностики моментов и значимых скачков. Запуск из корня репозитория, процесс один:
  python3 -W ignore -m models_v2.s12.run_v5_dx --stage DX1          # пересчитать и записать выходы листа
  python3 -W ignore -m models_v2.s12.run_v5_dx --stage DX1 --check  # ТОЛЬКО ЧТЕНИЕ: пересчёт в памяти, сверка с диском, хэш выходов до и после
Моделей не обучает; конфигурации не расходуются."""
from __future__ import annotations

import argparse
import sys

from models_v2.s12 import v5_dx1_stage as ST


def evidence(r: dict) -> list:
    md, jd = r["mdec"], r["jdec"]
    return ["G16: строк моментов %d; направление 3: %s; p Вальда (K8, 2017–2021) = %.4g; крупнейший |t| (φ₁…φ₆) %s = %.2f; max|Δφ| при обрезке будущего = %.1e" % (len(r["moments"]), md["verdict"], md["p"], md["big_phi"], md["big_phi_t"], r["pit"]),
            "G17: строк скачков %d; направление 5: %s; доля дней %.4f, доля QLIKE K8 %.4f, M %.4f, H %.4f; событий на бумагу: min %d, медиана %.0f; условия %s/%s/%s"
            % (len(r["jumps"]), jd["verdict"], jd["share"], jd["ql_k8"], jd["ql_m"], jd["ql_h"], jd["min_events"], jd["median_events"], jd["c1"], jd["c2"], jd["c3"])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["DX1"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    before = ST.digest()
    r = ST.compute(write_cache=not a.check)
    if not a.check:
        ST.write(r)
        print("стадия DX1 записана; хэш выходов %s" % ST.digest())
        print("\n".join(evidence(r)))
        return 0
    bad = ST.check(r)
    after = ST.digest()
    if before != after:
        bad.append("--check изменил файлы: хэш выходов до %s, после %s" % (before, after))
    print("\n".join(evidence(r)))
    print("хэш выходов до = после: %s (%s)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G16, G17: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
