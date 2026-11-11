"""Л6: скрипт единственного открытия нового периода (HAR-J Hybrid SBER, протокол HARJ_HYBRID_SBER_FWD, раздел 6; `otbor_statistika.md` п. 6; образец `scripts/otkrytie_gl4.py`).

Запуск (только дирижёр, один раз, после «да» владельца, репетиции и закрытых гейтов Л1–Л5):
    python3 scripts/fwd/otkrytie_harj_fwd.py            # без флагов и с --check: только проверки до замка, ничего не пишет, замка нет
    python3 scripts/fwd/otkrytie_harj_fwd.py --otkryt   # открытие
Порядок: (1) отказ без owner_yes в FREEZE и без строки «да» в журнале; (2) отказ при несовпадении sha кода (с замыканием импортов), входов, манифестов, пакетов;
(3) отказ, если репетиция не «0 расхождений» или у Л1–Л5 есть pending; (4) отказ, если замок уже есть; (5) замок O_EXCL ДО первого чтения цен
(`Src` открывает новые файлы только по замку); (6) один прогон узлов, таблицы, отчёт из таблиц, sha выходов в замок; (7) сбой после замка — причина в замок,
повтор только по решению владельца. Отказ на шагах 1–4 открытие не тратит (код 2).
Допущение: preflight читает байты новых файлов и `rv_fwd_SEALED.parquet` только для sha256, значения не разбирает (как `otkrytie_gl4`).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

import otkrytie_proverki as PV  # noqa: E402
import otkrytie_raschet as RS  # noqa: E402

NEW_NODES = ("k1", "k23", "cfg8", "x1", "zb", "zb0")
CSV = dict(family="FWD_FAMILY.csv", series="FWD_SERIES.csv", mcs="FWD_MCS.csv", months="FWD_MONTHS.csv", states="FWD_STATES.csv",
           descr="FWD_DESCR.csv", pairs="FWD_PAIRS.csv", checks="FWD_CHECKS.csv")
REPORT = "FWD_REPORT.md"
CHAIN = sorted(p.stem for p in (ROOT / "models_v2/s17").glob("fwd_*.py"))  # все модули цепочки, как в `fwd_frz_parts.import_chain`


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def import_chain() -> None:
    """Все модули цепочки — до проверок: замыкание импортов для сверки code_sha256 полное."""
    import importlib
    import torch  # noqa: F401  (первым: общий OpenMP, класс № 152)
    for m in CHAIN:
        importlib.import_module(f"models_v2.s17.{m}")


def acquire_lock(ctx, info: dict) -> None:
    """Шаг 5: замок до чтения цен; O_EXCL — два одновременных запуска не пройдут оба."""
    ctx.lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(ctx.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(dict(info, status="начато", vremya=now()), ensure_ascii=False, indent=1))


def finish_lock(ctx, extra: dict, status: str) -> bool:
    """Итог в замок. Пустой/битый замок не роняет обработчик: причина и итог — в `OPENED_HARJ_FWD.err` рядом; повтор всё равно невозможен (preflight видит и замок, и .err)."""
    try:
        d = json.loads(ctx.lock.read_text(encoding="utf-8"))
        d.update(extra, status=status, vremya_konets=now())
        os.chmod(ctx.lock, 0o644)
        ctx.lock.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        os.chmod(ctx.lock, 0o444)
        return True
    except Exception as e:  # noqa: BLE001
        ctx.lock.with_suffix(".err").write_text(json.dumps(dict(extra, status=status, vremya_konets=now(), lock_error=f"{type(e).__name__}: {e}"), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return False


def _put(path: Path, write) -> str:
    """Запись нового файла через tmp; существующий не перезаписывается."""
    if path.exists():
        raise FileExistsError(f"{path} уже есть")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)
    return PV.sha256_file(path)


def write_outputs(ctx, res: dict, violation: list | None = None, lock_status: str | None = None) -> dict:
    """Новые строки узлов — `data/fwd/models/new/*.parquet` (отдельно от архивных); таблицы — FWD_*.csv; отчёт — из таблиц, прочитанных обратно из CSV."""
    from models_v2.s17 import fwd_doc, fwd_eval
    out = {}
    for k in NEW_NODES:
        out[f"data/fwd/models/new/{k}.parquet"] = _put(ctx.new_dir / f"{k}.parquet", lambda p, df=res["new"][k]: df.to_parquet(p, index=False))
    for k, name in CSV.items():
        out[f"docs/otchety/harj_fwd/{name}"] = _put(ctx.rep_dir / name, lambda p, df=res["tables"][k]: df.to_csv(p, index=False))
    t = {k: pd.read_csv(ctx.rep_dir / CSV[k]) for k in ("family", "series", "mcs", "months", "states", "descr", "checks")}
    power = pd.read_csv(ctx.power, comment="#") if ctx.power.is_file() else None
    text = fwd_doc.render(t, power, violation)
    bad = fwd_doc.verify(text, t, power, violation, lock_status) + fwd_eval.check_family(t["family"])
    if bad:
        raise RuntimeError(f"отчёт или таблица семейства не согласованы: {bad}")
    out[f"docs/otchety/harj_fwd/{REPORT}"] = _put(ctx.rep_dir / REPORT, lambda p: p.write_text(text, encoding="utf-8"))
    return out


def run(ctx, run_nodes=RS.run_nodes, dry: bool = True, nodes=None) -> int:
    probs = PV.preflight(ctx)
    if probs:
        print("СТОП (замок не создан, открытие не потрачено):\n  " + "\n  ".join(probs), file=sys.stderr, flush=True)
        return 2
    if dry:
        print("проверки до замка пройдены; ничего не записано (для открытия — флаг --otkryt)")
        return 0
    fz = json.loads(ctx.freeze.read_text(encoding="utf-8"))
    try:
        acquire_lock(ctx, dict(freeze_sha256=PV.sha256_file(ctx.freeze), pid=os.getpid()))
    except FileExistsError:
        print("СТОП: замок уже есть — открытие было", file=sys.stderr, flush=True)
        return 2
    try:
        res = run_nodes(ctx, fz) if nodes is None else run_nodes(ctx, fz, nodes)
        extra = sorted(ctx.closure() - set(fz["code_sha256"]))  # замыкание после прогона: всё, что бегун загрузил, заморожено
        status = "выполнено" if not extra else f"нарушение заморозки: {', '.join(extra)}"
        outs = write_outputs(ctx, res, extra, status)  # выходы пишутся и при нарушении: отчёт помечен первой строкой, замок — статусом
        if not finish_lock(ctx, dict(outputs_sha256=outs), status):
            return 1
        print(f"открытие выполнено; выходов {len(outs)}" if not extra else f"{status}; выходы записаны, отчёт помечен")
        return 3 if extra else 0
    except BaseException as e:  # сбой после замка: открытие израсходовано, причина — в замок
        finish_lock(ctx, dict(prichina=f"{type(e).__name__}: {e}", trassa=traceback.format_exc()[-1500:]), "сбой")
        print(f"СБОЙ ПОСЛЕ ЗАМКА (открытие израсходовано, повтор — только по решению владельца): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1


def main(argv: list) -> int:
    import_chain()
    if "--pesochnica" in argv:  # сквозной путь на поддельном периоде из архива, всё в tmp; реальный замок и новые файлы не открываются
        import otkrytie_pesochnica as PS
        return PS.main(argv[argv.index("--pesochnica") + 1])
    return run(PV.Ctx(), dry="--otkryt" not in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
