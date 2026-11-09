"""V7-1b: стадия `run_v7 --stage 1b` — Stage A (G22), E0 (G23), конфигурации 1–2 (1b-B), конфигурация 3, оценка и отчёт (1b-C: `v7_1b_cfg3`, `v7_1b_evrun`).
Без `--check`: считает недостающие артефакты Stage A (возобновляемо), E0, пишет `1b_E0.csv` и в CHECKPOINT ключ `1b` подразделы `stageA` и `E0` (слиянием; чужие подразделы 1b и ключи сохраняются).
С `--check`: ничего не пишет; без артефактов Stage A отказывает (rc 1); иначе сверяет хэши файлов с CHECKPOINT, пересчитывает год 2018 и seed 0 (TCN и CNN) побитно, E0 — с записанным."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from models_v2.s14 import v7_0_io as IO, v7_1b_e0 as EZ, v7_1b_sta as STA

STATUS_NOTE = "G22–G23: Stage A и E0 на SBER; конфигураций 0"


def code_hash() -> str:
    from pathlib import Path
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v7_1b_*.py")) + sorted(Path(__file__).parent.glob("v7_1e_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def read_cp() -> dict:
    p = IO.REP / "CHECKPOINT.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def update_1b(sub: str, value: dict) -> None:
    """Подраздел ключа `1b` — слиянием: остальные подразделы сохраняются."""
    cur = dict(read_cp().get("1b", {}))
    cur[sub] = value
    IO.merge_checkpoint("1b", cur)


def e0_record(r: dict) -> dict:
    rec = {k: r[k] for k in STA.KINDS}
    rec.update(threshold=r["threshold"], both_no_new_info=r["both_no_new_info"], rule="R² компоненты — среднее по seeds 0–4 (взвешено по годам 2018–2021 числом дат); 'нет новой информации' — все 4 ≥ 0,9",
               stop_4_11=r["both_no_new_info"])
    return rec


def write() -> int:
    import shutil
    if shutil.disk_usage(IO.ROOT).free / 1e9 < 40:
        raise SystemExit("свободно < 40 ГБ: запись отменена")
    STA.train_missing()
    hs = {n: STA.file_hash(*t) for t, n in ((t, STA.name(*t)) for t in STA.tasks())}
    update_1b("stageA", dict(n_files=len(hs), kinds=list(STA.KINDS), years=list(STA.YEARS), seeds=list(STA.SEEDS), hashes=hs, code=code_hash(), note=STATUS_NOTE))
    r = EZ.run()
    update_1b("E0", e0_record(r))                                 # решение — в CHECKPOINT ДО любого счёта корректоров
    IO.write_csv(r["table"], "1b_E0.csv", "G23: частный R² компонент z на (S, Σ, ln H, ℓ), окно 2018–2021 по годам энкодера, потолок")
    print("E0: " + "; ".join("%s R²=%s нет новой информации=%s" % (k, np.round(r[k]["r2_mean"], 3).tolist(), r[k]["no_new_info"]) for k in STA.KINDS))
    from models_v2.s14 import v7_1b_cfgrun as CR                  # лист 1b-B: конфигурации 1–2 и choice3 (конфигурация 3 — лист 1b-C)
    rc = CR.write_cfgs()
    from models_v2.s14 import v7_1b_cfg3 as C3, v7_1b_evrun as EVR  # лист 1b-C: конфигурация 3 (энкодер из choice3), оценка 1–3, реестр, отчёт
    return rc or C3.write3() or EVR.write_eval()


def check() -> int:
    if STA.missing():
        print("нет артефактов Stage A (%d из %d отсутствуют): --check ничего не пишет" % (len(STA.missing()), len(STA.tasks())))
        return 1
    bad, cp = [], read_cp().get("1b", {})
    if "stageA" not in cp or "E0" not in cp:
        return _fail(["CHECKPOINT 1b: нет stageA/E0"])
    for t in STA.tasks():
        if STA.file_hash(*t) != cp["stageA"]["hashes"].get(STA.name(*t)):
            bad.append("%s: хэш содержимого ≠ CHECKPOINT" % STA.name(*t))
    from models_v2.s14 import v7_1e_pre as PRE
    pack = PRE.pack_from_npz(STA.WINDOW, IO.OUT)
    for k in STA.KINDS:
        new, old = STA.compute(k, 2018, 0, pack), STA.load(STA.directory() / STA.name(k, 2018, 0))
        if STA.content_hash(new) != STA.content_hash(old):
            bad.append("%s 2018 seed 0: пересчёт не совпал побитно" % k)
    r = EZ.run()
    for k in STA.KINDS:
        if not np.allclose(r[k]["r2_mean"], cp["E0"][k]["r2_mean"], rtol=0, atol=1e-12) or r[k]["no_new_info"] != cp["E0"][k]["no_new_info"]:
            bad.append("E0 %s: пересчёт ≠ CHECKPOINT" % k)
    from models_v2.s14 import v7_1b_cfgrun as CR                  # лист 1b-B: конфигурации 1–2, выбор для 3
    bad += CR.check(cp)
    from models_v2.s14 import v7_1b_cfg3 as C3, v7_1b_evrun as EVR  # лист 1b-C: конфигурация 3 и оценка
    bad += C3.check3(cp) + EVR.check_eval(cp)
    return _fail(bad) if bad else _ok()


def _ok() -> int:
    print("OK: G22 %d артефактов с хэшами CHECKPOINT, пересчёт 2018/seed 0 (TCN, CNN) побитно; G23 E0 сошёлся с CHECKPOINT; G24 (1–3): cfg1/cfg2/cfg3 по sha256, переоценка и энкодер окна 1 побитно, choice3 сошёлся; G25 таблицы оценки и отчёт равны пересчёту" % len(STA.tasks()))
    return 0


def _fail(bad: list) -> int:
    print("\n".join(bad))
    return 1


def run(check_only: bool) -> int:
    before = IO.outputs_digest() if check_only else None
    rc = check() if check_only else write()
    if check_only and IO.outputs_digest() != before:
        print("--check изменил выходы (хэш до ≠ после)")
        return 1
    return rc
