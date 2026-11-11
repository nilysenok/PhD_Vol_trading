"""Л5: заморозка — `docs/otchety/harj_fwd/FREEZE.json` версии 1 + ключ `FWD.freeze` в `docs/otchety/harj_x1/CHECKPOINT.json` (слиянием; чужие ключи сохраняются).
Протокол HARJ_HYBRID_SBER_FWD, раздел 6. Условия записи (иначе отказ, ничего не пишется): FREEZE.json ещё нет; REHEARSAL.csv — «0 расхождений»; строка «да» владельца (Ж-248 п. 4) есть
в `docs/resheniya/resheniya.md`; POWER.csv посчитан для n_new календаря. Поверх существующего файла не пишет (замок открытия сверяет sha заморозки).
Запуск: `python3 -m models_v2.s17.fwd_freeze --write` (сборка и запись) / `--check` (только читает: замыкание импортов и sha файлов против FREEZE.json).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from models_v2.s17 import fwd_frz_parts as P, fwd_power as PW, fwd_rehearsal as RH
from models_v2.s17.fwd_src import ARCHIVE_END, LAST_SESSION, ROOT, Src

OUT = ROOT / "docs/otchety/harj_fwd/FREEZE.json"
CP = ROOT / "docs/otchety/harj_x1/CHECKPOINT.json"
JOURNAL = ROOT / "docs/resheniya/resheniya.md"
OWNER = {"zh": "Ж-248", "date": "2026-11-11", "punkt": 4, "quote": "После фиксации протокола и технической репетиции разрешаю проверить доступный новый период до заранее указанной последней полностью завершённой сессии."}
FAMILY = dict(hypotheses=["H1 X1<K2", "H2 X1<K3", "H3 ZB<K2", "H4 ZB<K3", "H5 ZB<X1", "H6 X1<K1", "H7 ZB<K1"], holm_m=7, alpha=0.05, dm="односторонний, Ньюи–Уэст, лаг 5", bootstrap=dict(block=10, B=2000, seed=0), interval="90 % двусторонний")


class Refusal(RuntimeError):
    """Условие заморозки не выполнено."""


def owner_in_journal(owner: dict | None = None, journal: Path | None = None) -> bool:
    return (owner or OWNER)["quote"].rstrip(".") in (journal or JOURNAL).read_text(encoding="utf-8")


def preconditions(path: Path = OUT) -> None:
    if path.exists():
        raise Refusal(f"{path.name} уже есть: заморозка не перезаписывается")
    if not owner_in_journal():
        raise Refusal("строка «да» владельца (Ж-248 п. 4) не найдена в resheniya.md")
    if not RH.OUT.exists():
        raise Refusal("репетиция: REHEARSAL.csv не записан")
    prob = RH.problems(pd.read_csv(RH.OUT, comment="#"))
    if prob:
        raise Refusal(f"репетиция: расхождений/нарушений {len(prob)}")
    if PW.n_new_of_calendar() != int(pd.read_csv(PW.OUT, comment="#")["n_new"].iloc[0]):
        raise Refusal("POWER.csv посчитан не для n_new календаря")


def build(src: Src, commit: str | None = None) -> dict:
    cal, pairs = P.calendar_section(src)
    P.import_chain()
    code = P.loaded_files()
    ze = ROOT / "docs/otchety/harj_x1/ZE_CRIT.csv"
    return dict(v=1, written_at=datetime.now(ZoneInfo("Europe/Moscow")).isoformat(timespec="seconds"), git=P.git_section(list(code), commit), code_sha256=code, packages=P.packages(), inputs_archive_sha256=P.inputs_archive(),
                derived_archive_sha256=P.derived_sha(), inputs_new_sha256=P.inputs_new(), params=P.params(), initial_state=P.snapshot(src), calendar=cal,
                dates=dict(first_d=str(pairs["date"].iloc[0].date()), last_d=str(pairs["date"].iloc[-1].date()), first_t_plus=str(pairs["t_plus"].iloc[0].date()), last_t_plus=str(pairs["t_plus"].iloc[-1].date()), last_session=str(LAST_SESSION.date())),
                candidates=["X1", "ZB"], not_admitted=dict(ZE=dict(status="не допущена", ref="docs/otchety/harj_x1/ZE_CRIT.csv", sha256=P.sha(ze))), controls=["K1", "K2", "K3"], family=FAMILY,
                outputs=dict(models="data/fwd/models/", lock="data/fwd/models/OPENED_HARJ_FWD.json", report="docs/otchety/harj_fwd/FWD_*"), rehearsal=dict(sha256=P.sha(RH.OUT), result="0 расхождений"),
                power=dict(sha256=P.sha(PW.OUT), n_new=cal["n_new"]), owner_yes=OWNER)


def merge_cp(frz: dict, path: Path = CP) -> None:
    """Слияние ключа FWD.freeze: чужие ключи сохраняются; повторная запись отказывает."""
    cp = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "freeze" in cp.get("FWD", {}):
        raise Refusal("FWD.freeze в CHECKPOINT уже есть")
    cp.setdefault("FWD", {})["freeze"] = dict(path="docs/otchety/harj_fwd/FREEZE.json", sha256=P.sha(OUT), v=frz["v"], written_at=frz["written_at"], n_new=frz["calendar"]["n_new"], rehearsal_sha256=frz["rehearsal"]["sha256"], power_sha256=frz["power"]["sha256"])
    tmp = path.with_name("tmp_" + path.name)
    tmp.write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write(src: Src | None = None, path: Path = OUT, commit: str | None = None) -> dict:
    preconditions(path)
    frz = build(src or Src(), commit)
    tmp = path.with_name("tmp_" + path.name)
    tmp.write_text(json.dumps(frz, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.link(tmp, path)                       # атомарно и без перезаписи: если файл появился между проверкой и записью — FileExistsError
    tmp.unlink()
    merge_cp(frz)
    return frz


def check(path: Path = OUT) -> dict:
    """Только читает: замыкание импортов бегуна против `code_sha256` и sha входов против заморозки."""
    frz = json.loads(path.read_text(encoding="utf-8"))
    P.import_chain()
    return dict(closure=P.closure_problems(frz["code_sha256"], P.loaded_files()), archive_changed=sorted(k for k, v in P.inputs_archive().items() if frz["inputs_archive_sha256"].get(k) != v))


def main(argv: list) -> int:
    if "--write" in argv:
        frz = write(commit=argv[argv.index("--commit") + 1] if "--commit" in argv else None)
        g = frz["git"]
        print(f"FREEZE.json v1 записан: n_new={frz['calendar']['n_new']}, файлов в замыкании {len(frz['code_sha256'])}; не в коммите {len(g['not_in_commit'])}, отличаются {len(g['differs_from_commit'])}")
        return 0
    if "--check" in argv:
        res = check()
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return int(any(res["closure"].values()) or bool(res["archive_changed"]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
