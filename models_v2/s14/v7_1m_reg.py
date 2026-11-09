"""V7-1a-M, G19: реестр `ALL_TRIALS.csv` (11 конфигураций плана, редакция 2, раздел 5; Ж-233) ДО счёта. Только дополняется: существующий файл не переписывается, строки не меняются.
Строка = конфигурация (ключ `Ж-233:<n>`), статус «зарегистрирована», время регистрации по часам машины (`date`). Резерв 12 в реестр не входит. N поправки = 722 + 11 = 733, N_wide = 3122 + 11·4 = 3166.
`gate` — отказ бегуна: не из реестра; резерв 12; условная конфигурация (9, 11), пока в CHECKPOINT нет ключа условия = True."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path

from models_v2.s14 import v7_0_io as IO

TRIALS = IO.REP / "ALL_TRIALS.csv"
CHECKPOINT = IO.REP / "CHECKPOINT.json"
HEADER = ["key", "program", "source", "direction", "config", "kind", "registered_at", "status"]
PLAN = "docs/otchety/HARJ_HYBRID_SBER_V7_M10_MOE_PLAN.md"
STATUS_REG = "зарегистрирована"
N_HOLM, N_WIDE, RESERVE = 733, 3166, "12"
ENC_RULE = "энкодер — лучший из 1 и 2 по QLIKE 2018–2021 (окно 5 дней), правило объявлено"
COND9 = "условная: у 6 оба эксперта получают π > 0,3 не менее чем на 10 % дней и 6 несёт новую информацию против 4 (правило 6.2)"
COND11 = "условная: 10 несёт новую информацию против 6 (правило 6.2); для 11 против 8 — 10 против 6 или 8"
CONDITION_KEY = {"9": "cond_9", "11": "cond_11"}
_R = "F=H*exp(Σ_{k=1..K} π_k·g_k(z)), g0≡0, g_k=c*tanh((a_k+b_k·z)/c)"
_c = lambda i, direction, name, formula, base, ex, ro, cond="безусловная": dict(id=i, direction=direction, name=name, formula=formula, base=base, expert_in=ex, router_in=ro, condition=cond,
                                                                                encoder_rule=("TCN/CNN, окно 5 дней — свой энкодер конфигурации" if i in ("1", "2") else ENC_RULE))
CONFIGS = [
    _c("1", "encoder", "TCN-w5 + плоский корректор g(z)", "F=H*exp(c*tanh(MLP(z)))", "H", "z", "—"),
    _c("2", "encoder", "CNN-w5 + плоский корректор g(z)", "F=H*exp(c*tanh(MLP(z)))", "H", "z", "—"),
    _c("3", "encoder", "лучший из 1–2 с окном w=1 (по QLIKE 2018–2021)", "F=H*exp(c*tanh(MLP(z_w1)))", "H", "z_w1", "—"),
    _c("4", "corrector_no_router", "плоский корректор g(z,Z), 76 параметров", "F=H*exp(c*tanh(MLP(z,Z)))", "H", "(z,Z)", "—"),
    _c("5", "residual_moe_router_A", "Residual MoE K=2, маршрутизатор A", _R, "H", "z", "Z"),
    _c("6", "residual_moe_router_B", "Residual MoE K=2, маршрутизатор B (основная)", _R, "H", "z", "(Z,z)"),
    _c("7", "residual_moe_router_C", "Residual MoE K=2, маршрутизатор C", _R, "H", "z", "z"),
    _c("8", "residual_moe_anchor", "Residual MoE, якорь на M", "F=H*exp(w*ln(M/H)+Σ π_k·g_k(z))", "H, M", "z", "(Z,z)"),
    _c("9", "residual_moe_K3", "Residual MoE K=3", _R + ", K=3", "H", "z", "(Z,z)", COND9),
    _c("10", "joint_stage_B", "совместное обучение Stage B для 6", _R, "H", "X_w5", "(Z,z)"),
    _c("11", "joint_stage_B_anchor", "совместное обучение Stage B для 8", "F=H*exp(w*ln(M/H)+Σ π_k·g_k(z))", "H, M", "X_w5", "(Z,z)", COND11)]
IDS = [c["id"] for c in CONFIGS]
STAGE_CONFIGS = {"1b": ["1", "2", "3"], "2": ["4", "5", "6", "7"], "3": ["8", "9"], "4": ["10", "11"]}


class Refusal(Exception):
    """Бегун отказывает конфигурации (не из реестра, резерв, условие не выполнено)."""


def _row(c: dict, when: str) -> list:
    cfg = json.dumps({k: v for k, v in c.items() if k not in ("id", "direction")}, ensure_ascii=False, sort_keys=True)
    return ["Ж-233:%s" % c["id"], "Ж-233", PLAN + " раздел 5 (редакция 2), строка %s" % c["id"], c["direction"], cfg, "trial", when, STATUS_REG]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def text(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    w.writerows(rows)
    return buf.getvalue()


def read(path: Path | None = None) -> list:
    with open(path or TRIALS, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def ensure(path: Path | None = None, when: str | None = None) -> str:
    """Создать реестр, если его нет; иначе ДОПИСАТЬ недостающие строки плана в конец (существующие байты не трогаются). -> 'создан' | 'дополнен' | 'без изменений'."""
    path, when = path or TRIALS, when or now()
    if not path.exists():
        path.write_text(text([_row(c, when) for c in CONFIGS]), encoding="utf-8")
        return "создан"
    have = {r[0] for r in read(path)[1:]}
    miss = [c for c in CONFIGS if "Ж-233:%s" % c["id"] not in have]
    if not miss:
        return "без изменений"
    with open(path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([_row(c, when) for c in miss])
    return "дополнен"


RESULT_KIND, RESULT_SUFFIX, STATUS_RES = "result", ":результат", "посчитана"


V2_SUFFIX, ARCH_SUFFIX = ":результат_v2", ":результат_v1_архивирован"        # лист FX (09.11): результат версии 2 и пометка «результат v1 архивирован»; строки версии 1 остаются
STATUS_ARCH = "результат v1 архивирован: дефект согласования z"
R2B_SUFFIX = ":результат_2b"                                                              # лист DX (G41): результат оценки 2b (конфигурации 1–7, версия 2); строки регистрации не меняются
NOT_RUN_SUFFIX, STATUS_NOT_RUN = ":не_считается", "не считается: условие ложно (2.cond9)"          # лист 3 (G42): строка «9 — не считается»; регистрация не меняется
NOT_RUN11_SUFFIX, STATUS_NOT_RUN11 = ":не_считается_4cond11", "не считается: условие ложно (4.cond11)"                    # лист 4a (G46): строка «11 — не считается», если 4.cond11 ложно
SUFFIX_STATUS = {RESULT_SUFFIX: STATUS_RES, V2_SUFFIX: STATUS_RES, ARCH_SUFFIX: STATUS_ARCH, R2B_SUFFIX: STATUS_RES, NOT_RUN_SUFFIX: STATUS_NOT_RUN, NOT_RUN11_SUFFIX: STATUS_NOT_RUN11}


def is_result(r: list, decl: dict) -> bool:
    """Строка результата: ключ `<ключ плана>` + суффикс (`:результат`, `:результат_v2`, `:результат_v1_архивирован`), kind = result, статус суффикса, полный ряд колонок."""
    suf = next((s for s in SUFFIX_STATUS if r[0].endswith(s)), None)
    return len(r) == len(HEADER) and suf is not None and r[0][:-len(suf)] in decl and r[5] == RESULT_KIND and r[7] == SUFFIX_STATUS[suf] and bool(r[6])


def add_result(cfg: str, payload: dict, path: Path | None = None, when: str | None = None, suffix: str = RESULT_SUFFIX) -> str:
    """Дописать строку результата конфигурации `cfg` в конец реестра (существующие байты не трогаются). Повтор — 'уже есть' (результат не переписывается).
    suffix: `:результат` (версия 1), `V2_SUFFIX` (версия 2, лист FX) или `ARCH_SUFFIX` (пометка «результат v1 архивирован»); статус — по суффиксу."""
    path, key = path or TRIALS, "Ж-233:%s%s" % (cfg, suffix)
    if key in {r[0] for r in read(path)[1:]}:
        return "уже есть"
    base = [r for r in read(path)[1:] if r[0] == "Ж-233:%s" % cfg][0]
    with open(path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow([key, "Ж-233", base[2], base[3], json.dumps(payload, ensure_ascii=False, sort_keys=True), RESULT_KIND, when or now(), SUFFIX_STATUS[suffix]])
    return "дописана"


def bad(path: Path | None = None) -> list:
    """Нарушения реестра (пусто = ок): заголовок, 11 строк плана, дубли, текст конфигурации = объявленному, статус, время."""
    path = path or TRIALS
    if not path.exists():
        return ["ALL_TRIALS.csv: нет"]
    rows = read(path)
    out = [] if rows and rows[0] == HEADER else ["заголовок ALL_TRIALS изменён"]
    keys = [r[0] for r in rows[1:]]
    out += ["дубль: %s" % k for k in sorted(set(keys)) if keys.count(k) > 1]
    decl = {"Ж-233:%s" % c["id"]: _row(c, "") for c in CONFIGS}
    for r in rows[1:]:
        if is_result(r, decl):                                                                # строка результата (лист 1b+): дописана после регистрации, строка регистрации не тронута
            continue
        if r[0] not in decl:
            out.append("строка вне плана: %s" % r[0])
        elif r[:6] != decl[r[0]][:6] or len(r) != len(HEADER) or not r[6]:
            out.append("строка %s не равна объявленной или без времени" % r[0])
    out += ["нет строки плана: %s" % k for k in decl if k not in keys]
    return out


def checkpoint_cond(key: str, path: Path | None = None) -> bool:
    path = path or CHECKPOINT
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get(key) is True


def gate(cfg: str, trials: Path | None = None, checkpoint: Path | None = None) -> dict:
    """Допуск конфигурации к счёту; иначе Refusal. -> строка реестра как dict."""
    cfg, trials = str(cfg), trials or TRIALS
    if cfg == RESERVE:
        raise Refusal("резерв 12: только по отдельному разрешению владельца")
    if not trials.exists() or bad(trials):
        raise Refusal("реестр отсутствует или нарушен: %s" % (bad(trials)[:2]))
    row = {r[0]: r for r in read(trials)[1:]}.get("Ж-233:%s" % cfg)
    if row is None:
        raise Refusal("конфигурация %s не из реестра" % cfg)
    if row[7] != STATUS_REG:
        raise Refusal("конфигурация %s: статус %r, нужен «зарегистрирована»" % (cfg, row[7]))
    key = CONDITION_KEY.get(cfg)
    if key and not checkpoint_cond(key, checkpoint):
        raise Refusal("конфигурация %s условная: в CHECKPOINT нет %s = true" % (cfg, key))
    return dict(zip(HEADER, row))
