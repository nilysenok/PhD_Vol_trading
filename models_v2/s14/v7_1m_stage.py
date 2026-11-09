"""V7-1a-M: стадии бегуна `run_v7` кроме нуля. `1a` — инфраструктура (формы, реестр, CHECKPOINT ключ `1a`); `1b`, `2`, `3`, `4` — каркас вызова конфигураций с отказами (не из реестра, резерв 12, условная без условия).
`--check` ничего не пишет и данных SBER не читает; без `--check` стадия 1a пишет ТОЛЬКО ALL_TRIALS.csv (создаёт или дополняет) и ключ `1a` в CHECKPOINT.json (ключ `0` не трогает). Обучений на SBER и конфигураций в листе 1a нет.
Счёт конфигураций подключает лист 1b: `RUNNERS[<id>] = функция` (здесь пусто — бегун честно отвечает, что счёт не подключён)."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import torch

from models_v2.s14 import v7_0_io as IO, v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_loss as LS, v7_1m_reg as RG, v7_1m_wf as WF

DECLARED = dict(router=dict(A=51, B=67, C=31), experts=10, flat=76, moe_b=77, anchor_b=78)
RUNNERS: dict = {}


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v7_1m_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def zero_identity() -> list:
    """g ≡ 0 ⇒ F = H побитно для MoE (A/B/C), якоря и плоского корректора на случайном входе; F > 0 на крайних входах. -> список нарушений."""
    g = torch.Generator().manual_seed(0)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    Zg, z, ell, H = rn(64, 9), rn(64, 4), 0.3 * rn(64), torch.exp(rn(64))
    bad = []
    with FM.deterministic():
        for name, m in [("A", FM.ResMoE("A")), ("B", FM.ResMoE("B")), ("C", FM.ResMoE("C")), ("anchor", FM.ResMoE("B", anchor=True)), ("flat", FM.Flat()), ("flat_z", FM.Flat(FM.DZ))]:
            if not torch.equal(FM.forecast(H, m(Zg, z, ell)[0]), H):
                bad.append("g≡0: F ≠ H побитно у %s" % name)
            if not bool((FM.forecast(H, m(1e30 * Zg, 1e30 * z, ell)[0]) > 0).all()):
                bad.append("F ≤ 0 на крайних входах у %s" % name)
    return bad


def info() -> dict:
    return dict(params=FM.counts(), n_configs=len(RG.CONFIGS), N_holm=RG.N_HOLM, N_wide=RG.N_WIDE, grid=[list(x) for x in FT.GRID], seeds=list(FT.SEEDS), reg=LS.REG, first_eval=WF.FIRST_EVAL, first_fit_stage_b=WF.FIRST_FIT_B,
                hold=WF.HOLD, min_train=WF.MIN_TRAIN, lr=FT.LR, max_epochs=FT.MAX_EP, patience=FT.PATIENCE, code=code_hash(), trainings_on_sber=0, configs_computed=0)


def check_1a(cp_path: Path | None = None, trials: Path | None = None) -> list:
    cp_path, trials = cp_path or RG.CHECKPOINT, trials or RG.TRIALS
    bad = zero_identity()
    c = FM.counts()
    if c != DECLARED:
        bad.append("числа параметров %s ≠ объявленным %s" % (c, DECLARED))
    bad += RG.bad(trials)
    cp = json.loads(cp_path.read_text(encoding="utf-8")) if cp_path.exists() else {}
    if "1a" not in cp:
        bad.append("CHECKPOINT.json: нет ключа 1a")
    elif {k: v for k, v in cp["1a"].items() if k != "written_at"} != json.loads(json.dumps(info())):
        bad.append("CHECKPOINT ключ 1a ≠ пересчёту (код или параметры изменились)")
    if "0" not in cp:
        bad.append("CHECKPOINT.json: нет ключа 0")
    return bad


def write_1a(cp_path: Path | None = None, trials: Path | None = None) -> str:
    cp_path, trials = cp_path or RG.CHECKPOINT, trials or RG.TRIALS
    bad = zero_identity()
    if FM.counts() != DECLARED or bad:
        raise SystemExit("1a: формы не сошлись, запись отменена: %s" % bad)
    act = RG.ensure(trials)
    left = RG.bad(trials)
    if left:
        raise SystemExit("реестр нарушен, ключ 1a не пишется: %s" % left)
    cp = json.loads(cp_path.read_text(encoding="utf-8")) if cp_path.exists() else {}
    cp["1a"] = dict(info(), written_at=RG.now())
    cp_path.write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return act


def run_configs(stage: str, config: str | None, check: bool) -> int:
    ids = [config] if config else RG.STAGE_CONFIGS[stage]
    if stage == "2":                                                                          # лист 2a: счёт конфигураций 4–7 (`v7_2_run`); уже стоящие обработчики не заменяются
        from models_v2.s14 import v7_2_run as S2
        S2.connect(RUNNERS)
    ok, refused = [], []
    for cfg in ids:
        try:
            RG.gate(cfg)
            ok.append(cfg)
        except RG.Refusal as e:
            refused.append(cfg)
            print("ОТКАЗ конфигурации %s: %s" % (cfg, e))
    if check and stage == "2" and not refused:                                                # 2a: проверка читает данные SBER и файлы, ничего не пишет
        return S2.check_run(ok)
    if check:
        print("%s: допущены реестром %s, отказано %s (--check: данные не читались, ничего не записано)" % ("OK" if not refused else "ОТКАЗ", ok, refused))
    elif ok and any(c not in RUNNERS for c in ok):
        print("счёт конфигураций %s не подключён (лист 1a — инфраструктура; счёт — с листа 1b)" % [c for c in ok if c not in RUNNERS])
        return 3 if refused else 2
    else:
        for cfg in ok:
            RUNNERS[cfg]()
    return 3 if refused else 0


def run(stage: str, check: bool, config: str | None = None) -> int:
    if stage != "1a":
        return run_configs(stage, config, check)
    before = IO.outputs_digest()
    if check:
        bad = check_1a()
        after = IO.outputs_digest()
        if before != after:
            bad.append("--check изменил выходы (хэш до ≠ после)")
        print("\n".join(bad) if bad else "OK: G16 g≡0 ⇒ F=H, числа параметров %s; G19 реестр 11 строк, CHECKPOINT 1a; выходы не изменены (хэш %s)" % (FM.counts(), after[:16]))
        return 1 if bad else 0
    print("1a: реестр %s; CHECKPOINT ключ 1a записан" % write_1a())
    return 0
