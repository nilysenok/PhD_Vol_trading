"""T1 (Ж-036): ledger пишет пробу только при периоде `setup` и комиссии отбора 0,04. Синтетика, без реальных данных.
Тест красный, пока кодер 1 не запушил правку (ledger сейчас пишет и на control/train)."""
import json

from fastbt.sweep import Sweep
from fastbt.sweep_spec import Spec
from fastbt.sweep_synth import SynthEvaluator

P = {"a": 5, "w": 1.0}


def mk():
    return Sweep(Spec(obj={"kind": "Synth"}, base=dict(P)), SynthEvaluator())


def ok_ledger_setup_only() -> bool:
    """Точка: setup -> control -> train. SR в ledger остаётся SR за setup, N не меняется; пересчёт при 0,035 не пишет."""
    sw = mk()
    r0 = sw.evaluate(P, "setup")
    sr_setup, n0 = list(sw.trials_sr()), r0["N"]
    r1, r2 = sw.evaluate(P, "control"), sw.evaluate(P, "train")
    sw.evaluate(P, "setup", 0.035)
    return bool(n0 == 1 and r1["N"] == 1 and r2["N"] == 1 and sw.trials_sr() == sr_setup
                and sr_setup == [r0["sr_d"]] and r1["sr_d"] != r0["sr_d"])


def ok_control_alone_not_a_trial() -> bool:
    sw = mk()
    sw.evaluate(P, "control")
    sw.evaluate(P, "train")
    return sw.trials_sr() == [] and sw.evaluate(P, "setup")["N"] == 1


def test_ledger_pishet_tolko_na_setup_pri_komissii_otbora():
    assert ok_ledger_setup_only()


def test_control_i_train_bez_setup_ne_proba():
    assert ok_control_alone_not_a_trial()


def test_mut_ledger_pishet_na_lyubom_periode(monkeypatch):
    orig = Sweep.evaluate

    def ev(self, params, period="train", comm_pct=None):
        r = orig(self, params, period, comm_pct)
        if comm_pct is None and period != "validation":
            self.ledger[json.dumps(sorted({**self.spec.base, **params}.items()), default=str)] = r["sr_d"]
        return r
    monkeypatch.setattr(Sweep, "evaluate", ev)
    assert not ok_ledger_setup_only() and not ok_control_alone_not_a_trial()
