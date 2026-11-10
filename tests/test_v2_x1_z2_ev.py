"""Z2: события (порог только по прошлым барам, тождество e_d = 1[ΣN ≥ 1]), контроль Hc (перестановка внутри дня), τ^last, n ≤ 0,9."""
import math

import numpy as np
import pandas as pd
import pytest

from models_v2.s15 import de_e1_state as S
from models_v2.s16 import x1_z2_ev as EV, x1_z2_hawkes as HK, x1_z2_synth as SY


def synth_events(seed=0, n=120):
    days, r = SY.store(n, seed)
    q = EV.thresholds(np.abs(r))
    N, m = EV.events(r, q[:, 1])
    return days, r, q, N, m


def test_threshold_uses_only_past_bars_and_day_d_mutation_caught(monkeypatch):
    days, r = SY.store(30, 1)
    r[25] = 5.0                                              # день-всплеск: все бары огромны
    N = lambda: EV.events(r, EV.thresholds(np.abs(r))[:, 1])[0]
    assert N()[25].all()                                     # порог по барам < 25: всплеск = событие
    r2 = r.copy()
    r2[26:] = 9.0                                            # будущее не меняет порог дня 25
    assert np.array_equal(EV.thresholds(np.abs(r))[25], EV.thresholds(np.abs(r2))[25])
    monkeypatch.setattr(EV, "past_end", lambda p: p + 1)
    assert not N()[25].any()                                 # мутант: день d в пороге — событий нет


def test_identity_e_equals_any_event_and_mutation_caught(monkeypatch):
    days, r = SY.store(60, 2)
    r[25] = 5.0                                              # день-всплеск: при пороге с днём d события нет
    q = EV.thresholds(np.abs(r))
    N = EV.events(r, q[:, 1])[0]
    st = S.states(dict(days=days, r=r), days)
    ok = np.isfinite(q[:, 1])
    assert np.array_equal(N[ok].any(1).astype(int), st[ok])
    monkeypatch.setattr(EV, "past_end", lambda p: p + 1)
    N2 = EV.events(r, EV.thresholds(np.abs(r))[:, 1])[0]
    assert not np.array_equal(N2[ok].any(1).astype(int), st[ok])


def test_identity_on_real_store_all_days():
    days, r, q = EV.real_thresholds()
    from models_v2.s15 import de0_data as D
    ok = np.isfinite(q[:, 1])
    N = EV.events(r, q[:, 1])[0]
    assert np.array_equal(N[ok].any(1).astype(int), S.states(D.load_store(), days[ok])) and ok.sum() > 2800


def test_events_marks_positive_and_known_answer():
    r = np.full((1, EV.NS), np.nan)
    r[0, :4] = [0.1, -0.5, 0.2, 0.9]
    N, m = EV.events(r, np.array([0.4]))
    assert N[0, :4].tolist() == [False, True, False, True] and np.allclose(m[0, [1, 3]], np.log([0.5 / 0.4, 0.9 / 0.4])) and not N[0, 4:].any()


def test_hc_permutation_invariants_and_mutations_caught():
    days, r, q, N, m = synth_events(3)
    pres = np.isfinite(r)
    N2, m2 = EV.permute_events(N, m, pres)
    EV.check_control(N, m, N2, m2, pres)                     # N_d, набор меток, присутствие — сохранены
    assert not np.array_equal(N, N2) and np.array_equal(N2, EV.permute_events(N, m, pres)[0])
    with pytest.raises(SystemExit):                          # Hc без перестановки ≡ H1
        EV.check_control(N, m, N, m, pres)
    with pytest.raises(SystemExit):                          # перестановка меняет N_d (между днями)
        EV.check_control(N, m, np.roll(N2, 1, axis=0), np.roll(m2, 1, axis=0), pres)
    with pytest.raises(SystemExit):                          # событие на отсутствующем баре
        bad = N2.copy()
        j = int(np.flatnonzero(~pres.all(1))[0])
        bad[j, np.flatnonzero(~pres[j])[0]], bad[j, np.flatnonzero(N2[j])[0] if N2[j].any() else 0] = True, False
        EV.check_control(N, m, bad, m2, pres)


def test_last_tau_known_answer():
    N = np.zeros((3, EV.NS), bool)
    N[0, 52], N[1, 0], N[1, 10] = True, True, True
    t = EV.last_tau(N)
    assert t[0] == 0.0 and t[1] == 530.0 - 10 * 11 and math.isnan(t[2])


def test_branching_n_capped_at_09_and_mutation_caught(monkeypatch):
    sim = SY.simulate(0.0005, 0.97, math.log(2) / 60, 0.3, 0.5, 900, seed=5)
    W = SY.window_of(sim["ctx"])
    assert HK.fit(W, sim["f"])["n"] <= 0.9 + 1e-12
    monkeypatch.setattr(HK, "N_MAX", 0.99)
    assert HK.fit(W, sim["f"])["n"] > 0.9
