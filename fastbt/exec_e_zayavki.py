"""V6-a (Ж-111 пп. 1–3): жизнь заявки входа «пока сигнал активен» и повторный вход в тот же день. Правила — для Run (exec_e_run).

T (Params.T): не исполнилась за день d+1 → если сигнал дня d+1 жив (та же сторона) и возраст < T, заявка ставится заново на d+2 от
close и σ̂ дня d+1 (пересчёт каждый вечер); сигнал ушёл или сменил сторону — снята (смену стороны подхватит _decide); возраст = T — тайм-аут,
дальше как у T = 1 (f: propusk | zakrytie_d1 | povtor). reentry_den: после внутридневного выхода в день d+1 заявка вечера d
(тот же уровень, что поставил бы Run) стоит на барах после бара выхода; не более одного повторного входа за день.
"""
from __future__ import annotations

import numpy as np

from fastbt.exec_e_rules import limit_mask, limit_px, stop_hit

INTRA = ("stop", "trail", "bu", "take", "lim", "kanal")     # внутридневные выходы: после них возможен повторный вход


def prodlit(s, o, i, j) -> bool:
    """Не исполненная за день заявка o живёт дальше? True — заявка поставлена на j+1 заново (уровень вечера i)."""
    sd, age = o["sd"], o.get("age", 1)
    if o["retry"] or age >= s.p.T or s.dir(i) != sd or j + 1 >= len(s.ix) or not np.isfinite(s.sigma[i]):
        return False
    s._place(s.close_of(i), sd, s.sigma[i], j, i, retry=False, age=age + 1)
    return True


def reenter(s, i, j, d) -> None:
    """Повторный вход после внутридневного выхода дня i (s.last_x): лимит вечера j−1 на барах строго после бара выхода."""
    x = s.last_x
    if (s.tr is not None or s.order is not None or j == 0 or x is None or x.exit_day != i or x.reason not in INTRA
            or s.done_re == i or x.exit_bar + 1 >= len(d.c)):
        return
    ip = s.ix[j - 1]
    sd, sg = s.dir(ip), s.sigma[ip]
    if not sd or not np.isfinite(sg):
        return
    s.done_re = i
    s._place(s.close_of(ip), sd, sg, j - 1, ip, retry=False)
    o, s.order = s.order, None
    if o is None:
        return
    b0 = x.exit_bar + 1
    hit = np.flatnonzero(limit_mask(sd, d.h[b0:], d.l[b0:], o["L"], s.p.tick)) + b0
    if s.p.zhizn is not None:
        hit = hit[hit < s.p.zhizn]
    if not hit.size:
        return
    b = int(hit[0])
    s.n_fills += 1
    s._enter(i, j, b, limit_px(sd, d.o[b], o["L"]), b, sg, ip, sd)
    S, _ = s._levels(j)
    if S is not None and stop_hit(sd, d.o[b], d.h[b], d.l[b], S, s.p.stop_touch)[0]:
        s._exit(i, j, b, S, s.sr)
    else:
        s._scan(i, j, d, b + 1)
