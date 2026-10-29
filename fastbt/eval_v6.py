"""V6-12a/b: оценщик точки куба v6 поверх EvaluatorV5 (docs/otchety/2026-10-29_setka_v6_inventarizaciya.md §5, V6-12).

`EvaluatorV6(EvaluatorV5)`: точка = точка v5 + новые оси куба v6 (canon_v6.novye). Все новые оси «выкл.» (или нет в точке) —
путь v5 побайтно. 12a: rel_signal (td_rel), dlit, asim, norm, фильтры hurst_th/naklon/vol_regime_max/tihiy_othod (v5_filtr),
контекст К9–К16, К20–К24 (kontekst_v6), окна событий sob_*, внешние условия vn_*, vhod_tip (тренд). 12b (eval_v6_osi): time_decay,
sigma_ist, ed_lim, zhizn, тень паузы pauza_*. 12c (eval_v6_ispoln): lestnica, gap, t_vyhod, vyhod_lim, f_neisp. Остальные оси и расширенные значения
осей v5 — ValueError «ещё не подключена» (tf — свой исполнитель; пивот V6-10 и тренд V6-11 — следующие листы). Данные не читает: ряды подаёт `IstochnikiV6`.
Вход разрешён, если long_ok/short_ok стороны сигнала истинны (данные <= d; исполнение >= d+1, Д2). Ключ кэша — ВСЕ оси v6 (Д5).
Отступления: фильтры v5 (hurst, ATR, объём) — по реальному td, не td_rel; norm=s252 + tejk=rodnoy — NeSchitana.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from fastbt import canon_v6, eval_v6_ispoln as isp6, eval_v6_osi as osi6, kontekst_v6, sobytiya_e1, vneshnie_uslovie
from fastbt.canon_v5 import NET, SIGNAL, tip
from fastbt.eval_v5 import EvaluatorV5, UROVNI, maska
from fastbt.eval_v5_osi import razbor
from fastbt.etap1_eval_v2 import NOVYE, EvaluatorV2, NeSchitana
from fastbt.eval_v6_osi import rel_td_ind
from fastbt.signals.interface import get_family
from fastbt.signals.v6_obertka import primenit_asim, primenit_dlit
from fastbt.stop_zayavki import okno as _okno, urovni as _urovni
from fastbt.unit import Unit

KONTEKST = ("sektor_trend", "shirina", "ma_n", "porog_shiriny", "relf_napr", "relf_k", "relf_okno", "relf_bench", "post_shok",
            "szhatie", "rv_perevorot", "obyom_anom", "rezhim")
FILTR6, SOB, VN = ("hurst_th", "naklon", "vol_regime_max", "tihiy_othod"), ("sob_tip", "sob_do", "sob_posle"), ("vn_uslovie", "vn_faktor")
PODKLYUCHENO = frozenset(("rel_signal", "dlit", "asim", "norm", "vhod_tip") + KONTEKST + FILTR6 + SOB + VN + osi6.DVIG + osi6.PAUZA
                         + isp6.ISPOLN)
REZHIM_ALIAS = {"rang_sigma_rynka_08": "rank08"}


@dataclass
class IstochnikiV6:
    """Готовые ряды (≤ 2021). dannye(t, td) → kontekst_v6.Dannye на td.dates; rel(t, rezhim) → дневной td_rel (колонки
    daily_1850.COLS, rel_ryad.td_rel_rezhim); vn(t, faktor) → (z, beta) pd.Series по датам; cbr/div — таблицы sobytiya_e1."""
    dannye: Callable | None = None
    rel: Callable | None = None
    vn: Callable | None = None
    cbr: pd.DataFrame | None = None
    div: pd.DataFrame | None = None
    div_verified: bool = False


def razdelit(sid: str, pt: dict) -> tuple[dict, dict]:
    """(точка v5, оси v6 не в «выкл.»). Неподключённая ось или расширенное значение оси v5 — ValueError."""
    kub = canon_v6._kub(sid)
    nov = canon_v6.novye(kub)
    v6 = {k: v for k, v in pt.items() if k in nov and v != nov[k]["setka"][0]}
    for k in sorted(set(v6) - PODKLYUCHENO):
        raise ValueError(f"{sid}: ось {k} ещё не подключена (V6-12b/c или лист оси)")
    for n, r in canon_v6.rasshireno(kub).items():
        if n in pt and str(pt[n]) in map(str, r["v6_dobavleno"]):
            raise ValueError(f"{sid}: значение {n}={pt[n]!r} (v6) ещё не подключено")
    return {k: v for k, v in pt.items() if k not in nov}, v6


def fkey(filtr: dict, v6: dict) -> tuple:
    """Ключ кэша сигналов и позиций (Д5): фильтры v5 + ВСЕ оси v6 точки; без осей v6 — ключ v5."""
    return tuple(sorted(filtr.items())) + ((("v6", tuple(sorted(v6.items(), key=str))),) if v6 else ())


class EvaluatorV6(isp6.IspolnMixin, EvaluatorV5):
    def __init__(self, sid, inp, base_p, commission: float = 0.0004, fdr=None, ist: IstochnikiV6 | None = None):
        super().__init__(sid, inp, base_p, commission, **({} if fdr is None else {"fdr": fdr}))
        self.ist, self.v6, self._rel, self._dan, self._ten = ist or IstochnikiV6(), {}, {}, {}, None

    def _need(self, name: str, ось: str):
        x = getattr(self.ist, name)
        if x is None or (name == "cbr" and not len(x)):
            raise ValueError(f"{self.sid}: ось {ось} включена, а в IstochnikiV6 нет {name}")
        return x

    def _src(self, t, td, ind):
        rs = self.v6.get("rel_signal", "syroy")
        if rs == "syroy":
            return td, ind
        if (t, rs) not in self._rel:
            self._rel[t, rs] = rel_td_ind(self._need("rel", "rel_signal")(t, rs), td)
        return self._rel[t, rs]

    def _blok(self, t, td) -> np.ndarray:
        v, tp = self.v6, self.v6["sob_tip"]
        do, po = int(v.get("sob_do", 1)), int(v.get("sob_posle", 0))
        b = sobytiya_e1.cbr_block(td.dates, self._need("cbr", "sob_tip"), do, po) if "cbr" in tp else np.zeros(td.n, bool)
        if "div" in tp:
            if not self.ist.div_verified:
                raise ValueError(f"{self.sid}: sob_tip={tp} — дивиденды не сверены (div_verified=False, ❓5)")
            b = b | sobytiya_e1.div_block(td.dates, self._need("div", "sob_tip"), t, do, po)
        return b

    def _ok6(self, t, td) -> tuple:
        """(long_ok, short_ok) по td.dates: К9–К16, К20–К24, события, внешние условия — по «И»."""
        v, lo, sr = self.v6, np.ones(td.n, bool), np.ones(td.n, bool)
        kp = {k: v[k] for k in KONTEKST if k in v}
        if kp:
            kp["rezhim"] = REZHIM_ALIAS.get(kp.get("rezhim"), kp.get("rezhim", "net"))
            if t not in self._dan:
                self._dan[t] = self._need("dannye", "К9–К24")(t, td)
            lo, sr = kontekst_v6.long_short(kp, self._dan[t])
        if v.get("sob_tip", NET) != NET:
            b = self._blok(t, td)
            lo, sr = lo & ~b, sr & ~b
        if "vn_uslovie" in v:
            ix = pd.DatetimeIndex(td.dates)
            z, be = (x.reindex(ix) for x in self._need("vn", "vn_uslovie")(t, v.get("vn_faktor", "brent")))
            lo = lo & vneshnie_uslovie.uslovie(v["vn_uslovie"], z, be, 1).to_numpy(bool)
            sr = sr & vneshnie_uslovie.uslovie(v["vn_uslovie"], z, be, -1).to_numpy(bool)
        return lo, sr

    def _sig_src(self, fam, ts, is_, p):
        """Сигнал точки p на источнике сигнала: семья v5 → dlit."""
        ss = fam(ts, is_, [dict(p)])
        k = self.v6.get("dlit")
        if k is None:
            return ss, ss.sig[0]
        q = {**p, "vhod_tip": self.r5.nov["vhod_tip"]}
        return ss, primenit_dlit(self.sid, ts, is_, q, ss.sig[0], int(k))

    def _signals5(self, sp: dict) -> dict:
        fam, out, v = get_family(self.sid, setka="v5"), {}, self.v6
        spn = {**sp, "norm": v["norm"]} if "norm" in v else dict(sp)
        for t, (td, ind) in self.sk._td.items():
            ts, is_ = self._src(t, td, ind)
            ss, sig = self._sig_src(fam, ts, is_, spn)
            if "asim" in v:
                sig, _ = primenit_asim(self.sid, lambda q: self._sig_src(fam, ts, is_, q)[1], spn, v["asim"])
            if self.r5.filtr:
                sig = np.where(maska(td, tip(self.sid), self.r5.filtr, sig), sig, 0)
            lo, sr = self._ok6(t, td)
            sig = np.where(((sig > 0) & ~lo) | ((sig < 0) & ~sr), 0, sig)
            ss.sig[0] = sig.astype(ss.sig.dtype)
            out[t] = Unit(t, self.sid, td, ind, ss)
        return out

    def _urovni(self, t, s0, b, al):
        sp = {UROVNI[self.sid].get(k, k): x for k, x in self.r5.sp.items()}
        if self.v6.get("norm") == "s252" and self.sid == "S4":
            sp["atr_period"] = 252
        td, ind = self.sk._td[t]
        u = _urovni(self.sid, sp, td, ind, "obe")
        lo, sr = self._ok6(t, td)
        for k, z, ok in (("up", 1, lo), ("dn", -1, sr)):
            m = ok if not self.r5.filtr else ok & maska(td, "trend", self.r5.filtr, np.full(td.n, z))
            u[k] = np.where(m, u[k], np.nan)
        return _okno(u, s0, b, al)

    def _ed(self, t, e, *a):
        """ed_lim (V6-12b): единица отступа лимитки; v5 передаёт "sigma"."""
        return super()._ed(t, self.v6.get("ed_lim", e) if e == "sigma" else e, *a)

    def _allow(self, pt, t, dates, sig, f):
        a = super()._allow(pt, t, dates, sig, f)
        return a if self._ten is None else a & osi6.razreshit(self.v6, t, dates, self._ten)

    def _rows(self, pt, period, anchor):
        if self.r5 is not None and self.v6.get("norm") == "s252" and self.r5.nov["tejk_signal"] != NET:
            raise NeSchitana("norm=s252: родной уровень tejk считается по масштабу окна, не 252 дней")
        if self.r5 is not None:
            osi6.proverit(self.r5, self.v6)
            isp6.proverit(self.r5, self.v6)
        self._ten = None
        if "pauza_porog" in self.v6:                      # тень: та же точка без паузы (V6-12b), затем прогон с паузой
            self._ten = osi6.ten(self, self._legs(pt, period, anchor), period, anchor)
        return self._legs(pt, period, anchor)

    def evaluate_v2(self, pt: dict, period: str, vozm: dict | None = None):
        """Точка v6 → (строка ledger v2.0, excess портфеля, excess тикеров); контракт EvaluatorV5."""
        if vozm:
            raise ValueError("vozm у оценщика v6 не поддержан")
        pt5, self.v6 = razdelit(self.sid, pt)
        self.r5 = r = razbor(self.sid, pt5, SIGNAL[self.sid])
        if "vhod_tip" in self.v6 and self.sid in ("S3", "S4"):
            r.nov["vhod_tip"] = self.v6["vhod_tip"]
        osi6.primenit(r, self.v6, self.inp.sigma)
        isp6.primenit(r, self.v6)
        t = tip(self.sid)
        for k in FILTR6:
            if k in self.v6:
                self._filtr6(r.filtr, k, self.v6[k], t)
        self._fkey = fkey(r.filtr, self.v6)
        nov = {k: v for k, v in r.nov.items() if v != NOVYE[k]}
        return EvaluatorV2.evaluate_v2(self, {**r.pt4, **r.sp, **nov}, period)

    @staticmethod
    def _filtr6(f: dict, k: str, x, t: str) -> None:
        if k == "naklon":
            f["slope_max" if t == "kontr" else "slope_min"] = float(x)
        elif k == "tihiy_othod":
            f[k] = 1
        else:
            f[k] = float(x)
