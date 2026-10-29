"""Ledger v2.0 (docs/otchety/2026-10-28_progon_etap1.md §1); данных не читает. Шард: .part -> rename -> rows -> SHA.
Читатель отказывает, а не пропускает (§1.6); метрики 2020+ не входят.
v2.1 (Л7, docs/otchety/2026-10-29_setka_v5.md §3): вместо рядов port[T], tik[17,T] — суммы сегментов F_S…F_X
(fastbt.stats) по кварталам 2016Q1…2019Q4: port [16,NF], tik [17,16,NF] float64, ≈ 20 КБ/точку; окно — stats.window."""
import hashlib
import json
from pathlib import Path

import numpy as np

from fastbt import stats
from fastbt.metriki_e import sr_dnevnoy

FORMAT, V, V21 = "etap1-ledger", "2.0", "2.1"
NSEG = 16
NET, NTIK, GODY = "net", 17, (2016, 2017, 2018, 2019)
SUMS, ROWS = "SHA256SUMS.txt", "rows.jsonl"


def tok(v) -> str:
    return v if isinstance(v, str) else repr(float(v))


def canon_v2(sid: str, pt: dict) -> dict:
    q = dict(pt)
    if q.get("k_s") == NET:
        q.update(R=NET, k_tr=NET, bezubytok=NET)
    else:
        q["k_x"] = NET
    if q.get("tejk_signal", NET) != NET:
        q.update(R=NET, k_x=NET)                    # Ж-084: один тейк — от уровня сигнала
    if q.get("vhod_tip") == "stop_zayavka":         # Ж-091: вход заявкой по касанию, не лимитом k_e и не закрытием
        if sid not in ("S3", "S4"):
            raise ValueError(f"{sid}: vhod_tip = stop_zayavka только у S3 и S4")
        q["vhod"] = "rynok"                         # → ed_lim, zhizn, lestnica, gap схлопываются правилом ниже
        q.update({k: v for k, v in (("q_sily", 0.0), ("proverka", "bars5m")) if k in q})   # сила и close-проверка не действуют
        if q.get("storona") == "short_shag1":
            q["storona"] = "obe"
    if q.get("tejk_signal") == "rodnoy_uroven":     # Ж-091 п. 1.1: сигнальный выход выключен, τ остаётся и идёт по open2 — t_vyhod схлопнут
        q.update(t_vyhod="open2")
    if q.get("vhod") == "rynok":
        q.update(ed_lim="sigma", zhizn="den", lestnica=NET, gap="vhod")
    if q.get("proverka") == "zakrytie" and str(q.get("t_vyhod", "close")) != "close":
        q["proverka"] = "bars5m"                    # выход в 15:00/17:00/open2 существует только на барах (exec_e._check_params)
    if sid == "S5" and q.get("storona") == "short_shag1":
        q["storona"] = "obe"
    return q


def klyuch(sid: str, pt: dict, period: str, komissiya: float, fdr) -> dict:
    kl = {k: tok(v) for k, v in canon_v2(sid, pt).items()}
    kl.update(period=tok(period), komissiya=tok(komissiya), fdr="null" if fdr is None else tok(fdr))
    return kl


def klyuch_sha(kl: dict, kod: str, vhody: str) -> str:
    kan = json.dumps(kl, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"{kan}|{kod}|{vhody}|{V}".encode("utf-8")).hexdigest()


def kvartaly(daty: np.ndarray, x: np.ndarray) -> dict:
    """Шарп годовой шкалы (ddof = 1) по календарным кварталам 2016Q1…2019Q4; квартал без дней — nan.
    Защита 2016–2019 для ledger train4."""
    return kvartaly_gody(daty, x, GODY)


def kvartaly_gody(daty: np.ndarray, x: np.ndarray, gody: tuple) -> dict:
    """То же по кварталам заданных годов; gody ≠ GODY — только справочные периоды вне ledger train4 (test1)."""
    d, x = np.asarray(daty, "datetime64[D]"), np.asarray(x, float)
    if d.shape != x.shape:
        raise ValueError("daty и x разной длины")
    god = d.astype("datetime64[Y]").astype(int) + 1970
    if not np.isin(god, gody).all():
        raise ValueError("дата вне 2016–2019: метрики 2020+ в ledger не входят" if gody == GODY else f"дата вне годов {gody}")
    kv = (d.astype("datetime64[M]").astype(int) % 12) // 3 + 1
    sel = {f"{g}Q{q}": (god == g) & (kv == q) for g in gody for q in range(1, 5)}
    return {k: float(sr_dnevnoy(x[s]) * np.sqrt(252)) if s.any() else float("nan") for k, s in sel.items()}


def tselevaya(by_quarter: dict) -> tuple:
    """(mediana_kv, dolya_pol_kv): nan в квартале — целевая nan, и квартал не положительный."""
    v = np.array(list(by_quarter.values()), float)
    return (float("nan") if np.isnan(v).any() else float(np.median(v))), float((v > 0).sum() / len(v))


def segment_kvartala(daty: np.ndarray) -> np.ndarray:
    """Номер квартала 0…15 (2016Q1 = 0) для каждой даты; дата вне 2016–2019 — отказ."""
    d = np.asarray(daty, "datetime64[D]")
    god = d.astype("datetime64[Y]").astype(int) + 1970
    if not np.isin(god, GODY).all():
        raise ValueError("дата вне 2016–2019: метрики 2020+ в ledger не входят")
    return (god - GODY[0]) * 4 + (d.astype("datetime64[M]").astype(int) % 12) // 3


def nbars_kvartalov(daty: np.ndarray) -> np.ndarray:
    return np.bincount(segment_kvartala(daty), minlength=NSEG).astype(float)


def segment_do_goda(t: int) -> int:
    """Индекс j последнего сегмента окна [2016-01-01; 31.12(t−1)] для stats.window: Q4 года t−1."""
    if t - 1 not in GODY:
        raise ValueError(f"окно до конца {t - 1} года вне 2016–2019")
    return (t - 1 - GODY[0]) * 4 + 3


def summy(daty: np.ndarray, x: np.ndarray, poz=None, sdelki=None) -> np.ndarray:
    """Суммы сегментов F_S…F_X по кварталам из ряда x[..., T] (избыточная доходность после комиссий) → [..., 16, NF].
    F_S, F_Q — суммы x и x²; F_MX, F_MN, F_D — моноид просадки по накопленной сумме квартала (как stats._bar); F_L — x последнего дня.
    poz[..., T] — |позиция| (необязательно): F_X = она на последнем дне квартала (выход на конце окна, мутация № 6), F_E = дней в позиции.
    sdelki[..., T] — число сделок в день (необязательно): F_R = сумма за квартал. Без poz F_X = F_E = 0, без sdelki F_R = 0."""
    x = np.asarray(x, float)
    seg = segment_kvartala(daty)
    if seg.shape != x.shape[-1:] or any(a is not None and np.shape(a) != x.shape for a in (poz, sdelki)):
        raise ValueError("daty, x, poz разной длины")
    out = np.zeros(x.shape[:-1] + (NSEG, stats.NF))
    for k in range(NSEG):
        m = seg == k
        if not m.any():
            continue
        xs = x[..., m]
        c = np.cumsum(xs, axis=-1)
        out[..., k, stats.F_S], out[..., k, stats.F_Q] = xs.sum(-1), (xs * xs).sum(-1)
        out[..., k, stats.F_MX], out[..., k, stats.F_MN] = c.max(-1), c.min(-1)
        out[..., k, stats.F_D] = np.minimum(0.0, (c - np.maximum.accumulate(c, axis=-1)).min(-1))
        out[..., k, stats.F_L] = xs[..., -1]
        if poz is not None:
            ps = np.abs(np.asarray(poz, float)[..., m])
            out[..., k, stats.F_X], out[..., k, stats.F_E] = ps[..., -1], (ps > 0).sum(-1)
        if sdelki is not None:
            out[..., k, stats.F_R] = np.asarray(sdelki, float)[..., m].sum(-1)
    return out


def _sums(d: Path) -> dict:
    p = d / SUMS
    sp = (ln.split() for ln in p.read_text(encoding="utf-8").splitlines()) if p.exists() else ()
    return {s[1]: s[0] for s in sp if len(s) > 1}


def _shards(d: Path, pd: Path) -> list:
    """Закрытые шарды процесса: обе строки SHA есть; (номер, число строк)."""
    sm, out = _sums(d), []
    for f in sorted((pd / "port").glob("*.npy")):
        k = f"{pd.name}/%s/{f.name}"
        if k % "port" in sm and k % "tik" in sm and (pd / "tik" / f.name).exists():  # обе строки SHA
            out.append((int(f.stem), int(np.load(f, mmap_mode="r").shape[0])))
    return out


class Pisatel:
    """Один на процесс: p{NN}/ свой. Строки видны только после закрытия шарда."""
    VERSIYA, daty = V, None      # у Pisatel21 — 2.1 и даты рядов; версия из manifest (резюм по каталогу) приоритетнее

    def __init__(self, root: Path, sid: str, potok: int, manifest: dict, shard: int = 256):
        self.d, self.shard, self.potok, self.T = Path(root) / sid, shard, f"p{potok:02d}", int(manifest["T"])
        self.pd, self.v = self.d / self.potok, manifest.get("v") or self.VERSIYA
        if self.v not in (V, V21):
            raise ValueError(f"ledger v = {self.v}: известны {V}, {V21}")
        m = {**manifest, "v": self.v, "format": FORMAT, "strategiya": sid}
        mp = self.d / "manifest.json"
        self.d.mkdir(parents=True, exist_ok=True)
        ex = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else m
        if any(ex.get(k) != m.get(k) for k in ("kod", "vhody", "v")) or ex.get("poz_konec", False) != m.get("poz_konec", False):
            raise ValueError("manifest: другой kod/vhody/poz_konec — отказ")
        self.poz_konec = bool(ex.get("poz_konec"))
        if not mp.exists():
            mp.write_text(json.dumps(ex, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
        [(self.pd / s).mkdir(parents=True, exist_ok=True) for s in ("port", "tik")]
        self.buf = []
        self._vosstanovit()

    def _vosstanovit(self):
        """Обрыв: файлы без строки SHA удаляются, rows.jsonl обрезается до закрытых шардов."""
        zak = _shards(self.d, self.pd)
        nom = {k for k, _ in zak}
        for f in [g for s in ("port", "tik") for g in (self.pd / s).iterdir()]:
            if not f.name.endswith(".npy") or int(f.stem) not in nom:
                f.unlink()
        rp = self.pd / ROWS
        lines = rp.read_text(encoding="utf-8").splitlines() if rp.exists() else []
        lines = lines[: sum(n for _, n in zak)]
        rp.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        self.next, self._done = max(nom, default=-1) + 1, {json.loads(ln)["klyuch_sha"] for ln in lines}

    def done(self) -> set:
        return set(self._done)

    _poz = None         # позиции строки — задаёт Pisatel21.add (сигнатура v2.0 add не меняется, IG-контракт)

    def add(self, row: dict, ex_port, ex_tik) -> None:
        poz, self._poz = self._poz, None
        if row.get("status") != "ok":
            ex_port, ex_tik = np.full(self.T, np.nan), np.full((NTIK, self.T), np.nan)
            poz = (None,) * 4
        if ex_port.dtype != np.float64 or ex_port.shape != (self.T,) or ex_tik.shape != (NTIK, self.T):
            raise TypeError("port: float64 [T]; tik: [17, T]")
        if self.v == V21:
            if self.daty is None or self.daty.shape != (self.T,):
                raise ValueError("ledger 2.1: нужны daty [T] у писателя")
            if self.poz_konec and poz is None:
                raise ValueError("ledger 2.1 с poz_konec: нужны |позиции| и сделки (poz)")
            pp, pt, sp, st = poz or (None,) * 4
            ex_port, ex_tik = summy(self.daty, ex_port, pp, sp), summy(self.daty, np.asarray(ex_tik, float), pt, st)   # nan у не-ok остаётся nan
        self.buf.append((dict(row), ex_port, ex_tik))
        if len(self.buf) >= self.shard:
            self.close()

    def close(self) -> None:
        if not self.buf:
            return
        nom, hs = self.next, {}
        tk = np.stack([b[2] for b in self.buf])
        arr = {"port": np.stack([b[1] for b in self.buf]), "tik": tk if self.v == V21 else tk.astype(np.float32)}
        for sub, a in arr.items():
            p = self.pd / sub / f"{nom:05d}.npy"
            with open(f"{p}.part", "wb") as f:
                np.save(f, a)
            Path(f"{p}.part").rename(p)
            hs[sub] = f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {self.potok}/{sub}/{p.name}\n"
        for i, (r, _, _) in enumerate(self.buf):
            r["excess"] = {"potok": self.potok, "shard": nom, "stroka": i} if r.get("status") == "ok" else None
        with open(self.pd / ROWS, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r, _, _ in self.buf)
        with open(self.d / SUMS, "a", encoding="utf-8") as f:
            f.writelines(hs.values())
        self._done |= {r["klyuch_sha"] for r, _, _ in self.buf}
        self.next, self.buf = nom + 1, []


class Pisatel21(Pisatel):
    """ledger 2.1: add() принимает те же ряды, пишет суммы сегментов по кварталам; daty [T] — даты рядов."""
    VERSIYA = V21

    def __init__(self, root: Path, sid: str, potok: int, manifest: dict, shard: int = 256, daty=None, poz: bool = False):
        if daty is not None:
            self.daty = np.asarray(daty, "datetime64[D]")
            manifest = {**manifest, "nbars": nbars_kvartalov(self.daty).tolist(), "poz_konec": bool(poz)}
        super().__init__(root, sid, potok, manifest, shard)

    def add(self, row: dict, ex_port, ex_tik, poz=None) -> None:
        """poz — (|pos| порт [T], |pos| тик [17, T], сделок порт [T], сделок тик [17, T]); обязательна при poz_konec для status ok."""
        self._poz = poz
        super().add(row, ex_port, ex_tik)


class Chitatel:
    def __init__(self, root: Path, sid: str, sverka_kod: bool = False):
        self.d = Path(root) / sid
        if not (self.d / "manifest.json").exists():
            raise ValueError(f"{sid}: нет manifest.json")
        self.manifest = json.loads((self.d / "manifest.json").read_text(encoding="utf-8"))
        self.v = self.manifest.get("v")
        if self.v not in (V, V21):
            raise ValueError(f"manifest.v = {self.v}, ожидалось {V} или {V21}")
        self._mm, self._r, seen = {}, [], set()
        for pd in sorted(p for p in self.d.glob("p*") if p.is_dir()):
            for r, loc in self._potok(pd):
                if r["klyuch_sha"] not in seen:  # дубль: первая строка
                    seen.add(r["klyuch_sha"])
                    self._r.append((r, loc))
        for pole in ("kod", "vhody"):
            if len({r[pole] for r, _ in self._r} | {self.manifest[pole]}) > 1 and not (sverka_kod and pole == "kod"):
                raise ValueError(f"в каталоге разные {pole}")

    def _potok(self, pd: Path):
        sm, rp, out = _sums(self.d), pd / ROWS, []
        lines = rp.read_text(encoding="utf-8").splitlines() if rp.exists() else []
        for sub in ("port", "tik"):
            for f in sorted((pd / sub).glob("*.npy")):
                h = sm.get(f"{pd.name}/{sub}/{f.name}")
                if h is None or hashlib.sha256(f.read_bytes()).hexdigest() != h:
                    raise ValueError(f"{pd.name}/{sub}/{f.name}: нет строки SHA256SUMS или sha256 не совпал")
        pos = 0
        for nom, n in _shards(self.d, pd):
            chunk = [json.loads(x) for x in lines[pos:pos + n]]
            if len(chunk) < n or any(r.get("v") != self.v for r in chunk):
                raise ValueError(f"{pd.name}: строки шарда {nom} неполны или другой версии")
            for i, r in enumerate(chunk):
                if any(int(str(k)[:4]) not in GODY for p in ("by_year", "by_quarter") for k in (r.get(p) or {})):
                    raise ValueError("by_year/by_quarter: год вне 2016–2019")
                out.append((r, (pd.name, nom, i)))
            pos += n
        return out

    def rows(self, status="ok") -> list:
        return [r for r, _ in self._r if status is None or r.get("status") == status]

    def n_unique(self) -> int:
        return len({r["klyuch_sha"] for r in self.rows()})

    def _arr(self, sub, loc):
        p, nom, i = loc
        if (p, sub, nom) not in self._mm:
            self._mm[p, sub, nom] = np.load(self.d / p / sub / f"{nom:05d}.npy", mmap_mode="r")
        return self._mm[p, sub, nom][i]

    def _loc(self, rows):
        by = {r["klyuch_sha"]: loc for r, loc in self._r}
        return [by[r["klyuch_sha"]] for r in (self.rows() if rows is None else rows)]

    def matrica(self, rows=None) -> np.ndarray:
        return np.stack([np.asarray(self._arr("port", x), float) for x in self._loc(rows)], axis=1)

    def tik(self, rows=None) -> np.ndarray:
        return np.stack([self._arr("tik", x) for x in self._loc(rows)])

    def summy(self, rows=None, sub="port") -> np.ndarray:
        """v2.1: port → [n, 16, NF]; tik → [n, 17, 16, NF]."""
        if self.v != V21:
            raise ValueError("summy: только ledger 2.1")
        return np.stack([np.asarray(self._arr(sub, x), float) for x in self._loc(rows)])

    def okno(self, t: int, rows=None, sub="port", comm: float = 0.0) -> tuple:
        """(sharpe, maxdd, exposure_pct, trades_per_year) окна [2016-01-01; 31.12(t−1)] — stats.window на суммах сегментов.
        comm > 0 требует F_X (выход на конце окна): без неё — отказ, а не молчаливое «без выхода»."""
        if comm and not self.manifest.get("poz_konec"):
            raise ValueError("okno: comm > 0, а |позиция| на конце кварталов в ledger не записана (poz_konec = false)")
        nb = np.asarray(self.manifest["nbars"], float)
        return stats.window(self.summy(rows, sub), nb, segment_do_goda(t), comm)
