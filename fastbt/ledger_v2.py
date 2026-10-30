"""Ledger v2.0 (docs/otchety/2026-10-28_progon_etap1.md §1); данных не читает. Шард: .part -> rename -> rows -> SHA.
Читатель отказывает, а не пропускает (§1.6); метрики 2020+ не входят.
v2.1 (Л7, docs/otchety/2026-10-29_setka_v5.md §3): вместо рядов port[T], tik[17,T] — суммы сегментов F_S…F_X
(fastbt.stats) по кварталам 2016Q1…2019Q4: port [16,NF], tik [17,16,NF] float64, ≈ 20 КБ/точку; окно — stats.window.
v2.2 (V6-15, docs/otchety/2026-10-29_setka_v6_inventarizaciya.md §4, Ж-132 п. 4): тело как 2.1 + в строке setka, rol, klyuch_n.
klyuch_sha (с хэшем кода и данных, Д5, Ж-131 п. 4) не меняется; klyuch_n — тождество точки для счёта N, kod/vhody остаются в каждой строке."""
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from fastbt import period_v2, stats
from fastbt.metriki_e import sr_dnevnoy

FORMAT, V, V21, V22 = "etap1-ledger", "2.0", "2.1", "2.2"
SUMMY_V = (V21, V22)                                # версии с суммами сегментов
ROLI, KONTROL, SPRAVKA = ("sobol", "morris", "faza2", "faza3", "kontrol", "spravka"), "kontrol", "spravka"
VNE_N = (KONTROL, SPRAVKA)                          # роли вне N: контроль и справочная строка (выход по close, Ж-137 п. 6)
NSEG = 16
NET, NTIK, GODY = "net", 17, (2016, 2017, 2018, 2019)
GODY_V2 = (2015,) + GODY                            # период В2 (Ж-113): окно с 2015-01-01, 20 кварталов
SUMS, ROWS = "SHA256SUMS.txt", "rows.jsonl"


def gody_perioda(period=None) -> tuple:
    """Годы ledger по periodu из manifest (таблица — period_v2.PERIODY): v2 — 2015–2019, wf — 2015–2026, иначе (train4, поля нет) — 2016–2019."""
    return period_v2.gody(period) if period in period_v2.PERIODY else GODY


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


def klyuch_n(sid: str, kl: dict) -> str:
    """Тождество точки для N: strategiya + канонический klyuch (точка, период, комиссия, fdr), без kod, vhody и версии формата."""
    kan = json.dumps({"strategiya": sid, "klyuch": kl}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(kan.encode("utf-8")).hexdigest()


def proverit_stroku_22(r: dict, setka=None) -> None:
    """Строка 2.2: setka, rol ∈ ROLI, kod и vhody непусты (хэш кода и данных в каждой строке), klyuch_n сходится с klyuch."""
    for pole in ("kod", "vhody", "setka"):
        if not isinstance(r.get(pole), str) or not r[pole]:
            raise ValueError(f"ledger 2.2: в строке нет {pole}")
    if r.get("rol") not in ROLI:
        raise ValueError(f"ledger 2.2: rol = {r.get('rol')!r}, ожидалось одно из {ROLI}")
    if setka is not None and r["setka"] != setka:
        raise ValueError("ledger 2.2: setka строки ≠ manifest")
    if r.get("klyuch_n") != klyuch_n(r.get("strategiya"), r.get("klyuch")):
        raise ValueError("ledger 2.2: klyuch_n не сходится с klyuch")
    if r.get("agr") is not None:                    # v7 (Ж-155 п. 3): необязательное поле агрегатов, формат — fastbt.agregaty_v7
        a, ng = r["agr"], len(r["agr"].get("gody", ()))
        if not ng or [len(x) for x in a.get("d", ())] != [ng] * 5 or [len(x) for x in a.get("w", ())] != [ng] * 5 \
                or [len(x) for x in a.get("r", ())] != [3] * 5:
            raise ValueError("ledger 2.2: поле agr неверной формы")


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


def segment_kvartala(daty: np.ndarray, gody: tuple = GODY) -> np.ndarray:
    """Номер квартала 0…4·len(gody)−1 (первый квартал первого года = 0) для каждой даты; дата вне годов — отказ."""
    d = np.asarray(daty, "datetime64[D]")
    god = d.astype("datetime64[Y]").astype(int) + 1970
    if not np.isin(god, gody).all():
        raise ValueError(f"дата вне {gody[0]}–{gody[-1]}: метрики 2020+ в ledger не входят")
    return (god - gody[0]) * 4 + (d.astype("datetime64[M]").astype(int) % 12) // 3


def nbars_kvartalov(daty: np.ndarray, gody: tuple = GODY) -> np.ndarray:
    return np.bincount(segment_kvartala(daty, gody), minlength=4 * len(gody)).astype(float)


def segment_do_goda(t: int, gody: tuple = GODY) -> int:
    """Индекс j последнего сегмента окна [1 января первого года; 31.12(t−1)] для stats.window: Q4 года t−1."""
    if t - 1 not in gody:
        raise ValueError(f"окно до конца {t - 1} года вне {gody[0]}–{gody[-1]}")
    return (t - 1 - gody[0]) * 4 + 3


def summy(daty: np.ndarray, x: np.ndarray, poz=None, sdelki=None, gody: tuple = GODY) -> np.ndarray:
    """Суммы сегментов F_S…F_X по кварталам из ряда x[..., T] (избыточная доходность после комиссий) → [..., 16, NF].
    F_S, F_Q — суммы x и x²; F_MX, F_MN, F_D — моноид просадки по накопленной сумме квартала (как stats._bar); F_L — x последнего дня.
    poz[..., T] — |позиция| (необязательно): F_X = она на последнем дне квартала (выход на конце окна, мутация № 6), F_E = дней в позиции.
    sdelki[..., T] — число сделок в день (необязательно): F_R = сумма за квартал. Без poz F_X = F_E = 0, без sdelki F_R = 0."""
    x = np.asarray(x, float)
    seg, nseg = segment_kvartala(daty, gody), 4 * len(gody)
    if seg.shape != x.shape[-1:] or any(a is not None and np.shape(a) != x.shape for a in (poz, sdelki)):
        raise ValueError("daty, x, poz разной длины")
    out = np.zeros(x.shape[:-1] + (nseg, stats.NF))
    for k in range(nseg):
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


def _zapis_atomarno(mp: Path, text: str) -> None:
    """Уникальный tmp рядом + os.replace: читатель видит либо нет файла, либо целый (гонка первых кусков)."""
    tmp = mp.with_name(f"{mp.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, mp)


class Pisatel:
    """Один на процесс: p{NN}/ свой. Строки видны только после закрытия шарда."""
    VERSIYA, daty = V, None      # у Pisatel21 — 2.1 и даты рядов; версия из manifest (резюм по каталогу) приоритетнее

    def __init__(self, root: Path, sid: str, potok: int, manifest: dict, shard: int = 256):
        self.d, self.shard, self.potok, self.T = Path(root) / sid, shard, f"p{potok:02d}", int(manifest["T"])
        self.pd, self.v = self.d / self.potok, manifest.get("v") or self.VERSIYA
        if self.v not in (V, V21, V22):
            raise ValueError(f"ledger v = {self.v}: известны {V}, {V21}, {V22}")
        if self.v == V22 and not manifest.get("setka"):
            raise ValueError("ledger 2.2: в manifest нужна setka")
        m = {**manifest, "v": self.v, "format": FORMAT, "strategiya": sid}
        mp = self.d / "manifest.json"
        self.d.mkdir(parents=True, exist_ok=True)
        ex = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else m
        if any(ex.get(k) != m.get(k) for k in ("kod", "vhody", "v", "setka")) or ex.get("poz_konec", False) != m.get("poz_konec", False):
            raise ValueError("manifest: другой kod/vhody/poz_konec — отказ")
        self.poz_konec, self.period, self.setka = bool(ex.get("poz_konec")), ex.get("period"), ex.get("setka")
        if not mp.exists():
            _zapis_atomarno(mp, json.dumps(ex, ensure_ascii=False, sort_keys=True, indent=1))
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
        if self.v == V22:
            proverit_stroku_22(row, self.setka)
        if self.v in SUMMY_V:
            if self.daty is None or self.daty.shape != (self.T,):
                raise ValueError("ledger 2.1: нужны daty [T] у писателя")
            if self.poz_konec and poz is None:
                raise ValueError("ledger 2.1 с poz_konec: нужны |позиции| и сделки (poz)")
            pp, pt, sp, st = poz or (None,) * 4
            g = gody_perioda(self.period)
            ex_port, ex_tik = summy(self.daty, ex_port, pp, sp, g), summy(self.daty, np.asarray(ex_tik, float), pt, st, g)   # nan у не-ok остаётся nan
        self.buf.append((dict(row), ex_port, ex_tik))
        if len(self.buf) >= self.shard:
            self.close()

    def close(self) -> None:
        if not self.buf:
            return
        nom, hs = self.next, {}
        tk = np.stack([b[2] for b in self.buf])
        arr = {"port": np.stack([b[1] for b in self.buf]), "tik": tk if self.v in SUMMY_V else tk.astype(np.float32)}
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
            manifest = {**manifest, "nbars": nbars_kvartalov(self.daty, gody_perioda(manifest.get("period"))).tolist(), "poz_konec": bool(poz)}
        super().__init__(root, sid, potok, manifest, shard)

    def add(self, row: dict, ex_port, ex_tik, poz=None) -> None:
        """poz — (|pos| порт [T], |pos| тик [17, T], сделок порт [T], сделок тик [17, T]); обязательна при poz_konec для status ok."""
        self._poz = poz
        super().add(row, ex_port, ex_tik)


class Pisatel22(Pisatel21):
    """ledger 2.2: как 2.1; manifest["setka"] обязательна, каждая строка несёт setka, rol, klyuch_n, kod, vhody (проверка в add)."""
    VERSIYA = V22


class Chitatel:
    def __init__(self, root: Path, sid: str, sverka_kod: bool = False):
        self.d = Path(root) / sid
        if not (self.d / "manifest.json").exists():
            raise ValueError(f"{sid}: нет manifest.json")
        self.manifest = json.loads((self.d / "manifest.json").read_text(encoding="utf-8"))
        self.v, self.gody = self.manifest.get("v"), gody_perioda(self.manifest.get("period"))
        if self.v not in (V, V21, V22):
            raise ValueError(f"manifest.v = {self.v}, ожидалось {V}, {V21} или {V22}")
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
                if self.v == V22:
                    proverit_stroku_22(r, self.manifest.get("setka"))
                if any(int(str(k)[:4]) not in self.gody for p in ("by_year", "by_quarter") for k in (r.get(p) or {})):
                    raise ValueError(f"by_year/by_quarter: год вне {self.gody[0]}–{self.gody[-1]}")
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
        if self.v not in SUMMY_V:
            raise ValueError("summy: только ledger 2.1 и 2.2")
        return np.stack([np.asarray(self._arr(sub, x), float) for x in self._loc(rows)])

    def klyuch_n_stroki(self, r: dict) -> str:
        """klyuch_n строки: у 2.2 записан (и сверен при чтении), у 2.0/2.1 (v5) считается из klyuch."""
        return r["klyuch_n"] if self.v == V22 else klyuch_n(r["strategiya"], r["klyuch"])

    def okno(self, t: int, rows=None, sub="port", comm: float = 0.0) -> tuple:
        """(sharpe, maxdd, exposure_pct, trades_per_year) окна [2016-01-01; 31.12(t−1)] — stats.window на суммах сегментов.
        comm > 0 требует F_X (выход на конце окна): без неё — отказ, а не молчаливое «без выхода»."""
        if comm and not self.manifest.get("poz_konec"):
            raise ValueError("okno: comm > 0, а |позиция| на конце кварталов в ledger не записана (poz_konec = false)")
        nb = np.asarray(self.manifest["nbars"], float)
        return stats.window(self.summy(rows, sub), nb, segment_do_goda(t, self.gody), comm)


class Obedinenie:
    """Читатель объединения ledger v5 (2.1) ∪ v6 (2.2) одной стратегии (Ж-132 п. 4): N_k = |уникальных klyuch_n|, роль kontrol не в счёте.
    Дубль klyuch_n — первая строка по порядку читателей; общая матрица сумм — по тем же ключам. Ledger 2.0 не объединяется (нет сумм)."""

    def __init__(self, *chitateli: "Chitatel"):
        if not chitateli:
            raise ValueError("Obedinenie: нет читателей")
        if any(c.v not in SUMMY_V for c in chitateli):
            raise ValueError("Obedinenie: только ledger 2.1 и 2.2")
        if len({c.manifest.get("strategiya") for c in chitateli}) > 1 or len({tuple(c.manifest["nbars"]) for c in chitateli}) > 1:
            raise ValueError("Obedinenie: разные strategiya или nbars (период)")
        seen, self._poz = {}, []
        for c in chitateli:
            for r in c.rows():
                n = c.klyuch_n_stroki(r)
                if r.get("rol") in VNE_N:
                    continue
                if n in seen:
                    self._sverit_dubl(n, seen[n], (c, r))
                else:
                    seen[n] = (c, r)
                    self._poz.append((n, c, r))

    @staticmethod
    def _sverit_dubl(n: str, a: tuple, b: tuple, dopusk: float = 1e-12) -> None:
        """Дубль klyuch_n: суммы сегментов (port и tik) совпали ≤ dopusk — один раз; иначе код изменил точку v5 — отказ."""
        d = 0.0
        for sub in ("port", "tik"):
            x, y = (np.asarray(c.summy([r], sub)[0], float) for c, r in (a, b))
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                d = float("inf")
            elif not np.isnan(x).all():
                d = max(d, float(np.nanmax(np.abs(x - y))))
        if d > dopusk:
            raise ValueError(f"Obedinenie: дубль klyuch_n {n} (Sk {a[1].get('strategiya')}) — суммы разошлись: "
                             f"kod {a[1]['kod']} ≠ {b[1]['kod']}, max|Δ| = {d:.3e} > {dopusk}")

    def klyuchi(self) -> list:
        return [n for n, _, _ in self._poz]

    def n_k(self) -> int:
        return len(self._poz)

    def summy(self, sub="port") -> np.ndarray:
        """Общая матрица сумм по klyuchi(): port → [N_k, 16·, NF]; tik → [N_k, 17, 16·, NF]."""
        return np.stack([np.asarray(c.summy([r], sub)[0], float) for _, c, r in self._poz])
