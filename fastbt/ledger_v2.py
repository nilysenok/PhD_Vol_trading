"""Ledger v2.0 (docs/otchety/2026-10-28_progon_etap1.md §1); данных не читает. Шард: .part -> rename -> rows -> SHA.
Читатель отказывает, а не пропускает (§1.6); метрики 2020+ не входят."""
import hashlib
import json
from pathlib import Path

import numpy as np

from fastbt.metriki_e import sr_dnevnoy

FORMAT, V = "etap1-ledger", "2.0"
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
    if q.get("vhod") == "rynok":
        q.update(ed_lim="sigma", zhizn="den", lestnica=NET, gap="vhod")
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
    """Шарп годовой шкалы (ddof = 1) по календарным кварталам 2016Q1…2019Q4; квартал без дней — nan."""
    d, x = np.asarray(daty, "datetime64[D]"), np.asarray(x, float)
    if d.shape != x.shape:
        raise ValueError("daty и x разной длины")
    god = d.astype("datetime64[Y]").astype(int) + 1970
    if not np.isin(god, GODY).all():
        raise ValueError("дата вне 2016–2019: метрики 2020+ в ledger не входят")
    kv = (d.astype("datetime64[M]").astype(int) % 12) // 3 + 1
    sel = {f"{g}Q{q}": (god == g) & (kv == q) for g in GODY for q in range(1, 5)}
    return {k: float(sr_dnevnoy(x[s]) * np.sqrt(252)) if s.any() else float("nan") for k, s in sel.items()}


def tselevaya(by_quarter: dict) -> tuple:
    """(mediana_kv, dolya_pol_kv): nan в квартале — целевая nan, и квартал не положительный."""
    v = np.array(list(by_quarter.values()), float)
    return (float("nan") if np.isnan(v).any() else float(np.median(v))), float((v > 0).sum() / len(v))


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
    def __init__(self, root: Path, sid: str, potok: int, manifest: dict, shard: int = 256):
        self.d, self.shard, self.potok, self.T = Path(root) / sid, shard, f"p{potok:02d}", int(manifest["T"])
        self.pd = self.d / self.potok
        m = {**manifest, "v": V, "format": FORMAT, "strategiya": sid}
        mp = self.d / "manifest.json"
        self.d.mkdir(parents=True, exist_ok=True)
        ex = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else m
        if any(ex.get(k) != m.get(k) for k in ("kod", "vhody", "v")):
            raise ValueError("manifest: другой kod/vhody — отказ")
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

    def add(self, row: dict, ex_port, ex_tik) -> None:
        if row.get("status") != "ok":
            ex_port, ex_tik = np.full(self.T, np.nan), np.full((NTIK, self.T), np.nan)
        if ex_port.dtype != np.float64 or ex_port.shape != (self.T,) or ex_tik.shape != (NTIK, self.T):
            raise TypeError("port: float64 [T]; tik: [17, T]")
        self.buf.append((dict(row), ex_port, ex_tik))
        if len(self.buf) >= self.shard:
            self.close()

    def close(self) -> None:
        if not self.buf:
            return
        nom, hs = self.next, {}
        arr = {"port": np.stack([b[1] for b in self.buf]), "tik": np.stack([b[2] for b in self.buf]).astype(np.float32)}
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


class Chitatel:
    def __init__(self, root: Path, sid: str, sverka_kod: bool = False):
        self.d = Path(root) / sid
        if not (self.d / "manifest.json").exists():
            raise ValueError(f"{sid}: нет manifest.json")
        self.manifest = json.loads((self.d / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("v") != V:
            raise ValueError(f"manifest.v = {self.manifest.get('v')}, ожидалось {V}")
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
            if len(chunk) < n or any(r.get("v") != V for r in chunk):
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
