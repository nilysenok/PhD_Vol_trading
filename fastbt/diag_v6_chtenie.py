"""Диагностика v6 train4 (Ж-145), лист Д-1: читатель ledger 2.2 без каталога tik/ (лёгкий забор). Только чтение.
Штатный Chitatel строк без tik/ не видит (ledger_v2._shards требует tik/<n>.npy), поэтому закрытый шард здесь — port/<n>.npy
со строкой SHA в SHA256SUMS.txt и совпавшим sha256. Хвост rows.jsonl без закрытого шарда отброшен (как Pisatel._vosstanovit).
Дубль (klyuch_n, rol) — один раз; суммы port у дубля разошлись — ValueError (мутация № 72: не молча)."""
import hashlib
import json
from pathlib import Path

import numpy as np

from fastbt.ledger_v2 import ROWS, V22, _sums, gody_perioda, proverit_stroku_22

DOPUSK = 1e-12


def _sha_ok(f: Path, h) -> bool:
    return h is not None and hashlib.sha256(f.read_bytes()).hexdigest() == h


def _zakrytye(sm: dict, pd: Path) -> list:
    """Закрытые шарды процесса: [(номер, число строк)]. Нет строки SHA port — шард не закрыт (пропуск);
    строка есть, а sha256 не совпал — отказ. Закрытые должны идти подряд с 0 (иначе строки rows.jsonl не сопоставить)."""
    out = []
    for f in sorted((pd / "port").glob("*.npy")):
        h = sm.get(f"{pd.name}/port/{f.name}")
        if h is None:
            continue
        if not _sha_ok(f, h):
            raise ValueError(f"{pd.name}/port/{f.name}: sha256 не совпал с SHA256SUMS")
        out.append((int(f.stem), int(np.load(f, mmap_mode="r").shape[0])))
    if [k for k, _ in out] != list(range(len(out))):
        raise ValueError(f"{pd.name}: закрытые шарды port не подряд с 0: {[k for k, _ in out]}")
    return out


def _sovpali(x: np.ndarray, y: np.ndarray) -> bool:
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return False
    return bool(np.isnan(x).all()) or float(np.nanmax(np.abs(x - y))) <= DOPUSK


class ChitatelPort:
    """Строки и суммы port ledger 2.2 по закрытым шардам; tik/ не нужен."""

    def __init__(self, root: Path, sid: str):
        self.d = Path(root) / sid
        mp = self.d / "manifest.json"
        if not mp.exists():
            raise ValueError(f"{sid}: нет manifest.json")
        self.manifest = json.loads(mp.read_text(encoding="utf-8"))
        if self.manifest.get("v") != V22:
            raise ValueError(f"manifest.v = {self.manifest.get('v')}, ожидалось {V22}")
        self.gody, self._mm, self._r, self._seen = gody_perioda(self.manifest.get("period")), {}, [], {}
        sm = _sums(self.d)
        for pd in sorted(p for p in self.d.glob("p*") if p.is_dir()):
            for r, loc in self._potok(sm, pd):
                self._dobavit(r, loc)

    def _potok(self, sm: dict, pd: Path):
        rp = pd / ROWS
        lines = rp.read_text(encoding="utf-8").splitlines() if rp.exists() else []
        pos = 0
        for nom, n in _zakrytye(sm, pd):             # строки сверх закрытых шардов не читаются: хвост отброшен
            chunk = [json.loads(x) for x in lines[pos:pos + n]]
            if len(chunk) < n or any(r.get("v") != V22 for r in chunk):
                raise ValueError(f"{pd.name}: строки шарда {nom} неполны или другой версии")
            for i, r in enumerate(chunk):
                proverit_stroku_22(r, self.manifest.get("setka"))
                if any(int(str(k)[:4]) not in self.gody for p in ("by_year", "by_quarter") for k in (r.get(p) or {})):
                    raise ValueError(f"by_year/by_quarter: год вне {self.gody[0]}–{self.gody[-1]}")
                yield r, (pd.name, nom, i)
            pos += n

    def _dobavit(self, r: dict, loc: tuple) -> None:
        key = (r["klyuch_n"], r["rol"])
        if key in self._seen:
            a = self._seen[key]
            if not _sovpali(self._arr(a), self._arr(loc)):
                raise ValueError(f"дубль klyuch_n {key[0]} (rol {key[1]}): суммы port разошлись > {DOPUSK}")
            return
        self._seen[key] = loc
        self._r.append((r, loc))

    def _arr(self, loc: tuple) -> np.ndarray:
        p, nom, i = loc
        if (p, nom) not in self._mm:
            self._mm[p, nom] = np.load(self.d / p / "port" / f"{nom:05d}.npy", mmap_mode="r")
        return np.asarray(self._mm[p, nom][i], float)

    def rows(self, status="ok") -> list:
        return [r for r, _ in self._r if status is None or r.get("status") == status]

    def summy(self, rows=None) -> np.ndarray:
        """port → [n, 16, NF] по порядку rows (по умолчанию все ok)."""
        by = {(r["klyuch_n"], r["rol"]): loc for r, loc in self._r}
        return np.stack([self._arr(by[r["klyuch_n"], r["rol"]]) for r in (self.rows() if rows is None else rows)])
