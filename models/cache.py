"""Кэш стадий: ключ = sha256(стадия + входы + параметры + git-версии файлов стадии); каталог стадии считается готовым по done.json.
Прогнозы: val -> data/models_cache/val/<вид>/<ключ>.parquet, test -> data/models_cache/test_locked/<вид>/<ключ>.parquet."""
import hashlib, json, os, time
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from . import config as C


def key(stage, **parts):
    s = json.dumps({'stage': stage, **parts}, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def stage_dir(stage, k):
    return C.CACHE / 'stages' / stage / k


def done(stage, k):
    return (stage_dir(stage, k) / 'done.json').exists()


def mark(stage, k, **meta):
    d = stage_dir(stage, k)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / 'done.json.tmp'
    tmp.write_text(json.dumps(meta, default=str))
    os.replace(tmp, d / 'done.json')


def meta(stage, k):
    return json.loads((stage_dir(stage, k) / 'done.json').read_text())


def pred_path(scope, kind, k):
    root = C.CACHE / ('val' if scope == 'val' else 'test_locked') / kind
    root.mkdir(parents=True, exist_ok=True)
    return root / f'{k}.parquet'


def write_atomic(df, path):
    tmp = Path(str(path) + f'.{os.getpid()}.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_val(kind, k):
    return pd.read_parquet(pred_path('val', kind, k))


def file_digest(paths):
    """Содержимое входных файлов (sha1 по байтам) — входы стадии «данные»."""
    h = hashlib.sha1()
    for p in sorted(map(str, paths)):
        h.update(p.rsplit('/', 1)[-1].encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


class Timer:
    """Секунды и попадания в кэш по стадиям (для таблицы «стадия -> секунды»)."""

    def __init__(self):
        self.sec, self.hits, self.miss = {}, {}, {}

    @contextmanager
    def stage(self, name):
        t0 = time.time()
        yield
        self.sec[name] = self.sec.get(name, 0.0) + time.time() - t0

    def add(self, name, sec=0.0, hit=None):
        self.sec[name] = self.sec.get(name, 0.0) + sec
        if hit is not None:
            d = self.hits if hit else self.miss
            d[name] = d.get(name, 0) + 1

    def table(self):
        names = list(self.sec)
        return pd.DataFrame({'stage': names, 'sec': [round(self.sec[n], 2) for n in names],
                             'cache_hit': [self.hits.get(n, 0) for n in names], 'computed': [self.miss.get(n, 0) for n in names]})
