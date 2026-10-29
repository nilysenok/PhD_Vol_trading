import os as _os

from fastbt.cache import numba_cache_dir as _numba_cache_dir

# Д5: до первого импорта numba любым модулем быстрого слоя; явный env побеждает
_os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache_dir()))
