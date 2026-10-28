# Активация окружения R2 на сервере: `source env/activate_r2.sh` (так делает `scripts/server.sh run`).
# Окружение создаёт env/setup_server.sh; путь — PHD_ENV_DIR (по умолчанию ~/.phd_r2/py3.9.6).
PHD_ENV_DIR="${PHD_ENV_DIR:-$HOME/.phd_r2/py3.9.6}"
if [ ! -x "$PHD_ENV_DIR/bin/python3" ]; then
  echo "нет окружения R2 в $PHD_ENV_DIR — сначала scripts/server.sh setup" >&2
  return 1 2>/dev/null || exit 1
fi
export PATH="$PHD_ENV_DIR/bin:$HOME/.cargo/bin:$PATH"
export PHD_R2_SERVER=1
export TZ=Europe/Moscow            # время в логах — как на Маке
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
# BLAS/OpenMP — один поток на процесс: параллелизм даёт --workers (multiprocessing),
# бустинг и так однопоточный (l2_lib.fit: XGBoost nthread=1, LightGBM num_threads=1, force_col_wise=True).
export OMP_NUM_THREADS="${PHD_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${PHD_THREADS:-1}"
export MKL_NUM_THREADS="${PHD_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${PHD_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${PHD_THREADS:-1}"
