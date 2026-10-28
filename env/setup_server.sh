#!/usr/bin/env bash
# Окружение второго раунда гибрида (R2) на сервере. Запускается НА СЕРВЕРЕ из корня копии проекта:
#   bash env/setup_server.sh            (обычно через `scripts/server.sh setup` с Мака)
# Повторный запуск безопасен: готовые шаги пропускаются.
#
# Шаги:
#   1) Python 3.9.6 (как на Маке; FINAL_DESIGN §6) — micromamba (conda-forge, без root; по умолчанию) или pyenv (PHD_PY_TOOL=pyenv);
#   2) пакеты — pip, точные версии из env/requirements_r2.txt, только колёса (--only-binary=:all:);
#   3) ссылка /Users/nilysenok/Desktop/Старое/PhD_Vol_trading -> копия проекта: в коде R2 зашиты абсолютные пути Мака
#      (l2_lib.py, harx.py, gibrid_io.py, r2_*.py), код первого раунда не меняется (протокол R2) — путь воспроизводится ссылкой (sudo один раз);
#   4) Rust 1.93.0 (как на Маке) и `cargo build --release --locked` крейта moex_iss (PHD_SKIP_RUST=1 — пропустить);
#   5) проверка версий: env/check_env.py.
# Активация для счёта — env/activate_r2.sh (в git). Сеть нужна только серверу (conda-forge/pyenv, PyPI, rustup, crates.io).
set -euo pipefail

RD="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RD"
PY_VER=3.9.6
RUST_VER=1.93.0
PHD_PY_TOOL="${PHD_PY_TOOL:-micromamba}"
export PHD_ENV_DIR="${PHD_ENV_DIR:-$HOME/.phd_r2/py$PY_VER}"
MAC_ROOT='/Users/nilysenok/Desktop/Старое/PhD_Vol_trading'

say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
die() { say "ОШИБКА: $*" >&2; exit 1; }
pyver() { "$1" -c 'import platform; print(platform.python_version())' 2>/dev/null || true; }

say "setup R2: копия $RD, $(uname -s) $(uname -m), Python через $PHD_PY_TOOL, окружение $PHD_ENV_DIR"

# ---------------------------------------------------------------- 0. системные предпосылки
miss=()
for c in curl tar bzip2 gcc make; do command -v "$c" >/dev/null 2>&1 || miss+=("$c"); done
if [ "${#miss[@]}" -gt 0 ]; then
  say "нет системных утилит: ${miss[*]}"
  say "Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y curl tar bzip2 build-essential cmake pkg-config git"
  [ "$PHD_PY_TOOL" = pyenv ] && say "для pyenv ещё: libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev tk-dev"
  die "системные утилиты"
fi
command -v cmake >/dev/null 2>&1 || say "внимание: нет cmake — может понадобиться aws-lc-sys (moex_iss): sudo apt-get install -y cmake"

# ---------------------------------------------------------------- 1. Python 3.9.6
PY="$PHD_ENV_DIR/bin/python3"
if [ -x "$PY" ] && [ "$(pyver "$PY")" = "$PY_VER" ]; then
  say "Python $PY_VER уже есть: $PY"
elif [ "$PHD_PY_TOOL" = micromamba ]; then
  MM="$HOME/.phd_r2/bin/micromamba"
  if [ ! -x "$MM" ]; then
    case "$(uname -s)-$(uname -m)" in
      Linux-x86_64) plat=linux-64 ;;
      Linux-aarch64 | Linux-arm64) plat=linux-aarch64 ;;
      Darwin-arm64) plat=osx-arm64 ;;
      Darwin-x86_64) plat=osx-64 ;;
      *) die "нет сборки micromamba для $(uname -s)-$(uname -m); попробуйте PHD_PY_TOOL=pyenv" ;;
    esac
    mkdir -p "$HOME/.phd_r2"
    say "скачиваю micromamba ($plat)"
    curl -fsSL "https://micro.mamba.pm/api/micromamba/$plat/latest" | tar -xj -C "$HOME/.phd_r2" bin/micromamba
  fi
  export MAMBA_ROOT_PREFIX="$HOME/.phd_r2/mamba"
  say "создаю окружение Python $PY_VER (conda-forge)"
  "$MM" create -y -p "$PHD_ENV_DIR" -c conda-forge --override-channels "python=$PY_VER" pip \
    || die "micromamba не создал Python $PY_VER — запустите с PHD_PY_TOOL=pyenv"
elif [ "$PHD_PY_TOOL" = pyenv ]; then
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
  if [ ! -x "$PYENV_ROOT/bin/pyenv" ]; then
    say "ставлю pyenv в $PYENV_ROOT"
    git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
  fi
  "$PYENV_ROOT/bin/pyenv" install -s "$PY_VER"
  "$PYENV_ROOT/versions/$PY_VER/bin/python3" -m venv "$PHD_ENV_DIR"
else
  die "PHD_PY_TOOL: micromamba или pyenv"
fi
[ "$(pyver "$PY")" = "$PY_VER" ] || die "в $PHD_ENV_DIR не Python $PY_VER"

# ---------------------------------------------------------------- 2. пакеты
say "ставлю пакеты из env/requirements_r2.txt (только колёса)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --only-binary=:all: -r env/requirements_r2.txt \
  || die "pip: нет колеса под эту платформу — docs/server_r2.md, раздел «Если pip не нашёл колесо»"
"$PY" -m pip freeze > env/pip_freeze_server.txt
say "pip freeze сервера -> env/pip_freeze_server.txt"

# ---------------------------------------------------------------- 3. абсолютный путь Мака
if [ "$RD" = "$MAC_ROOT" ] || [ "$(readlink -f "$MAC_ROOT" 2>/dev/null || true)" = "$(readlink -f "$RD")" ]; then
  say "путь Мака $MAC_ROOT -> $RD: есть"
elif [ -e "$MAC_ROOT" ] || [ -L "$MAC_ROOT" ]; then
  die "$MAC_ROOT уже существует и указывает не на $RD — разберитесь вручную"
elif sudo -n true 2>/dev/null; then
  say "создаю ссылку $MAC_ROOT -> $RD (sudo)"
  sudo mkdir -p "$(dirname "$MAC_ROOT")"
  sudo ln -s "$RD" "$MAC_ROOT"
else
  die "нужна ссылка (sudo с паролем — руками, один раз): sudo mkdir -p '$(dirname "$MAC_ROOT")' && sudo ln -s '$RD' '$MAC_ROOT'; потом повторите setup"
fi

# ---------------------------------------------------------------- 4. Rust и moex_iss
if [ "${PHD_SKIP_RUST:-0}" = 1 ]; then
  say "Rust пропущен (PHD_SKIP_RUST=1)"
else
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v rustup >/dev/null 2>&1; then
    say "ставлю rustup (без тулчейна по умолчанию)"
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal --no-modify-path
  fi
  rustup toolchain install "$RUST_VER" --profile minimal
  say "cargo +$RUST_VER build --release --locked (moex_iss)"
  cargo "+$RUST_VER" build --release --locked --manifest-path moex_iss/Cargo.toml
  [ -x moex_iss/target/release/moex_iss ] || die "нет moex_iss/target/release/moex_iss"
  say "moex_iss собран: moex_iss/target/release/moex_iss"
fi

# ---------------------------------------------------------------- 5. проверка
say "проверка версий"
"$PY" env/check_env.py || die "версии пакетов не совпадают с FINAL_DESIGN §6"
say "setup R2 готов. Дальше с Мака: scripts/server.sh run repro"
