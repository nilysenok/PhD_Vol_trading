#!/usr/bin/env bash
# Д24.8 / Д24.10: выгрузка баров крейтом moex_iss (только GET к iss.moex.com, 4 параллельных, пауза 100 мс) и дневная RV внешних рядов.
#   data/iss_ext  — USD000UTSTOM (CETS) и ближние BR (FORTS) с 2014-05-01 (разгон RV_m к первой строке матрицы 2014-06-09) по 2026-02-27;
#   data/iss_hist — 17 тикеров TQBR (+ EQNL для VTBR/PHOR) с 2011-12-01 по 2014-05-31.
set -u
cd "$(dirname "$0")/../.."
D=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data
BIN=moex_iss/target/release/moex_iss
mkdir -p "$D/regen/D24"
cargo build --release -j 2 --manifest-path moex_iss/Cargo.toml > "$D/regen/D24/build.log" 2>&1 || exit 1
nice "$BIN" --out "$D/iss_ext" ext-bars --what all > "$D/regen/D24/ext_bars.log" 2>&1
nice "$BIN" --out "$D/iss_ext" ext-rv --dest "$D/regen/D24/rv_ext_daily.parquet" >> "$D/regen/D24/ext_bars.log" 2>&1
nice "$BIN" --out "$D/iss_hist" hist-bars > "$D/regen/D24/hist_bars.log" 2>&1
echo done > "$D/regen/D24/bars.done"
