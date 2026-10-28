//! Доступ (только чтение) к хранимым данным основной копии репозитория.
//!
//! Корректировки корпоративных событий. В пайплайне цен НЕТ ни сплит-, ни дивидендных поправок
//! (поиск по moex_discovery/src и moex_volatility/src: только карантин плохих баров):
//! moex_discovery/src/pipeline_d001_d003.py:201-246 (detect_quarantine_flags: нулевой объём,
//! |r|>15%, |r|>10σ скользящих 100 баров, битый OHLC, дыра по времени >10 мин внутри 10-19 ч;
//! дубли begin — keep first). Бары-«карантин» лежат в data/quarantine/candles_10m; хранимые
//! признаки считались по остальным (data/dataset_final/01_stocks/candles_10m).
//! Поэтому «те же корректировки» для ISS-баров = выбросить бары из карантина и дубли.
use crate::model::Bar;
use crate::store;
use anyhow::Result;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

pub const DEFAULT_ROOT: &str = "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading";

#[derive(Clone)]
pub struct Stored {
    pub root: PathBuf,
}

impl Stored {
    pub fn new(root: &str) -> Self {
        Self { root: PathBuf::from(root) }
    }
    pub fn bars(&self, t: &str) -> Result<Vec<Bar>> {
        store::read_bars(&self.root.join(format!("moex_discovery/data/dataset_final/01_stocks/candles_10m/{t}.parquet")))
    }
    /// Метки времени баров, выброшенных пайплайном в карантин.
    pub fn quarantine(&self, t: &str) -> HashSet<i64> {
        let p = self.root.join(format!("moex_discovery/data/quarantine/candles_10m/{t}.parquet"));
        store::read_bars(&p).map(|v| v.iter().map(|b| b.ts).collect()).unwrap_or_default()
    }
    pub fn features_path(&self) -> PathBuf {
        self.root.join("moex_volatility/data/features/stock_features.parquet")
    }
    pub fn master_path(&self) -> PathBuf {
        self.root.join("moex_discovery/data/final_v6/master_dataset.parquet")
    }
}

/// Применить к ISS-барам корректировки пайплайна (карантин).
pub fn apply_corrections(bars: &[Bar], quarantine: &HashSet<i64>) -> Vec<Bar> {
    bars.iter().filter(|b| !quarantine.contains(&b.ts)).copied().collect()
}

/// Хранимые дневные признаки одного тикера: день -> (rv, bv, jump, rsv_pos, rsv_neg, n_bars).
pub type Feat = BTreeMap<i64, [f64; 6]>;

pub fn load_features(path: &Path, ticker: &str) -> Result<Feat> {
    let mut out = Feat::new();
    for b in store::read_batches(path)? {
        let tk = store::col_str(&b, "ticker")?;
        let dt = store::col_str(&b, "date")?;
        let cols: Vec<Vec<f64>> =
            ["rv", "bv", "jump", "rsv_pos", "rsv_neg", "n_bars"].iter().map(|c| store::col_f64(&b, c)).collect::<Result<_>>()?;
        for i in 0..tk.len() {
            if tk[i] != ticker {
                continue;
            }
            if let Some(d) = crate::model::parse_day(&dt[i]) {
                out.insert(d, [cols[0][i], cols[1][i], cols[2][i], cols[3][i], cols[4][i], cols[5][i]]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrections_drop_quarantined() {
        let b = |ts| Bar { ts, open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: 1, value: 1.0 };
        let q: HashSet<i64> = [2].into_iter().collect();
        let out = apply_corrections(&[b(1), b(2), b(3)], &q);
        assert_eq!(out.iter().map(|x| x.ts).collect::<Vec<_>>(), vec![1, 3]);
    }
}
