//! Г2: сверка ISS-баров с хранимыми бар к бару (основная сессия обеих сторон).
use crate::model::{day_str, ts_str, Bar};
use crate::session::{filter_main, EXPECTED_BARS};
use crate::stored::Stored;
use crate::store;
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

pub const PRICE_TOL: f64 = 1e-6;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct TickerVerify {
    pub ticker: String,
    pub from: String,
    pub to: String,
    pub days: usize,
    pub iss_bars: usize,
    pub stored_bars: usize,
    pub matched: usize,
    pub mismatched: usize,
    pub only_iss_quarantine: usize,
    pub only_iss_extra: usize,
    pub only_stored: usize,
    pub share_matched: f64,
    pub days_incomplete_iss: usize,
    pub days_incomplete_stored: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Diff {
    pub ticker: String,
    pub ts: String,
    pub score: f64,
    pub what: String,
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-12)
}

/// Сравнение двух рядов основной сессии; `lo..=hi` — диапазон дней хранимых данных.
pub fn compare(
    ticker: &str,
    iss: &[Bar],
    stored: &[Bar],
    quar: &HashSet<i64>,
    (lo, hi): (i64, i64),
) -> (TickerVerify, Vec<Diff>, Vec<(i64, usize, usize)>) {
    let iss: Vec<&Bar> = iss.iter().filter(|b| (lo..=hi).contains(&b.day())).collect();
    let sm: HashMap<i64, &Bar> = stored.iter().map(|b| (b.ts, b)).collect();
    let mut v = TickerVerify { ticker: ticker.into(), from: day_str(lo), to: day_str(hi), ..Default::default() };
    let (mut diffs, mut seen) = (Vec::new(), HashSet::new());
    let mut per_day: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
    for b in &iss {
        per_day.entry(b.day()).or_default().0 += 1;
        seen.insert(b.ts);
        match sm.get(&b.ts) {
            None if quar.contains(&b.ts) => v.only_iss_quarantine += 1,
            None => v.only_iss_extra += 1,
            Some(s) => {
                let pr = [("open", b.open, s.open), ("high", b.high, s.high), ("low", b.low, s.low), ("close", b.close, s.close)];
                let bad: Vec<String> =
                    pr.iter().filter(|p| (p.1 - p.2).abs() > PRICE_TOL).map(|p| format!("{} {} vs {}", p.0, p.1, p.2)).collect();
                if bad.is_empty() && b.volume == s.volume {
                    v.matched += 1;
                    continue;
                }
                v.mismatched += 1;
                let mut what = bad;
                if b.volume != s.volume {
                    what.push(format!("volume {} vs {}", b.volume, s.volume));
                }
                let score = pr.iter().map(|p| rel(p.1, p.2)).fold(rel(b.volume as f64, s.volume as f64), f64::max);
                diffs.push(Diff { ticker: ticker.into(), ts: ts_str(b.ts), score, what: what.join("; ") });
            }
        }
    }
    for s in stored.iter().filter(|s| !seen.contains(&s.ts)) {
        per_day.entry(s.day()).or_default();
        v.only_stored += 1;
        diffs.push(Diff { ticker: ticker.into(), ts: ts_str(s.ts), score: f64::INFINITY, what: "нет в ISS".into() });
    }
    for s in stored {
        per_day.entry(s.day()).or_default().1 += 1;
    }
    v.days = per_day.len();
    v.iss_bars = iss.len();
    v.stored_bars = stored.len();
    let denom = v.matched + v.mismatched + v.only_iss_extra + v.only_stored;
    v.share_matched = if denom == 0 { 0.0 } else { v.matched as f64 / denom as f64 };
    let inc: Vec<(i64, usize, usize)> =
        per_day.iter().filter(|(_, c)| c.0 != EXPECTED_BARS || c.1 != EXPECTED_BARS).map(|(d, c)| (*d, c.0, c.1)).collect();
    v.days_incomplete_iss = inc.iter().filter(|x| x.1 != EXPECTED_BARS).count();
    v.days_incomplete_stored = inc.iter().filter(|x| x.2 != EXPECTED_BARS).count();
    (v, diffs, inc)
}

pub fn run(out: &Path, st: &Stored, tickers: &[String]) -> Result<()> {
    let res: Vec<Result<_>> = tickers
        .par_iter()
        .map(|t| {
            let iss = filter_main(&store::read_bars(&out.join("bars_10m").join(format!("{t}.parquet")))?);
            let all = st.bars(t)?;
            let range = match (all.first(), all.last()) {
                (Some(a), Some(b)) => (a.day(), b.day()),
                _ => (0, -1),
            };
            let (v, d, inc) = compare(t, &iss, &filter_main(&all), &st.quarantine(t), range);
            Ok((v, d, inc.into_iter().map(|x| (t.clone(), x)).collect::<Vec<_>>()))
        })
        .collect();
    let (mut sum, mut diffs, mut inc) = (Vec::new(), Vec::new(), Vec::new());
    for r in res {
        let (v, d, i) = r?;
        sum.push(v);
        diffs.extend(d);
        inc.extend(i);
    }
    diffs.sort_by(|a, b| b.score.total_cmp(&a.score));
    std::fs::create_dir_all(out.join("report"))?;
    std::fs::write(out.join("report/verify_summary.json"), serde_json::to_vec_pretty(&sum)?)?;
    std::fs::write(out.join("report/verify_diffs_top.json"), serde_json::to_vec_pretty(&diffs.iter().take(1000).collect::<Vec<_>>())?)?;
    let mut csv = String::from("ticker,day,iss_bars,stored_bars\n");
    for (t, (d, a, b)) in &inc {
        csv += &format!("{t},{},{a},{b}\n", day_str(*d));
    }
    std::fs::write(out.join("report/incomplete_days.csv"), csv)?;
    for v in &sum {
        eprintln!("{}: совпало {:.4}% ({} из {}), неполных дней ISS {}", v.ticker, v.share_matched * 100.0, v.matched, v.iss_bars, v.days_incomplete_iss);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(ts: i64, c: f64, vol: i64) -> Bar {
        Bar { ts, open: c, high: c, low: c, close: c, volume: vol, value: 0.0 }
    }

    #[test]
    fn classifies_bars() {
        let iss = vec![b(1, 1.0, 5), b(2, 2.0, 5), b(3, 3.0, 5), b(4, 4.0, 5)];
        let st = vec![b(1, 1.0, 5), b(2, 2.5, 5), b(4, 4.0, 6), b(9, 9.0, 1)];
        let q: HashSet<i64> = [3].into_iter().collect();
        let (v, d, _) = compare("X", &iss, &st, &q, (0, 0));
        assert_eq!((v.matched, v.mismatched, v.only_iss_quarantine, v.only_iss_extra, v.only_stored), (1, 2, 1, 0, 1));
        assert_eq!(d.len(), 3);
        assert!((v.share_matched - 0.25).abs() < 1e-12);
    }
}
