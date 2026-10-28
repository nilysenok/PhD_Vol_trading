//! Число баров до 18:50 по тикер-дням: медиана, мода, наличие баров 09:50, 18:40, 18:50.
use crate::model::{day_str, minute_of_day, Bar};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct DayCount {
    pub ticker: String,
    pub day: i64,
    /// begin ∈ [10:00; 18:50)
    pub n: usize,
    pub has0950: bool,
    pub has1840: bool,
    pub has1850: bool,
}

pub fn count_day(t: &str, bars: &[Bar]) -> Vec<DayCount> {
    let mut m: BTreeMap<i64, DayCount> = BTreeMap::new();
    for b in bars {
        let x = minute_of_day(b.ts);
        let e = m.entry(b.day()).or_insert(DayCount { ticker: t.into(), day: b.day(), n: 0, has0950: false, has1840: false, has1850: false });
        e.n += (600..1130).contains(&x) as usize;
        e.has0950 |= x == 590;
        e.has1840 |= x == 1120;
        e.has1850 |= x == 1130;
    }
    m.into_values().filter(|c| c.n > 0).collect()
}

pub fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

pub fn mode(v: &[usize]) -> usize {
    let mut h: HashMap<usize, usize> = HashMap::new();
    for x in v {
        *h.entry(*x).or_default() += 1;
    }
    h.into_iter().max_by_key(|(k, c)| (*c, *k)).map_or(0, |x| x.0)
}

fn line(label: &str, c: &[&DayCount]) -> String {
    let ns: Vec<usize> = c.iter().map(|x| x.n).collect();
    let sh = |f: &dyn Fn(&DayCount) -> bool| c.iter().filter(|x| f(x)).count() as f64 / c.len().max(1) as f64;
    let mut f: Vec<f64> = ns.iter().map(|x| *x as f64).collect();
    format!(
        "{label},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
        c.len(), median(&mut f), mode(&ns),
        sh(&|x| x.n == 52), sh(&|x| x.n == 53), sh(&|x| x.n == 54),
        sh(&|x| x.has0950), sh(&|x| x.has1840), sh(&|x| x.has1850)
    )
}

pub fn write(out: &Path, all: &[DayCount]) -> Result<()> {
    let head = "группа,тикер-дней,медиана,мода,доля_52,доля_53,доля_54,доля_с_баром_0950,доля_с_баром_1840,доля_с_баром_1850\n";
    let (mut by_year, mut by_t): (BTreeMap<String, Vec<&DayCount>>, BTreeMap<String, Vec<&DayCount>>) = Default::default();
    for c in all {
        by_year.entry(day_str(c.day)[..4].to_string()).or_default().push(c);
        by_t.entry(c.ticker.clone()).or_default().push(c);
    }
    let y: String = by_year.iter().map(|(k, v)| line(k, v)).collect::<String>() + &line("все", &all.iter().collect::<Vec<_>>());
    let t: String = by_t.iter().map(|(k, v)| line(k, v)).collect();
    std::fs::write(out.join("report/bars_count_by_year.csv"), head.to_string() + &y)?;
    std::fs::write(out.join("report/bars_count_by_ticker.csv"), head.to_string() + &t)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_mode() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(mode(&[53, 54, 53, 52, 53]), 53);
    }

    #[test]
    fn counts_window() {
        let mk = |h: i64, m: i64| Bar { ts: (h * 60 + m) * 60_000, open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: 1, value: 1.0 };
        let c = count_day("X", &[mk(9, 50), mk(10, 0), mk(18, 40), mk(18, 50), mk(19, 0)]);
        assert_eq!((c[0].n, c[0].has0950, c[0].has1840, c[0].has1850), (2, true, true, true));
    }
}
