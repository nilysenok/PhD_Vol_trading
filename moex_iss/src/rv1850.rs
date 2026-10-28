//! RV со срезом 18:50 МСК из биржевых баров (bars_10m_full) и сверка со старыми признаками.
//! Формула — rv::daily (intraday_features.py:105-173); меняется только окно баров:
//!  основное окно begin ∈ [09:50; 18:50) — как в старых признаках до 22.06.2020 (они включают
//!  бар предоткрытия 09:50), отличие лишь в том, что вечерние/утренние бары отброшены;
//!  строгое окно begin ∈ [10:00; 18:50) — отдельный файл.
use crate::cutstats::{count_day, DayCount};
use crate::model::{day_str, minute_of_day, parse_day, Bar};
use crate::rv::{daily, DayRv};
use crate::rvcmp::{rel, TOL};
use crate::stored::{load_features, Feat, Stored};
use crate::store;
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

pub const SPLIT: &str = "2020-06-22";

#[derive(Clone, Copy)]
pub struct Win {
    pub from_min: i64,
    pub to_min: i64,
}
pub const W0950: Win = Win { from_min: 9 * 60 + 50, to_min: 18 * 60 + 50 };
pub const W1000: Win = Win { from_min: 10 * 60, to_min: 18 * 60 + 50 };

pub fn cut(bars: &[Bar], w: Win, drop: Option<&HashSet<i64>>) -> Vec<Bar> {
    bars.iter()
        .filter(|b| (w.from_min..w.to_min).contains(&minute_of_day(b.ts)) && drop.map_or(true, |q| !q.contains(&b.ts)))
        .copied()
        .collect()
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct VarStat {
    pub variant: String,
    pub ticker: String,
    pub pre_days: usize,
    pub pre_ok: usize,
    pub pre_off_quar: usize,
    pub pre_off_other: usize,
    pub post_days: usize,
    pub post_ok: usize,
    /// после SPLIT: (день, совпал, rv_новый/rv_старый, bv_новый/bv_старый, у старого баров больше, т.е. есть вечер)
    pub post: Vec<(i64, bool, f64, f64, bool)>,
}

fn same(g: &DayRv, r: &[f64; 6]) -> bool {
    rel(g.rv, r[0]) <= TOL && rel(g.bv, r[1]) <= TOL && rel(g.jump, r[2]) <= TOL
}

pub fn compare(variant: &str, ticker: &str, got: &BTreeMap<i64, DayRv>, rf: &Feat, quar_days: &HashSet<i64>, split: i64) -> (VarStat, Vec<String>) {
    let mut s = VarStat { variant: variant.into(), ticker: ticker.into(), ..Default::default() };
    let mut lines = Vec::new();
    for (d, r) in rf {
        let g = got.get(d);
        let ok = g.map_or(false, |g| same(g, r));
        if *d < split {
            s.pre_days += 1;
            if ok {
                s.pre_ok += 1;
            } else {
                let q = quar_days.contains(d);
                if q { s.pre_off_quar += 1 } else { s.pre_off_other += 1 }
                let (a, b) = g.map_or((f64::NAN, f64::NAN), |g| (g.rv, g.n as f64));
                lines.push(format!("{variant},{ticker},{},{},{:.6e},{:.6e},{b},{}", day_str(*d), if q { "карантин" } else { "прочее" }, a, r[0], r[5]));
            }
        } else {
            s.post_days += 1;
            s.post_ok += ok as usize;
            if let Some(g) = g {
                let q = |x: f64, y: f64| if x > 0.0 && y > 0.0 { x / y } else { 0.0 };
                s.post.push((*d, ok, q(g.rv, r[0]), q(g.bv, r[1]), r[5] > g.n as f64));
            }
        }
    }
    (s, lines)
}

type Rows = Vec<(String, i64, DayRv, i64)>;
struct Job {
    a: Rows,
    b: Rows,
    c: Rows,
    stats: Vec<VarStat>,
    off: Vec<String>,
    counts: Vec<DayCount>,
}

fn rows(t: &str, m: &BTreeMap<i64, DayRv>, nq: &BTreeMap<i64, i64>) -> Rows {
    m.iter().map(|(d, v)| (t.to_string(), *d, *v, nq.get(d).copied().unwrap_or(0))).collect()
}

fn job(out: &Path, st: &Stored, t: &str) -> Result<Job> {
    let full = store::read_bars(&out.join("bars_10m_full").join(format!("{t}.parquet")))?;
    let (q, feat) = (st.quarantine(t), load_features(&st.features_path(), t)?);
    let split = parse_day(SPLIT).unwrap_or(0);
    let first = feat.keys().next().copied().unwrap_or(0);
    let full: Vec<Bar> = full.into_iter().filter(|b| b.day() >= first - 5).collect();
    let (a_bars, b_bars) = (cut(&full, W0950, None), cut(&full, W1000, None));
    let mut nq: BTreeMap<i64, i64> = BTreeMap::new();
    for b in a_bars.iter().filter(|b| q.contains(&b.ts)) {
        *nq.entry(b.day()).or_default() += 1;
    }
    let qdays: HashSet<i64> = nq.keys().copied().collect();
    let (a, b) = (daily(&a_bars), daily(&b_bars));
    let aq = daily(&cut(&full, W0950, Some(&q)));
    let mut stats = Vec::new();
    let mut off = Vec::new();
    for (name, m) in [("0950_raw", &a), ("0950_без_карантина", &aq), ("1000_raw", &b)] {
        let (s, l) = compare(name, t, m, &feat, &qdays, split);
        stats.push(s);
        off.extend(l);
    }
    Ok(Job { a: rows(t, &a, &nq), b: rows(t, &b, &BTreeMap::new()), c: rows(t, &aq, &BTreeMap::new()), stats, off, counts: count_day(t, &full) })
}

pub fn run(out: &Path, st: &Stored, tickers: &[String]) -> Result<()> {
    let jobs: Vec<Result<Job>> = tickers.par_iter().map(|t| job(out, st, t)).collect();
    let mut c = Vec::new();
    let (mut a, mut b, mut stats, mut off, mut counts) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for j in jobs {
        let j = j?;
        a.extend(j.a);
        b.extend(j.b);
        c.extend(j.c);
        stats.extend(j.stats);
        off.extend(j.off);
        counts.extend(j.counts);
    }
    store::write_rv(&out.join("rv_1850.parquet"), &a)?;
    store::write_rv(&out.join("rv_1850_from1000.parquet"), &b)?;
    store::write_rv(&out.join("rv_1850_noquar.parquet"), &c)?;
    std::fs::create_dir_all(out.join("report"))?;
    std::fs::write(out.join("report/rv1850_summary.json"), serde_json::to_vec(&stats)?)?;
    let head = "variant,ticker,day,причина,rv_новый,rv_старый,n_новый,n_старый\n";
    std::fs::write(out.join("report/rv1850_pre_off.csv"), head.to_string() + &off.join("\n"))?;
    crate::cutstats::write(out, &counts)?;
    eprintln!("rv_1850: {} строк, from1000: {} строк, несовпадений до {SPLIT}: {}", a.len(), b.len(), off.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_cuts_evening_and_keeps_1840() {
        let mk = |h: i64, m: i64| Bar { ts: (h * 60 + m) * 60_000, open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: 1, value: 1.0 };
        let v = [mk(6, 50), mk(9, 50), mk(10, 0), mk(18, 40), mk(18, 50), mk(19, 0)];
        assert_eq!(cut(&v, W0950, None).len(), 3);
        assert_eq!(cut(&v, W1000, None).len(), 2);
        let q: HashSet<i64> = [mk(10, 0).ts].into_iter().collect();
        assert_eq!(cut(&v, W1000, Some(&q)).len(), 1);
    }
}
