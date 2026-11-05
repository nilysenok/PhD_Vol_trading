//! Ж-201: сводка по дням для сверки календаря (только чтение parquet): число баров основной
//! сессии [10:00; 18:40) и close последнего её бара; дневной close индекса из index_d.
use crate::model::{day_of, day_str};
use crate::{session, store};
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;

/// День -> (число баров основной сессии, close последнего бара основной сессии, число всех баров).
/// Четвёртое число — close последнего бара с begin < 19:00 (включая аукцион закрытия 18:40/18:50).
pub fn per_day(bars: &[(i64, f64)]) -> BTreeMap<i64, (usize, f64, usize, f64)> {
    let mut m: BTreeMap<i64, (usize, f64, usize, f64, i64, i64)> = BTreeMap::new();
    for &(ts, close) in bars {
        let e = m.entry(day_of(ts)).or_insert((0, f64::NAN, 0, f64::NAN, i64::MIN, i64::MIN));
        e.2 += 1;
        if session::in_main(ts) {
            e.0 += 1;
            if ts > e.4 {
                e.4 = ts;
                e.1 = close;
            }
        }
        if crate::model::minute_of_day(ts) < 19 * 60 && ts > e.5 {
            e.5 = ts;
            e.3 = close;
        }
    }
    m.into_iter().map(|(d, (n, c, a, c2, _, _))| (d, (n, c, a, c2))).collect()
}

fn read_close(path: &Path) -> Result<Vec<(i64, f64)>> {
    let mut out = Vec::new();
    for b in store::read_batches(path)? {
        let (ts, c) = (store::col_ts(&b, "begin")?, store::col_f64(&b, "close")?);
        out.extend(ts.into_iter().zip(c));
    }
    Ok(out)
}

fn write_days(path: &Path, bars: &[(i64, f64)]) -> Result<()> {
    let mut s = String::from("date,n_main,close_last_main,n_all,close_last_1850\n");
    let f = |c: f64| if c.is_nan() { String::new() } else { c.to_string() };
    for (d, (n, c, a, c2)) in per_day(bars) {
        s += &format!("{},{n},{},{a},{}\n", day_str(d), f(c), f(c2));
    }
    std::fs::write(path, s)?;
    Ok(())
}

pub fn run(out: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    write_days(&dest.join("days_imoex.csv"), &read_close(&out.join("index_10m/IMOEX.parquet"))?)?;
    write_days(&dest.join("days_sber.csv"), &read_close(&out.join("bars_10m_full/SBER.parquet"))?)?;
    let mut s = String::from("date,close\n");
    for b in store::read_batches(&out.join("index_d/IMOEX.parquet"))? {
        let (d, c) = (store::col_str(&b, "date")?, store::col_f64(&b, "close")?);
        for (d, c) in d.iter().zip(c) {
            s += &format!("{d},{c}\n");
        }
    }
    std::fs::write(dest.join("index_d_imoex.csv"), s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::parse_ts;

    #[test]
    fn per_day_counts_main_only_and_last_main_close() {
        let t = |s: &str| parse_ts(s).unwrap_or(0);
        let bars = [
            (t("2020-01-09 09:50:00"), 1.0),
            (t("2020-01-09 10:00:00"), 2.0),
            (t("2020-01-09 18:30:00"), 3.0),
            (t("2020-01-09 18:40:00"), 4.0),
            (t("2020-01-09 19:05:00"), 5.0),
        ];
        let m = per_day(&bars);
        assert_eq!(m.values().next(), Some(&(2, 3.0, 5, 4.0)));
    }
}
