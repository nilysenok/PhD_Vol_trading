//! Дневная история индексов MOEX: /iss/history/.../index/securities/{INDEX}.json, постранично.
use crate::fetch::Http;
use crate::parse::{parse_index, IndexRow};
use crate::store::write_daily;
use anyhow::Result;
use std::path::Path;

fn url(index: &str, from: &str, till: &str, start: u64) -> String {
    format!("/iss/history/engines/stock/markets/index/securities/{index}.json?from={from}&till={till}&start={start}")
}

/// Оставляет одну строку на дату (последнюю); возвращает (строки, число отброшенных дублей).
pub fn dedup_days(mut rows: Vec<IndexRow>) -> (Vec<IndexRow>, usize) {
    rows.sort_by_key(|r| r.day);
    let n = rows.len();
    let mut out: Vec<IndexRow> = Vec::with_capacity(n);
    for r in rows {
        match out.last_mut() {
            Some(l) if l.day == r.day => *l = r,
            _ => out.push(r),
        }
    }
    let d = n - out.len();
    (out, d)
}

pub async fn fetch_one(http: &Http, index: &str, from: &str, till: &str) -> Result<Vec<IndexRow>> {
    let (mut all, mut start) = (Vec::new(), 0u64);
    loop {
        let (rows, cur) = parse_index(&http.get_text(&url(index, from, till, start)).await?)?;
        let step = cur.map(|c| c.pagesize).unwrap_or(rows.len() as u64).max(rows.len() as u64);
        let n = rows.len();
        all.extend(rows);
        start += step;
        let total = cur.map(|c| c.total).unwrap_or(0);
        if n == 0 || (cur.is_some() && start >= total) {
            break;
        }
        tokio::time::sleep(crate::download::PAUSE).await;
    }
    Ok(all)
}

pub async fn run(indices: &[String], out: &Path, from: &str, till: &str) -> Result<()> {
    let http = Http::new()?;
    for ix in indices {
        let (rows, dups) = dedup_days(fetch_one(&http, ix, from, till).await?);
        let days: Vec<i64> = rows.iter().map(|r| r.day).collect();
        let c = |g: fn(&IndexRow) -> Option<f64>| rows.iter().map(g).collect::<Vec<_>>();
        let cols = [
            ("close", c(|r| r.close)),
            ("open", c(|r| r.open)),
            ("high", c(|r| r.high)),
            ("low", c(|r| r.low)),
            ("value", c(|r| r.value)),
            ("volume", c(|r| r.volume)),
        ];
        write_daily(&out.join("index_d").join(format!("{ix}.parquet")), &days, &cols)?;
        eprintln!("{ix}: {} дней (дублей по дате отброшено {dups})", rows.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_keeps_last() {
        let r = |day, close| IndexRow { day, close: Some(close), ..Default::default() };
        let (v, d) = dedup_days(vec![r(2, 1.0), r(1, 5.0), r(2, 2.0)]);
        assert_eq!(d, 1);
        assert_eq!(v.iter().map(|x| (x.day, x.close)).collect::<Vec<_>>(), vec![(1, Some(5.0)), (2, Some(2.0))]);
    }
}
