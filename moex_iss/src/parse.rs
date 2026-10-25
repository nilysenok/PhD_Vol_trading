//! Разбор ответов ISS (JSON-таблицы: columns + data).
use crate::model::{parse_day, parse_ts, Bar};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::collections::HashMap;

/// Таблица ISS: имя колонки -> индекс и строки.
fn table<'a>(v: &'a Value, name: &str) -> Result<(HashMap<&'a str, usize>, &'a Vec<Value>)> {
    let t = v.get(name).ok_or_else(|| anyhow!("нет блока {name}"))?;
    let cols = t.get("columns").and_then(Value::as_array).ok_or_else(|| anyhow!("нет columns"))?;
    let idx = cols.iter().enumerate().filter_map(|(i, c)| c.as_str().map(|s| (s, i))).collect();
    let data = t.get("data").and_then(Value::as_array).ok_or_else(|| anyhow!("нет data"))?;
    Ok((idx, data))
}

fn num(row: &Value, i: usize) -> Option<f64> {
    row.get(i).and_then(Value::as_f64)
}

/// Страница свечей -> бары (как есть, без фильтра сессии).
pub fn parse_candles(text: &str) -> Result<Vec<Bar>> {
    let v: Value = serde_json::from_str(text).context("не JSON")?;
    let (c, data) = table(&v, "candles")?;
    let g = |k: &str| c.get(k).copied().ok_or_else(|| anyhow!("нет колонки {k}"));
    let (io, ih, il, ic, iv, ivl, ib) =
        (g("open")?, g("high")?, g("low")?, g("close")?, g("volume")?, g("value")?, g("begin")?);
    let mut out = Vec::with_capacity(data.len());
    for row in data {
        let ts = row.get(ib).and_then(Value::as_str).and_then(parse_ts).ok_or_else(|| anyhow!("begin"))?;
        let f = |i| num(row, i).ok_or_else(|| anyhow!("пустое поле в строке {ts}"));
        out.push(Bar {
            ts,
            open: f(io)?,
            high: f(ih)?,
            low: f(il)?,
            close: f(ic)?,
            // Д24: у валютного рынка (CETS) volume/value в свечах бывают null — тогда 0 (цены обязательны, как раньше)
            volume: num(row, iv).unwrap_or(0.0).round() as i64,
            value: num(row, ivl).unwrap_or(0.0),
        });
    }
    Ok(out)
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct IndexRow {
    pub day: i64,
    pub close: Option<f64>,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub value: Option<f64>,
    pub volume: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cursor {
    pub index: u64,
    pub total: u64,
    pub pagesize: u64,
}

/// Страница истории индекса -> строки и курсор.
pub fn parse_index(text: &str) -> Result<(Vec<IndexRow>, Option<Cursor>)> {
    let v: Value = serde_json::from_str(text).context("не JSON")?;
    let (c, data) = table(&v, "history")?;
    let g = |k: &str| c.get(k).copied();
    let id = g("TRADEDATE").ok_or_else(|| anyhow!("нет TRADEDATE"))?;
    let mut rows = Vec::with_capacity(data.len());
    for r in data {
        let day = r.get(id).and_then(Value::as_str).and_then(parse_day).ok_or_else(|| anyhow!("TRADEDATE"))?;
        let f = |k: &str| g(k).and_then(|i| num(r, i));
        rows.push(IndexRow {
            day,
            close: f("CLOSE"),
            open: f("OPEN"),
            high: f("HIGH"),
            low: f("LOW"),
            value: f("VALUE"),
            volume: f("VOLUME"),
        });
    }
    let cursor = table(&v, "history.cursor").ok().and_then(|(c, d)| {
        let r = d.first()?;
        let u = |k: &str| r.get(*c.get(k)?).and_then(Value::as_u64);
        Some(Cursor { index: u("INDEX")?, total: u("TOTAL")?, pagesize: u("PAGESIZE")? })
    });
    Ok((rows, cursor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candles_fixture() -> Result<()> {
        let bars = parse_candles(include_str!("../tests/fixtures/candles_page.json"))?;
        assert!(bars.len() > 40);
        let b = bars[0];
        assert_eq!(b.ts, parse_ts("2014-06-09 10:00:00").unwrap_or(-1));
        assert_eq!((b.open, b.close, b.high, b.low), (89.21, 89.22, 89.4, 89.08));
        assert_eq!(b.volume, 3_673_650);
        assert!(bars.windows(2).all(|w| w[1].ts > w[0].ts));
        Ok(())
    }

    #[test]
    fn index_fixture() -> Result<()> {
        let (rows, cur) = parse_index(include_str!("../tests/fixtures/index_page.json"))?;
        assert_eq!(rows.len(), 100);
        assert_eq!(rows[0].close, Some(7374.93));
        assert_eq!(rows[0].day, parse_day("2022-01-03").unwrap_or(-1));
        assert_eq!(cur, Some(Cursor { index: 0, total: 1035, pagesize: 100 }));
        Ok(())
    }

    #[test]
    fn broken_json_is_error() {
        assert!(parse_candles("{}").is_err());
        assert!(parse_index("не json").is_err());
    }
}
