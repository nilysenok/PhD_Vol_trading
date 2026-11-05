//! Ж-201: 10-мин свечи индекса IMOEX (/iss/engines/stock/markets/index/securities/IMOEX).
//! Поля, которых ISS не дал, остаются null (не нули). Докачка и повторы — как у `bars`.
use crate::fetch::{Http, BASE};
use crate::model::{day_of, parse_day, parse_ts, ts_str};
use anyhow::{anyhow, Context, Result};
use arrow::array::{ArrayRef, Float64Array, Int64Array, TimestampMillisecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

pub const SECID: &str = "IMOEX";

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IBar {
    pub ts: i64,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub volume: Option<f64>,
    pub value: Option<f64>,
}

#[derive(Serialize, Deserialize, Default)]
struct Progress {
    next_start: u64,
    done: bool,
}

pub fn index_base(secid: &str) -> String {
    format!("/iss/engines/stock/markets/index/securities/{secid}")
}

pub fn url(secid: &str, from: &str, till: &str, start: u64) -> String {
    crate::download::candles_url(&index_base(secid), from, till, start)
}

/// Страница свечей индекса -> бары; null остаётся None.
pub fn parse_page(text: &str) -> Result<Vec<IBar>> {
    let v: Value = serde_json::from_str(text).context("не JSON")?;
    let t = v.get("candles").ok_or_else(|| anyhow!("нет блока candles"))?;
    let cols = t.get("columns").and_then(Value::as_array).ok_or_else(|| anyhow!("нет columns"))?;
    let data = t.get("data").and_then(Value::as_array).ok_or_else(|| anyhow!("нет data"))?;
    let ix = |k: &str| cols.iter().position(|c| c.as_str() == Some(k)).ok_or_else(|| anyhow!("нет колонки {k}"));
    let (ib, io, ih, il, ic, iv, ivl) = (ix("begin")?, ix("open")?, ix("high")?, ix("low")?, ix("close")?, ix("volume")?, ix("value")?);
    let mut out = Vec::with_capacity(data.len());
    for r in data {
        let ts = r.get(ib).and_then(Value::as_str).and_then(parse_ts).ok_or_else(|| anyhow!("begin"))?;
        let f = |i: usize| r.get(i).and_then(Value::as_f64);
        out.push(IBar { ts, open: f(io), high: f(ih), low: f(il), close: f(ic), volume: f(iv), value: f(ivl) });
    }
    Ok(out)
}

/// Оставляет бары с днём в [from; till] включительно (граница периода, мутация № 36).
pub fn clip(bars: Vec<IBar>, from: &str, till: &str) -> Result<Vec<IBar>> {
    let (a, b) = (parse_day(from).ok_or_else(|| anyhow!("from"))?, parse_day(till).ok_or_else(|| anyhow!("till"))?);
    Ok(bars.into_iter().filter(|x| (a..=b).contains(&day_of(x.ts))).collect())
}

fn fmt(x: Option<f64>) -> String {
    x.map(|v| format!("{v}")).unwrap_or_default()
}

fn line(b: &IBar) -> String {
    format!("{},{},{},{},{},{},{}\n", b.ts, fmt(b.open), fmt(b.high), fmt(b.low), fmt(b.close), fmt(b.volume), fmt(b.value))
}

/// Читает сырой csv: пустое поле = None; оборванная последняя строка (без \n) и дубли отбрасываются.
pub fn load_raw(path: &Path) -> Vec<IBar> {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    let s = match s.rfind('\n') {
        Some(i) => &s[..=i],
        None => "",
    };
    let mut v: Vec<IBar> = s
        .lines()
        .filter_map(|l| {
            let p: Vec<&str> = l.split(',').collect();
            if p.len() != 7 {
                return None;
            }
            let f = |i: usize| p[i].parse::<f64>().ok();
            Some(IBar { ts: p[0].parse().ok()?, open: f(1), high: f(2), low: f(3), close: f(4), volume: f(5), value: f(6) })
        })
        .collect();
    v.sort_by_key(|b| b.ts);
    v.dedup_by_key(|b| b.ts);
    v
}

pub fn write_parquet(path: &Path, bars: &[IBar]) -> Result<()> {
    let f = |g: fn(&IBar) -> Option<f64>| -> ArrayRef { Arc::new(Float64Array::from(bars.iter().map(g).collect::<Vec<_>>())) };
    let fl = |n: &str| Field::new(n, DataType::Float64, true);
    let schema = Schema::new(vec![
        Field::new("begin", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        fl("open"),
        fl("high"),
        fl("low"),
        fl("close"),
        Field::new("volume", DataType::Int64, true),
        fl("value"),
    ]);
    let vol: Vec<Option<i64>> = bars.iter().map(|b| b.volume.map(|v| v.round() as i64)).collect();
    let cols: Vec<ArrayRef> = vec![
        Arc::new(TimestampMillisecondArray::from_iter_values(bars.iter().map(|b| b.ts))),
        f(|b| b.open),
        f(|b| b.high),
        f(|b| b.low),
        f(|b| b.close),
        Arc::new(Int64Array::from(vol)),
        f(|b| b.value),
    ];
    crate::store::write(path, schema, cols)
}

fn sha256(path: &Path) -> Result<String> {
    for (prog, args) in [("shasum", vec!["-a", "256"]), ("sha256sum", vec![])] {
        if let Ok(o) = std::process::Command::new(prog).args(&args).arg(path).output() {
            if o.status.success() {
                return String::from_utf8_lossy(&o.stdout).split_whitespace().next().map(str::to_string).ok_or_else(|| anyhow!("sha"));
            }
        }
    }
    Err(anyhow!("нет shasum/sha256sum"))
}

fn manifest(dir: &Path, bars: &[IBar], from: &str, till: &str) -> Result<()> {
    let file = dir.join(format!("{SECID}.parquet"));
    let (first, last) = (bars.first().map(|b| ts_str(b.ts)), bars.last().map(|b| ts_str(b.ts)));
    let nulls = |g: fn(&IBar) -> Option<f64>| bars.iter().filter(|b| g(b).is_none()).count();
    let m = serde_json::json!({
        "file": format!("{SECID}.parquet"), "rows": bars.len(), "first": first, "last": last,
        "sha256": sha256(&file)?, "created_utc": chrono::Utc::now().to_rfc3339(),
        "command": format!("moex_iss index-bars --from {from} --till {till}"),
        "url_template": format!("{BASE}{}", url(SECID, "{from}", "{till}", 0).replace("start=0", "start={start}")),
        "null_counts": { "open": nulls(|b| b.open), "close": nulls(|b| b.close), "volume": nulls(|b| b.volume), "value": nulls(|b| b.value) },
    });
    std::fs::write(dir.join("MANIFEST.json"), serde_json::to_vec_pretty(&m)?)?;
    Ok(())
}

pub async fn run(out: &Path, from: &str, till: &str) -> Result<()> {
    let dir = out.join("index_10m");
    let (raw, prog) = (dir.join("raw").join(format!("{SECID}.csv")), dir.join("progress").join(format!("{SECID}.json")));
    std::fs::create_dir_all(raw.parent().context("raw")?)?;
    std::fs::create_dir_all(prog.parent().context("progress")?)?;
    let mut p: Progress = std::fs::read(&prog).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let http = Http::new()?;
    while !p.done {
        let page = parse_page(&http.get_text(&url(SECID, from, till, p.next_start)).await?).with_context(|| format!("start={}", p.next_start))?;
        if page.is_empty() {
            p.done = true;
        } else {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&raw)?;
            f.write_all(page.iter().map(line).collect::<String>().as_bytes())?;
            f.sync_data()?;
            p.next_start += page.len() as u64;
        }
        std::fs::write(&prog, serde_json::to_vec(&p)?)?;
        tokio::time::sleep(crate::download::PAUSE).await;
    }
    let all = clip(load_raw(&raw), from, till)?;
    write_parquet(&dir.join(format!("{SECID}.parquet")), &all)?;
    manifest(&dir, &all, from, till)?;
    eprintln!("{SECID}: готово, баров {}", all.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = r#"{"candles":{"columns":["open","close","high","low","value","volume","begin","end"],
      "data":[[100.5,101.0,102.0,99.0,5000000.0,null,"2014-06-02 10:00:00","2014-06-02 10:09:59"],
              [null,null,null,null,null,null,"2014-05-30 10:00:00","2014-05-30 10:09:59"],
              [101.0,103.0,103.5,100.0,7.5,12,"2026-02-28 10:00:00","2026-02-28 10:09:59"]]}}"#;

    #[test]
    fn url_is_index_not_shares() {
        let u = url("IMOEX", "2014-06-01", "2026-02-27", 500);
        assert_eq!(u, "/iss/engines/stock/markets/index/securities/IMOEX/candles.json?interval=10&from=2014-06-01&till=2026-02-27&start=500");
        assert!(!u.contains("shares") && !u.contains("TQBR"));
    }

    #[test]
    fn parse_keeps_null_not_zero() -> Result<()> {
        let b = parse_page(PAGE)?;
        assert_eq!((b[0].open, b[0].volume, b[0].value), (Some(100.5), None, Some(5_000_000.0)));
        assert_eq!(b[1].close, None);
        assert!(parse_page("{}").is_err());
        Ok(())
    }

    #[test]
    fn clip_drops_outside_period() -> Result<()> {
        let c = clip(parse_page(PAGE)?, "2014-06-01", "2026-02-27")?;
        assert_eq!(c.len(), 1);
        assert_eq!(ts_str(c[0].ts), "2014-06-02 10:00");
        Ok(())
    }

    #[test]
    fn raw_roundtrip_nulls_and_partial_tail() -> Result<()> {
        let p = std::env::temp_dir().join("moex_iss_idx_raw_test.csv");
        let mut b = parse_page(PAGE)?;
        let text: String = b.iter().map(line).collect();
        std::fs::write(&p, format!("{text}{text}9999,1,2,3,4,5,6"))?;
        b.sort_by_key(|x| x.ts);
        assert_eq!(load_raw(&p), b);
        Ok(())
    }

    #[test]
    fn parquet_has_nulls() -> Result<()> {
        let p = std::env::temp_dir().join("moex_iss_idx_pq_test").join("x.parquet");
        write_parquet(&p, &parse_page(PAGE)?)?;
        let bt = crate::store::read_batches(&p)?;
        assert_eq!(bt[0].num_rows(), 3);
        assert!(crate::store::col_f64(&bt[0], "volume")?[0].is_nan());
        Ok(())
    }
}
