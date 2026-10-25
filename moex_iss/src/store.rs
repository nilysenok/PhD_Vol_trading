//! Чтение и запись parquet (arrow). Время баров — Timestamp(мс) без пояса, МСК.
use crate::model::Bar;
use anyhow::{anyhow, Result};
use arrow::array::{Array, ArrayRef, Date32Array, Float64Array, Int64Array, RecordBatch, StringArray, TimestampMillisecondArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const TS: DataType = DataType::Timestamp(TimeUnit::Millisecond, None);

pub fn read_batches(path: &Path) -> Result<Vec<RecordBatch>> {
    let f = File::open(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let r = ParquetRecordBatchReaderBuilder::try_new(f)?.build()?;
    Ok(r.collect::<std::result::Result<Vec<_>, _>>()?)
}

fn col(b: &RecordBatch, name: &str, to: &DataType) -> Result<ArrayRef> {
    let a = b.column_by_name(name).ok_or_else(|| anyhow!("нет колонки {name}"))?;
    Ok(cast(a, to)?)
}

pub fn col_f64(b: &RecordBatch, name: &str) -> Result<Vec<f64>> {
    let a = col(b, name, &DataType::Float64)?;
    let a = a.as_any().downcast_ref::<Float64Array>().ok_or_else(|| anyhow!("{name}: не f64"))?;
    Ok((0..a.len()).map(|i| if a.is_null(i) { f64::NAN } else { a.value(i) }).collect())
}

pub fn col_i64(b: &RecordBatch, name: &str) -> Result<Vec<i64>> {
    let a = col(b, name, &DataType::Int64)?;
    let a = a.as_any().downcast_ref::<Int64Array>().ok_or_else(|| anyhow!("{name}: не i64"))?;
    Ok((0..a.len()).map(|i| a.value(i)).collect())
}

/// Время в мс (из Timestamp любой точности).
pub fn col_ts(b: &RecordBatch, name: &str) -> Result<Vec<i64>> {
    let a = col(b, name, &TS)?;
    let a = a.as_any().downcast_ref::<TimestampMillisecondArray>().ok_or_else(|| anyhow!("{name}: не ts"))?;
    Ok((0..a.len()).map(|i| a.value(i)).collect())
}

/// Строки (даты, тикеры) — любой тип приводится к Utf8.
pub fn col_str(b: &RecordBatch, name: &str) -> Result<Vec<String>> {
    let a = col(b, name, &DataType::Utf8)?;
    let a = a.as_any().downcast_ref::<arrow::array::StringArray>().ok_or_else(|| anyhow!("{name}: не str"))?;
    Ok((0..a.len()).map(|i| a.value(i).to_string()).collect())
}

/// Бары из parquet (колонка времени begin или timestamp; value необязательна).
pub fn read_bars(path: &Path) -> Result<Vec<Bar>> {
    let mut out = Vec::new();
    for b in read_batches(path)? {
        let tname = if b.column_by_name("begin").is_some() { "begin" } else { "timestamp" };
        let ts = col_ts(&b, tname)?;
        let (o, h, l, c) = (col_f64(&b, "open")?, col_f64(&b, "high")?, col_f64(&b, "low")?, col_f64(&b, "close")?);
        let v = col_i64(&b, "volume")?;
        let val = col_f64(&b, "value").unwrap_or_else(|_| vec![f64::NAN; ts.len()]);
        for i in 0..ts.len() {
            out.push(Bar { ts: ts[i], open: o[i], high: h[i], low: l[i], close: c[i], volume: v[i], value: val[i] });
        }
    }
    out.sort_by_key(|b| b.ts);
    out.dedup_by_key(|b| b.ts);
    Ok(out)
}

fn write(path: &Path, schema: Schema, cols: Vec<ArrayRef>) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let schema = Arc::new(schema);
    let batch = RecordBatch::try_new(schema.clone(), cols)?;
    let props = WriterProperties::builder().set_compression(Compression::SNAPPY).build();
    let mut w = ArrowWriter::try_new(File::create(path)?, schema, Some(props))?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

pub fn write_bars(path: &Path, bars: &[Bar]) -> Result<()> {
    let f = |g: fn(&Bar) -> f64| -> ArrayRef { Arc::new(Float64Array::from_iter_values(bars.iter().map(g))) };
    let schema = Schema::new(vec![
        Field::new("begin", TS, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]);
    let cols: Vec<ArrayRef> = vec![
        Arc::new(TimestampMillisecondArray::from_iter_values(bars.iter().map(|b| b.ts))),
        f(|b| b.open),
        f(|b| b.high),
        f(|b| b.low),
        f(|b| b.close),
        Arc::new(Int64Array::from_iter_values(bars.iter().map(|b| b.volume))),
        f(|b| b.value),
    ];
    write(path, schema, cols)
}

/// Дневная таблица: date (Date32) + именованные колонки f64 (None -> null).
pub fn write_daily(path: &Path, days: &[i64], cols: &[(&str, Vec<Option<f64>>)]) -> Result<()> {
    let mut fields = vec![Field::new("date", DataType::Date32, false)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Date32Array::from_iter_values(days.iter().map(|d| *d as i32)))];
    for (n, v) in cols {
        fields.push(Field::new(*n, DataType::Float64, true));
        arrays.push(Arc::new(Float64Array::from(v.clone())));
    }
    write(path, Schema::new(fields), arrays)
}

/// Дневные RV: date, ticker, rv, bv, j, rsv_pos, rsv_neg, n_bars, n_quar.
pub fn write_rv(path: &Path, rows: &[(String, i64, crate::rv::DayRv, i64)]) -> Result<()> {
    let f = |g: fn(&crate::rv::DayRv) -> f64| -> ArrayRef { Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| g(&r.2)))) };
    let mut fields = vec![Field::new("date", DataType::Date32, false), Field::new("ticker", DataType::Utf8, false)];
    for n in ["rv", "bv", "j", "rsv_pos", "rsv_neg"] {
        fields.push(Field::new(n, DataType::Float64, false));
    }
    fields.push(Field::new("n_bars", DataType::Int64, false));
    fields.push(Field::new("n_quar", DataType::Int64, false));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Date32Array::from_iter_values(rows.iter().map(|r| r.1 as i32))),
        Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.0.as_str()))),
        f(|d| d.rv),
        f(|d| d.bv),
        f(|d| d.jump),
        f(|d| d.rsv_pos),
        f(|d| d.rsv_neg),
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2.n as i64))),
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.3))),
    ];
    write(path, Schema::new(fields), cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_roundtrip() -> Result<()> {
        let dir = std::env::temp_dir().join("moex_iss_test_rt");
        let p = dir.join("x.parquet");
        let bars = vec![
            Bar { ts: 36_000_000, open: 1.5, high: 2.0, low: 1.0, close: 1.75, volume: 7, value: 12.5 },
            Bar { ts: 36_600_000, open: 1.75, high: 2.0, low: 1.7, close: 1.9, volume: 9, value: 3.0 },
        ];
        write_bars(&p, &bars)?;
        assert_eq!(read_bars(&p)?, bars);
        Ok(())
    }
}
