//! Общие типы: бар, список тикеров и индексов, время «настенных часов» (МСК) в мс.

pub const TICKERS: [&str; 17] = [
    "SBER", "VTBR", "MOEX", "LKOH", "TATN", "NVTK", "ALRS", "MTLR", "HYDR", "IRAO", "OGKB", "RTKM",
    "MTSS", "MGNT", "AFLT", "PHOR", "LSRG",
];
pub const INDICES: [&str; 3] = ["MCFTR", "RGBITR", "IMOEX"];
pub const MS_DAY: i64 = 86_400_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bar {
    /// begin, мс от 1970-01-01 по московским настенным часам (без пояса)
    pub ts: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
    pub value: f64,
}

impl Bar {
    /// Номер дня (дней от 1970-01-01).
    pub fn day(&self) -> i64 {
        day_of(self.ts)
    }
}

pub fn day_of(ts: i64) -> i64 {
    ts.div_euclid(MS_DAY)
}

/// Минуты от полуночи.
pub fn minute_of_day(ts: i64) -> i64 {
    ts.rem_euclid(MS_DAY) / 60_000
}

/// «ГГГГ-ММ-ДД» по номеру дня.
pub fn day_str(day: i64) -> String {
    chrono::DateTime::from_timestamp(day * 86_400, 0)
        .map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| format!("day{day}"))
}

/// «ГГГГ-ММ-ДД ЧЧ:ММ:СС» -> мс.
pub fn parse_ts(s: &str) -> Option<i64> {
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
        .ok()
        .map(|d| d.and_utc().timestamp_millis())
}

/// «ГГГГ-ММ-ДД...» -> номер дня.
pub fn parse_day(s: &str) -> Option<i64> {
    let d = chrono::NaiveDate::parse_from_str(s.get(..10)?, "%Y-%m-%d").ok()?;
    Some(d.and_hms_opt(0, 0, 0)?.and_utc().timestamp().div_euclid(86_400))
}

pub fn ts_str(ts: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ts)
        .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_default()
}
