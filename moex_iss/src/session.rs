//! Основная сессия: begin в [10:00; 18:40) — ровно 52 бара (10:00..18:30).
//! Хранимый пайплайн (moex_discovery/src/pipeline_d001_d003.py:353-357) берёт begin 10:00..18:50
//! включительно, т.е. ещё и бары закрывающего аукциона 18:40 и 18:50 (до 54 баров в дне);
//! здесь они отброшены по заданию, при сверке хранимые бары режутся тем же фильтром.
use crate::model::{minute_of_day, Bar};

pub const START_MIN: i64 = 10 * 60;
/// begin строго меньше 18:40 (последний бар — 18:30..18:40).
pub const END_MIN_EXCL: i64 = 18 * 60 + 40;
pub const EXPECTED_BARS: usize = 52;

pub fn in_main(ts: i64) -> bool {
    let m = minute_of_day(ts);
    (START_MIN..END_MIN_EXCL).contains(&m)
}

pub fn filter_main(bars: &[Bar]) -> Vec<Bar> {
    bars.iter().filter(|b| in_main(b.ts)).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::parse_ts;

    fn bar(s: &str) -> Bar {
        let ts = parse_ts(s).unwrap_or(0);
        Bar { ts, open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: 1, value: 1.0 }
    }

    #[test]
    fn filters_sessions() {
        let all: Vec<Bar> = [
            "2024-03-01 06:50:00",
            "2024-03-01 09:50:00",
            "2024-03-01 10:00:00",
            "2024-03-01 18:30:00",
            "2024-03-01 18:40:00",
            "2024-03-01 18:50:00",
            "2024-03-01 19:00:00",
            "2024-03-01 23:40:00",
        ]
        .iter()
        .map(|s| bar(s))
        .collect();
        let m = filter_main(&all);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].ts, all[2].ts);
        assert_eq!(m[1].ts, all[3].ts);
    }

    #[test]
    fn full_day_has_52_bars() {
        let n = (0..144).map(|i| i * 600_000).filter(|t| in_main(*t)).count();
        assert_eq!(n, EXPECTED_BARS);
    }
}
