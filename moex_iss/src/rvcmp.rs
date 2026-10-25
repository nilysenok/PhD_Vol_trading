//! Г3: сверка RV/BV/J с хранимыми признаками.
//!  A «формула»: хранимые бары (все сессии, как у признаков) -> наш rv  vs  хранимые признаки;
//!  B «ISS»:     ISS-бары (все сессии, минус карантин) -> наш rv  vs  хранимые признаки;
//!  C «основная»: ISS основная сессия vs хранимые бары основной сессии (оба — наш rv).
//! Дни: те, что есть у эталона; день без пары у проверяемой стороны считается расхождением.
use crate::model::day_str;
use crate::rv::{daily, DayRv};
use crate::session::filter_main;
use crate::store;
use crate::stored::{apply_corrections, load_features, Stored};
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub const TOL: f64 = 1e-6;
/// Нижний предел знаменателя: для J около нуля относительная ошибка не определена.
const FLOOR: f64 = 1e-12;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct CmpStat {
    pub days: usize,
    pub off: usize,
    pub missing: usize,
    pub max_rel_rv: f64,
    pub max_rel_bv: f64,
    pub max_rel_j: f64,
    pub max_rel_rsv: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OffDay {
    pub variant: String,
    pub ticker: String,
    pub day: String,
    pub rel_rv: f64,
    pub rel_bv: f64,
    pub rel_j: f64,
    pub note: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RvTicker {
    pub ticker: String,
    pub a: CmpStat,
    pub b: CmpStat,
    pub c: CmpStat,
}

pub fn rel(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    if d == 0.0 {
        0.0
    } else {
        d / a.abs().max(b.abs()).max(FLOOR)
    }
}

fn arr(d: &DayRv) -> [f64; 5] {
    [d.rv, d.bv, d.jump, d.rsv_pos, d.rsv_neg]
}

/// `got` проверяется против эталона `reference` на днях эталона.
pub fn cmp_days(variant: &str, ticker: &str, got: &BTreeMap<i64, [f64; 5]>, reference: &BTreeMap<i64, [f64; 5]>) -> (CmpStat, Vec<OffDay>) {
    let (mut s, mut off) = (CmpStat::default(), Vec::new());
    for (day, r) in reference {
        s.days += 1;
        let Some(g) = got.get(day) else {
            s.off += 1;
            s.missing += 1;
            off.push(OffDay { variant: variant.into(), ticker: ticker.into(), day: day_str(*day), rel_rv: f64::INFINITY, rel_bv: f64::INFINITY, rel_j: f64::INFINITY, note: "нет дня".into() });
            continue;
        };
        let rl: Vec<f64> = (0..5).map(|i| rel(g[i], r[i])).collect();
        s.max_rel_rv = s.max_rel_rv.max(rl[0]);
        s.max_rel_bv = s.max_rel_bv.max(rl[1]);
        s.max_rel_j = s.max_rel_j.max(rl[2]);
        s.max_rel_rsv = s.max_rel_rsv.max(rl[3]).max(rl[4]);
        if rl[0] > TOL || rl[1] > TOL || rl[2] > TOL {
            s.off += 1;
            off.push(OffDay { variant: variant.into(), ticker: ticker.into(), day: day_str(*day), rel_rv: rl[0], rel_bv: rl[1], rel_j: rl[2], note: String::new() });
        }
    }
    (s, off)
}

fn to_arr(m: &BTreeMap<i64, DayRv>, lo: i64, hi: i64) -> BTreeMap<i64, [f64; 5]> {
    m.iter().filter(|(d, _)| (lo..=hi).contains(d)).map(|(d, v)| (*d, arr(v))).collect()
}

pub fn run(out: &Path, st: &Stored, tickers: &[String]) -> Result<()> {
    let res: Vec<Result<(RvTicker, Vec<OffDay>)>> = tickers
        .par_iter()
        .map(|t| {
            let feat = load_features(&st.features_path(), t)?;
            let reference: BTreeMap<i64, [f64; 5]> = feat.iter().map(|(d, f)| (*d, [f[0], f[1], f[2], f[3], f[4]])).collect();
            let (lo, hi) = (reference.keys().next().copied().unwrap_or(0), reference.keys().next_back().copied().unwrap_or(-1));
            let (stored, q) = (st.bars(t)?, st.quarantine(t));
            let iss_full = store::read_bars(&out.join("bars_10m_full").join(format!("{t}.parquet")))?;
            // ряд ISS начинается раньше хранимого: обрезаем по первому хранимому бару (иначе ночная доходность первого дня)
            let t0 = stored.first().map(|b| b.ts).unwrap_or(0);
            let iss: Vec<_> = apply_corrections(&iss_full, &q).into_iter().filter(|b| b.ts >= t0).collect();
            let a = to_arr(&daily(&stored), lo, hi);
            let b = to_arr(&daily(&iss), lo, hi);
            let (sm, im) = (daily(&filter_main(&stored)), daily(&filter_main(&iss)));
            let c_ref = to_arr(&sm, lo, hi);
            let c_got = to_arr(&im, lo, hi);
            let ((ca, oa), (cb, ob), (cc, oc)) = (cmp_days("A", t, &a, &reference), cmp_days("B", t, &b, &reference), cmp_days("C", t, &c_got, &c_ref));
            let mut off = [oa, ob, oc].concat();
            off.sort_by(|x, y| y.rel_rv.total_cmp(&x.rel_rv));
            Ok((RvTicker { ticker: t.clone(), a: ca, b: cb, c: cc }, off))
        })
        .collect();
    let (mut sum, mut offs) = (Vec::new(), Vec::new());
    for r in res {
        let (s, o) = r?;
        eprintln!("{}: дней {} | A off {} | B off {} (макс RV {:.2e}) | C off {}", s.ticker, s.a.days, s.a.off, s.b.off, s.b.max_rel_rv, s.c.off);
        sum.push(s);
        offs.extend(o);
    }
    std::fs::create_dir_all(out.join("report"))?;
    std::fs::write(out.join("report/rv_summary.json"), serde_json::to_vec_pretty(&sum)?)?;
    let mut csv = String::from("variant,ticker,day,rel_rv,rel_bv,rel_j,note\n");
    for o in offs.iter().take(4000) {
        csv += &format!("{},{},{},{:.3e},{:.3e},{:.3e},{}\n", o.variant, o.ticker, o.day, o.rel_rv, o.rel_bv, o.rel_j, o.note);
    }
    std::fs::write(out.join("report/rv_off_days.csv"), csv)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_off_days_and_missing() {
        let r: BTreeMap<i64, [f64; 5]> = [(1, [1e-4; 5]), (2, [2e-4; 5]), (3, [3e-4; 5])].into_iter().collect();
        let mut g = r.clone();
        g.insert(2, [2.1e-4; 5]);
        g.remove(&3);
        let (s, off) = cmp_days("X", "T", &g, &r);
        assert_eq!((s.days, s.off, s.missing, off.len()), (3, 2, 1, 2));
        assert!(s.max_rel_rv > 0.04);
        assert_eq!(cmp_days("X", "T", &r, &r).0.off, 0);
    }
}
