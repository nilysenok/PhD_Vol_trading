//! Г4: сборка отчёта из сводок report/*.json.
use crate::idxcmp::IndexStat;
use crate::rvcmp::{RvTicker, TOL};
use crate::verify::{Diff, TickerVerify};
use anyhow::Result;
use serde::de::DeserializeOwned;
use std::path::Path;

fn load<T: DeserializeOwned>(p: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(p)?)?)
}

fn pc(x: f64) -> String {
    format!("{:.3}%", x * 100.0).replace('.', ",")
}

fn sci(x: f64) -> String {
    if x.is_infinite() { "—".into() } else { format!("{x:.1e}") }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}

fn bars_table(v: &[TickerVerify]) -> String {
    let mut s = String::from("| Тикер | Баров ISS | Баров хранимых | Совпало | В карантине пайплайна | Лишних в ISS | Нет в ISS | Доля совпавших | Дней ≠52 бара (ISS) | Дней ≠52 бара (хранимые) |\n|---|---|---|---|---|---|---|---|---|---|\n");
    for t in v {
        s += &format!("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n", t.ticker, t.iss_bars, t.stored_bars, t.matched, t.only_iss_quarantine, t.only_iss_extra, t.only_stored, pc(t.share_matched), t.days_incomplete_iss, t.days_incomplete_stored);
    }
    let sum = |f: fn(&TickerVerify) -> usize| v.iter().map(f).sum::<usize>();
    let denom = sum(|t| t.matched + t.mismatched + t.only_iss_extra + t.only_stored);
    s += &format!("| **Итого** | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n", sum(|t| t.iss_bars), sum(|t| t.stored_bars), sum(|t| t.matched), sum(|t| t.only_iss_quarantine), sum(|t| t.only_iss_extra), sum(|t| t.only_stored), pc(ratio(sum(|t| t.matched), denom)), sum(|t| t.days_incomplete_iss), sum(|t| t.days_incomplete_stored));
    s
}

fn rv_table(v: &[RvTicker]) -> String {
    let mut s = String::from("| Тикер | Дней | A: формула, расх. дней | B: ISS-бары, расх. дней | B: макс. отн. RV | B: макс. отн. BV | B: макс. отн. J | C: основная сессия, расх. дней |\n|---|---|---|---|---|---|---|---|\n");
    for t in v {
        s += &format!("| {} | {} | {} ({}) | {} ({}) | {} | {} | {} | {} ({}) |\n", t.ticker, t.a.days, t.a.off, pc(ratio(t.a.off, t.a.days)), t.b.off, pc(ratio(t.b.off, t.b.days)), sci(t.b.max_rel_rv), sci(t.b.max_rel_bv), sci(t.b.max_rel_j), t.c.off, pc(ratio(t.c.off, t.c.days)));
    }
    let (d, a, b, c) = (v.iter().map(|t| t.a.days).sum(), v.iter().map(|t| t.a.off).sum(), v.iter().map(|t| t.b.off).sum(), v.iter().map(|t| t.c.off).sum());
    s += &format!("| **Итого** | {d} | {a} ({}) | {b} ({}) | | | | {c} ({}) |\n", pc(ratio(a, d)), pc(ratio(b, d)), pc(ratio(c, v.iter().map(|t| t.c.days).sum())));
    s
}

fn index_table(v: &[IndexStat]) -> String {
    let mut s = String::from("| Индекс | Дней ISS | Период | С чем сверено | Общих дней | Расхождений (>1e-6) | Макс. отн. | Только в ISS | Только в хранимом |\n|---|---|---|---|---|---|---|---|---|\n");
    for t in v {
        s += &format!("| {} | {} | {} … {} | {} ({} дн.) | {} | {} | {} | {} | {} |\n", t.index, t.rows, t.from, t.to, t.reference, t.ref_rows, t.common, t.off, sci(t.max_rel), t.only_iss, t.only_ref);
    }
    s
}

pub fn run(out: &Path, dest: &Path) -> Result<()> {
    let r = out.join("report");
    let bars: Vec<TickerVerify> = load(&r.join("verify_summary.json"))?;
    let rv: Vec<RvTicker> = load(&r.join("rv_summary.json"))?;
    let diffs: Vec<Diff> = load(&r.join("verify_diffs_top.json"))?;
    let idx: Vec<IndexStat> = load(&r.join("index_summary.json"))?;
    let days: usize = rv.iter().map(|t| t.b.days).sum();
    let off_b: usize = rv.iter().map(|t| t.b.off).sum();
    let share = ratio(off_b, days);
    let matched: usize = bars.iter().map(|t| t.matched).sum();
    let denom: usize = bars.iter().map(|t| t.matched + t.mismatched + t.only_iss_extra + t.only_stored).sum();
    let ix_bad: Vec<String> = idx.iter().filter(|t| t.off + t.only_iss + t.only_ref > 0).map(|t| format!("{} ({} дн. не совпало, {} только в ISS, {} только в хранимом)", t.index, t.off, t.only_iss, t.only_ref)).collect();
    let head = format!("бары совпали на {}, RV/BV/J расходятся в {} дней (допуск 0,1%)", pc(ratio(matched, denom)), pc(share));
    let verdict = if share > 0.001 {
        format!("**Расхождения в RV: {} дней ({off_b} из {days}) отличаются от хранимых признаков более чем на {TOL:.0e}; {head}. Признаки не пересчитывались (решение владельца).**", pc(share))
    } else if ix_bad.is_empty() {
        format!("**Данные подтверждены: {head}; индексы совпали.**")
    } else {
        format!("**Данные по акциям подтверждены ({head}); расхождения в индексах: {}. Признаки не пересчитывались (решение владельца).**", ix_bad.join("; "))
    };
    let mut md = format!("{verdict}\n\n# Проверка данных через MOEX ISS (дорожка Г)\n\n");
    md += "## 1. Бары (10 мин, TQBR, основная сессия begin ∈ [10:00; 18:40))\n\n";
    md += &bars_table(&bars);
    md += &format!("\n### Топ-20 расхождений бар к бару\n\n| № | Тикер | begin | Макс. отн. расхождение | Что |\n|---|---|---|---|---|\n");
    if diffs.is_empty() {
        md += "| — | — | — | — | расхождений бар к бару нет |\n";
    }
    for (i, d) in diffs.iter().take(20).enumerate() {
        md += &format!("| {} | {} | {} | {} | {} |\n", i + 1, d.ticker, d.ts, sci(d.score), d.what);
    }
    md += "\n## 2. RV, BV, J (допуск 1e-6 по относительной разнице)\n\n";
    md += &rv_table(&rv);
    md += "\n## 3. Индексы (дневные)\n\n";
    md += &index_table(&idx);
    md += "\n";
    md += include_str!("../report_notes.md");
    std::fs::write(dest, md)?;
    eprintln!("отчёт: {}", dest.display());
    Ok(())
}
