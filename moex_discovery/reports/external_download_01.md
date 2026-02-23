# External Data Download Report: MOEX ISS Indices & Currency

**Date:** 2026-02-05 19:16
**Source:** MOEX ISS API
**Interval:** 10 minutes

## Summary

| # | Instrument | Type | Board | Candles | Days | Bars/day | First Date | Last Date |
|---|------------|------|-------|---------|------|----------|------------|-----------|
| 1 | IMOEX | index | SNDX | 186,052 | 3,559 | 52.3 | 2011-12-08 | 2026-02-05 |
| 2 | MOEXOG | index | SNDX | 177,772 | 3,556 | 50.0 | 2011-12-08 | 2026-02-05 |
| 3 | MOEXFN | index | SNDX | 177,773 | 3,556 | 50.0 | 2011-12-08 | 2026-02-05 |
| 4 | MOEXMM | index | SNDX | 177,773 | 3,556 | 50.0 | 2011-12-08 | 2026-02-05 |
| 5 | MOEXEU | index | SNDX | 177,774 | 3,556 | 50.0 | 2011-12-08 | 2026-02-05 |
| 6 | MOEXTL | index | SNDX | 177,774 | 3,556 | 50.0 | 2011-12-08 | 2026-02-05 |
| 7 | MOEXCN | index | SNDX | 177,775 | 3,556 | 50.0 | 2011-12-08 | 2026-02-05 |
| 8 | MOEXCH | index | SNDX | 177,775 | 3,556 | 50.0 | 2011-12-08 | 2026-02-05 |
| 9 | RGBI | index | SNDX | 165,224 | 3,534 | 46.8 | 2011-12-08 | 2026-02-05 |
| 10 | RVI | index | RTSI | 223,000 | 3,057 | 72.9 | 2013-12-16 | 2026-02-05 |
| 11 | CNYRUB_TOM | currency | CETS | 151,054 | 3,220 | 46.9 | 2013-04-15 | 2026-02-05 |
| 12 | USD000UTSTOM | currency | CETS | 223,858 | 3,154 | 71.0 | 2011-12-08 | 2024-06-11 |

## Totals

- **Instruments downloaded:** 12
- **Total candles:** 2,193,604
- **Total unique days:** 41,416

## Data Gaps > 7 Days

### IMOEX

- 2012-12-28 — 2013-01-08 (11 days)
- 2022-02-25 — 2022-03-24 (27 days)

### MOEXOG

- 2012-12-28 — 2013-01-08 (11 days)
- 2022-02-25 — 2022-03-28 (31 days)

### MOEXFN

- 2012-12-28 — 2013-01-08 (11 days)
- 2022-02-25 — 2022-03-28 (31 days)

### MOEXMM

- 2012-12-28 — 2013-01-08 (11 days)
- 2022-02-25 — 2022-03-28 (31 days)

### MOEXEU

- 2012-12-28 — 2013-01-08 (11 days)
- 2022-02-25 — 2022-03-28 (31 days)

### MOEXTL

- 2012-12-28 — 2013-01-08 (11 days)
- 2022-02-25 — 2022-03-28 (31 days)

### MOEXCN

- 2012-12-28 — 2013-01-08 (11 days)
- 2022-02-25 — 2022-03-28 (31 days)

### MOEXCH

- 2012-12-28 — 2013-01-08 (11 days)
- 2022-02-25 — 2022-03-28 (31 days)

### RGBI

- 2012-12-28 — 2013-01-08 (11 days)
- 2016-02-17 — 2016-02-29 (12 days)
- 2016-02-29 — 2016-03-29 (29 days)
- 2022-02-25 — 2022-03-21 (24 days)

### RVI

- 2022-02-25 — 2022-03-09 (12 days)
- 2022-03-11 — 2022-03-24 (13 days)

### CNYRUB_TOM

- 2013-12-30 — 2014-01-09 (10 days)

### USD000UTSTOM

- 2012-12-28 — 2013-01-08 (11 days)


## File Locations

- **Indices:** `moex_discovery/data/external/moex_iss/indices_10m/`
- **Currency:** `moex_discovery/data/external/moex_iss/currency_10m/`

## Notes

- All data downloaded for the FULL available period (no date truncation)
- Pagination handled automatically (500 candles per request)
- Rate limiting: 0.2 sec between requests
