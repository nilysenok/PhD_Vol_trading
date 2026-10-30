# GATES wf-L1b (ветка wf-zona-b)

| Гейт | CHECK | EXPECT | EVIDENCE |
|---|---|---|---|
| G1 день ≥ 2020 с 54 барами валиден (wf) | `python3 -m pytest tests/test_wf_zona_b.py -q` | 15 passed | 15 passed in 1.41s |
| G2 wf без отметки — отказ (vk, konec, vselennaya, bars5m, _read_cut) | то же | passed | passed |
| G3 train4/dev побайтно прежнее | то же + старые тесты period_v2/bars5m/vselennaya/istochniki/stavka | 114 passed, 2 deselected (по замыслу) | 114 passed, 2 deselected in 78s |
| G4 ставка за концом ряда — отказ | то же | passed | passed |
| G5 мутации: END=2020, fill_value=False, ffill ставки, граница без отметки | то же | passed | passed |
