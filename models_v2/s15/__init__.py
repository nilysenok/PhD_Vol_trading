"""Deep Structural Extension (Ж-239), SBER, h = 1: лист DE0 — диагностика по СОХРАНЁННЫМ прогнозам V1–V7. Обучения нет.
torch импортируется первым: конфликт OpenMP lightgbm/xgboost/torch (`docs/otchety/harj_hybrid_v7/REPRODUCTION_V7.md`); сам пакет s15 ничего не обучает."""
import torch  # noqa: F401  # только порядок импорта (OpenMP), не обучение
