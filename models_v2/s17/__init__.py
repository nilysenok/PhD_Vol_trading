"""Проверка гибрида SBER на новом периоде (s17, Ж-247/Ж-248), листы Л1–Л6: защита от конфликта OpenMP на macOS, как в `s14`/`s16`.
torch импортируется и закрепляется на один поток при импорте пакета, ДО xgboost/lightgbm; числа не меняются (класс № 152)."""
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:     # уже задано или параллельная работа уже началась
    pass
