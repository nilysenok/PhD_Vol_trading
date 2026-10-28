# data/events.csv и sobytiya_pit (Ж-070 п. 2)
Статус: разработка, данные до 2022. SHA1 data/events.csv: 7faa80a55ea788f46a4750ee1952e51e1d77553a (576 строк; повторный запуск скрипта — тот же SHA).
Тест: tests/test_sobytiya_pit.py, 8 passed; 5 мутаций ловятся (по data_sobytiya, < вместо <=, * не раздаётся, запрос >= 2022, строки >= 2022).
Отступления: (1) dividendy.events() отдаёт только last_cum_date, поэтому announce_date взята из dividendy.TABLE (путь файла берётся из модуля). (2) sobytiya_na имеет параметр vpered (по умолчанию 0): E1 смотрит заседание на d+1.
sobytiya_e1 сейчас: берёт data/calendar/cbr.csv (date, known_from) и дивиденды (announce_date) своими путями, даёт bool-маски по торговым датам; ось div выключена без div_verified.
Для подключения: sobytiya_e1 читает типы cbr/div из sobytiya_pit.sobytiya_na(d, tk, vpered=1) вместо двух csv; сверить маски побайтно со старыми; ключи кэша дополнить sobytiya_pit.digest().
