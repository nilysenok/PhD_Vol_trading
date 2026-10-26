"""С9: короткий просмотр таблиц docs/tables/strat_dev/*.csv (python3 -m fastbt.s9_show <имя> [столбцы…])."""
import sys

import pandas as pd

from fastbt.s9 import TABLES


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    df = pd.read_csv(TABLES / a[0], sep=";", decimal=",")
    if len(a) > 1:
        df = df[[c for c in a[1:] if c in df.columns]]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 60)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
