"""Печать списка точек LB0 (для GATES_B_lgbm.md): python3 models_v2/s6/lgbm/print_grid.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models_v2.s6.lgbm import grid  # noqa: E402

if __name__ == "__main__":
    g = grid.full_grid()
    for p in grid.lb0_points():
        print("№%3d  %s" % (g.index(p), p.label()))
    print(len(g), len(grid.lb0_points()))
