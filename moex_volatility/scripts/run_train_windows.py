#!/usr/bin/env python3
"""Run daily_full_opt with 3 train window modes and compare."""
import subprocess, sys, re, time

modes = [
    (0, "expanding"),
    (2, "rolling_2yr"),
    (3, "rolling_3yr"),
]

results = {}
for tw, label in modes:
    print(f"\n{'#'*80}")
    print(f"# RUNNING: {label} (TRAIN_WINDOW={tw})")
    print(f"{'#'*80}\n")

    # Patch TRAIN_WINDOW in the script
    with open("daily_full_opt.py", "r") as f:
        code = f.read()
    code = re.sub(r'TRAIN_WINDOW = \d+', f'TRAIN_WINDOW = {tw}', code)
    with open("daily_full_opt.py", "w") as f:
        f.write(code)

    t0 = time.time()
    proc = subprocess.run([sys.executable, "daily_full_opt.py"],
                          capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    # Parse summary from output
    output = proc.stdout
    print(output[-3000:] if len(output) > 3000 else output)

    # Extract Mean(fb=0) per strategy
    summary = {}
    for line in output.split("\n"):
        for sid in ["S1_MeanRev","S2_Bollinger","S3_Donchian","S4_Supertrend","S5_PivotPoints","S6_VWAP"]:
            if sid in line and "fb=0" not in line and "Mean(fb=0)" in line:
                pass  # header
            if sid in line and "(fb=0)" not in line:
                # Try to parse from SUMMARY table
                parts = line.split()
                try:
                    idx = parts.index(sid)
                    # MeanExSh is next, Mean(fb=0) is after
                    mr = float(parts[idx+1])
                    mf = float(parts[idx+2])
                    summary[sid] = (mr, mf)
                except:
                    pass
    results[label] = (elapsed, summary)

# Final comparison
print(f"\n{'='*100}")
print("COMPARISON: Train Window Effect on Mean ExSh (fb=0)")
print(f"{'='*100}")
print(f"  {'Strategy':>16s}", end="")
for _, label in modes:
    print(f"  {label:>14s}", end="")
print()

sids = ["S1_MeanRev","S2_Bollinger","S3_Donchian","S4_Supertrend","S5_PivotPoints","S6_VWAP"]
for sid in sids:
    print(f"  {sid:>16s}", end="")
    for _, label in modes:
        _, summary = results[label]
        if sid in summary:
            mr, mf = summary[sid]
            print(f"  {mf:+14.2f}", end="")
        else:
            print(f"  {'N/A':>14s}", end="")
    print()

print(f"\n  {'Time (s)':>16s}", end="")
for _, label in modes:
    el, _ = results[label]
    print(f"  {el:14.0f}", end="")
print()
