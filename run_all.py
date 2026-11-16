"""Runs the full analysis pipeline in order. All results are written to output/."""
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

# (script, log file for console output or None)
STEPS = [
    ('main.py', None),                  # filtered dataset + figures (CBA-framed)
    ('main_comparison.py', None),       # figures/tables CBA-framed vs. general adaptation
    ('data_tables.py', None),           # Supplementary Annex D tables
    ('dimension_tests.py', 'dimension_tests_log.txt'),  # chi² tests of keyword dimensions
    ('regression.py', 'regression_log.txt'),            # trade / vulnerability regressions
]

(BASE / 'output').mkdir(exist_ok=True)
env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}

for script, log in STEPS:
    print(f"=== {script} ===", flush=True)
    if log:
        with open(BASE / 'output' / log, 'w', encoding='utf-8') as f:
            result = subprocess.run([sys.executable, script], cwd=BASE, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"    output written to output/{log}")
    else:
        result = subprocess.run([sys.executable, script], cwd=BASE, env=env)
    if result.returncode != 0:
        sys.exit(f"{script} failed with exit code {result.returncode}")

print("\nDone. All results are in output/.")
