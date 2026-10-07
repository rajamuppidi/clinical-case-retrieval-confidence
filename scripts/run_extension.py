"""Run the documented exploratory query-form analysis after run_study.py."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

for method, views in [('bm25', ['prefix64', 'prefix256', 'headtail128', 'highidf128']),
                      ('tfidf', ['highidf128'])]:
    for view in views:
        for split in ['dev', 'test']:
            subprocess.run([sys.executable, 'scripts/retrieval.py', 'retrieve',
                            '--split', split, '--method', method, '--view', view], check=True)

subprocess.run([sys.executable, 'scripts/extension.py'], check=True)
subprocess.run([sys.executable, 'scripts/extension_comparisons.py'], check=True)
