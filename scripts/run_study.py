"""Rebuild the study in its prescribed order; downloads require network access."""
from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'artifacts').mkdir(exist_ok=True)
def run(*args):subprocess.run([sys.executable,*args],check=True)
run('scripts/download_data.py')
run('-m','unittest','discover','-s','tests')
run('scripts/retrieval.py','audit')
run('scripts/retrieval.py','index')
for method in ['bm25','tfidf']:
    run('scripts/retrieval.py','retrieve','--split','dev','--method',method,'--view','full')
    run('scripts/analyze.py','fit','--method',method)
for method in ['bm25','tfidf']:
    run('scripts/retrieval.py','retrieve','--split','dev','--method',method,'--view','prefix128')
    for view in ['full','prefix128']:
        run('scripts/retrieval.py','retrieve','--split','test','--method',method,'--view',view)
        run('scripts/analyze.py','evaluate','--method',method,'--view',view)
run('scripts/analyze.py','summarize')
