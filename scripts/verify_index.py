"""Compare indexed BM25 scores with an independent scalar implementation."""
from pathlib import Path
import json,pickle,re
from collections import Counter
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
with (ROOT/'data/cache/index.pkl').open('rb') as f: index=pickle.load(f)
corpus=[json.loads(l) for l in (ROOT/'data/raw/PPR/corpus.jsonl').open()]
pattern=re.compile(r'(?u)\b\w\w+\b')
lengths=np.array([len(pattern.findall(x['text'].lower())) for x in corpus])
terms=set(pattern.findall('no fever chest pain'))
q=index['vectorizer'].transform(['no fever chest pain']);q.data[:]=1
scores=(sparse.load_npz(ROOT/'data/cache/bm25.npz')@q.T).toarray().ravel()
for i in [0,100,1000,int(np.argmax(scores))]:
    counts=Counter(pattern.findall(corpus[i]['text'].lower())); expected=0
    for term in terms:
        tf=counts[term]
        if tf:
            idf=index['idf'][index['vectorizer'].vocabulary_[term]]
            expected+=idf*tf*2.2/(tf+1.2*(.25+.75*lengths[i]/lengths.mean()))
    np.testing.assert_allclose(scores[i],expected,rtol=2e-6,atol=1e-6)
print('PASS: sparse BM25 matches independent scalar computation on four documents including the top match')
