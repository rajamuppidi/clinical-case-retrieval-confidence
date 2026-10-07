"""Paired descriptive comparisons for the post hoc shortening analysis."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VIEWS={'bm25':['full','prefix64','prefix128','headtail128','highidf128','prefix256'],
       'tfidf':['full','prefix128','highidf128']}

def read(split,method,view):
    return pd.read_csv(ROOT/f'results/{split}_{method}_{view}.csv',dtype={'query_id':str,'source_pmid':str}).sort_values('query_id').reset_index(drop=True)

def interval(df,a,b,n=2000):
    groups=list(df.groupby('source_pmid',sort=True).indices.values())
    rng=np.random.default_rng(20261007)
    delta=a-b
    sims=[]
    for _ in range(n):
        ix=np.concatenate([groups[g] for g in rng.integers(0,len(groups),len(groups))])
        sims.append(float(delta[ix].mean()))
    return [float(v) for v in np.quantile(sims,[.025,.975])]

def main():
    out=[]
    for method,views in VIEWS.items():
        for split in ['dev','test']:
            data={v:read(split,method,v) for v in views}
            base=data['prefix128']
            assert all(base.query_id.equals(d.query_id) for d in data.values())
            for view,d in data.items():
                row=dict(method=method,split=split,view=view,n=len(d),hit10=float(d.hit10.mean()),
                         ndcg10=float(d.ndcg10.mean()),unchanged_n=int((d.tokens_used==d.tokens_full).sum()),
                         mean_fraction_tokens_retained=float((d.tokens_used/d.tokens_full.clip(lower=1)).mean()))
                if view!='prefix128':
                    row['hit10_minus_prefix128']=float(d.hit10.mean()-base.hit10.mean())
                    row['hit10_minus_prefix128_95article_bootstrap']=interval(d,d.hit10.to_numpy(),base.hit10.to_numpy())
                out.append(row)
    (ROOT/'results/extension_retrieval_comparisons.json').write_text(json.dumps(out,indent=2)+'\n')
    pd.DataFrame(out).to_csv(ROOT/'results/extension_retrieval_comparisons.csv',index=False)
    for row in out:
        print(row)

if __name__=='__main__':main()
