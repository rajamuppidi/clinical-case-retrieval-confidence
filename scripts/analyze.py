"""Development-only fitting followed by frozen held-out evaluation."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from retrieval import FEATURES, ROOT, CACHE

COVERAGES=np.round(np.arange(.2,1.001,.05),2)
METRICS=['hit10','ndcg10','precision10','mrr1000','recall1000']

def curve(scores,loss,ids,weights=None):
    order=np.lexsort((np.asarray(ids).astype(str),-np.asarray(scores)))
    ends=np.ceil(COVERAGES*len(loss)).astype(int)-1
    w=np.ones(len(loss)) if weights is None else weights
    c=np.cumsum(np.asarray(loss)[order]*w[order])/np.cumsum(w[order])
    return c[ends]

def msf(scores,loss,ids,weights=None):
    return float(curve(scores,loss,ids,weights).mean())

def model():
    return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,solver='lbfgs',random_state=20261006))

def read(split,method,view):
    df=pd.read_csv(ROOT/f'results/{split}_{method}_{view}.csv',dtype={'query_id':str,'source_pmid':str})
    return df.sort_values('query_id').reset_index(drop=True)

def scores_for(df,bundle):
    return dict(logistic=bundle['model'].predict_proba(df[FEATURES])[:,1],
                length_only=bundle['length_model'].predict_proba(df[['log_tokens']])[:,1],
                selected_univariate=df[bundle['best']['feature']].to_numpy()*bundle['best']['sign'],
                raw_top_score=df.top_score.to_numpy(),relative_gap=df.relative_gap.to_numpy())

def fit(method):
    df=read('dev',method,'full'); fit=df[df.partition=='fit']; val=df[df.partition=='validation']; cal=df[df.partition=='calibration']
    candidates=[];loss=1-val.hit10.to_numpy()
    for feature in FEATURES:
        for sign in [1,-1]:
            candidates.append(dict(feature=feature,sign=sign,validation_msf=msf(sign*val[feature].to_numpy(),loss,val.query_id)))
    best=min(candidates,key=lambda x:x['validation_msf'])
    first=model().fit(fit[FEATURES],fit.hit10)
    validation_lr=msf(first.predict_proba(val[FEATURES])[:,1],loss,val.query_id)
    final=df[df.partition!='calibration']
    bundle={'model':model().fit(final[FEATURES],final.hit10),
            'length_model':model().fit(final[['log_tokens']],final.hit10),'best':best}
    scores=scores_for(cal,bundle)
    bundle['thresholds']={name:{str(c):float(np.quantile(s,1-c)) for c in [.2,.5,.8]} for name,s in scores.items()}
    report=dict(method=method,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                protocol_sha256=None,
                train_count=len(fit),validation_count=len(val),calibration_count=len(cal),
                validation_base_failure=float(loss.mean()),validation_logistic_msf=validation_lr,
                best=best,candidates=candidates,thresholds=bundle['thresholds'],
                final_standardized_coefficients=dict(zip(FEATURES,bundle['model'][-1].coef_[0].tolist())))
    CACHE.mkdir(exist_ok=True,parents=True)
    with (CACHE/f'{method}_selectors.pkl').open('wb') as f:pickle.dump(bundle,f)
    (ROOT/f'results/{method}_development_selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['method','validation_base_failure','validation_logistic_msf','best']}),flush=True)

def ece(y,p):
    bins=np.minimum((p*10).astype(int),9)
    return float(sum(np.mean(bins==b)*abs(np.mean(y[bins==b])-np.mean(p[bins==b])) for b in range(10) if np.any(bins==b)))

def paired_bootstrap(df,scores,threshold,iterations=2000):
    group_rows=list(df.groupby('source_pmid',sort=True).indices.values())
    rng=np.random.default_rng(20261006);loss=1-df.hit10.to_numpy();ids=df.query_id.to_numpy()
    arrays={k:[] for k in ['msf_gain','logistic50_coverage','logistic50_failure']}
    for _ in range(iterations):
        ix=np.concatenate([group_rows[g] for g in rng.integers(0,len(group_rows),len(group_rows))])
        arrays['msf_gain'].append(msf(scores['selected_univariate'][ix],loss[ix],ids[ix])-msf(scores['logistic'][ix],loss[ix],ids[ix]))
        take=scores['logistic'][ix]>=threshold
        arrays['logistic50_coverage'].append(take.mean())
        arrays['logistic50_failure'].append(float(loss[ix][take].mean()) if take.any() else np.nan)
    return {key:[float(x) for x in np.nanquantile(values,[.025,.975])] for key,values in arrays.items()}

def evaluate(method,view):
    with (CACHE/f'{method}_selectors.pkl').open('rb') as f:bundle=pickle.load(f)
    df=read('test',method,view);loss=1-df.hit10.to_numpy();ids=df.query_id.to_numpy()
    scores=scores_for(df,bundle);weights=1/df.groupby('source_pmid').source_pmid.transform('size').to_numpy()
    result=dict(method=method,view=view,n=len(df),articles=df.source_pmid.nunique(),base={k:float(df[k].mean()) for k in METRICS},
                selectors={},best_univariate=bundle['best'],article_equal_base_hit10=float(np.average(df.hit10,weights=weights)))
    curve_rows=[]
    for name,score in scores.items():
        cc=curve(score,loss,ids)
        report=dict(msf=float(cc.mean()),mean_selected_ndcg_loss=msf(score,1-df.ndcg10.to_numpy(),ids),
                    article_equal_msf=msf(score,loss,ids,weights),auroc=float(roc_auc_score(df.hit10,score)),operating_points={})
        for frac in [.2,.5,.8]:
            thresh=bundle['thresholds'][name][str(frac)];take=score>=thresh
            report['operating_points'][str(frac)]=dict(threshold=thresh,accepted=int(take.sum()),coverage=float(take.mean()),
                                                      failure=float(loss[take].mean()) if take.any() else None,
                                                      ndcg10=float(df.ndcg10.to_numpy()[take].mean()) if take.any() else None)
        result['selectors'][name]=report
        for c,r in zip(COVERAGES,cc):curve_rows.append(dict(method=method,view=view,selector=name,coverage=float(c),failure=float(r)))
    result['probability_calibration']=dict(brier=float(brier_score_loss(df.hit10,scores['logistic'])),ece10=ece(df.hit10.to_numpy(),scores['logistic']),
                                            predicted_success_mean=float(scores['logistic'].mean()),observed_success=float(df.hit10.mean()))
    result['msf_gain']=result['selectors']['selected_univariate']['msf']-result['selectors']['logistic']['msf']
    result['bootstrap95']=paired_bootstrap(df,scores,bundle['thresholds']['logistic']['0.5'])
    subgroup_rows=[]
    df['length_band']=pd.cut(df.tokens_full,[-1,128,512,np.inf],labels=['<=128','129-512','>512'])
    df['qrel_band']=pd.cut(df.qrel_count,[0,1,4,np.inf],labels=['1','2-4','>=5'])
    for dimension in ['length_band','qrel_band']:
        for label,group in df.groupby(dimension,observed=True):
            ix=group.index.to_numpy()
            for name in scores:
                take=scores[name][ix]>=bundle['thresholds'][name]['0.5']
                subgroup_rows.append(dict(method=method,view=view,dimension=dimension,subgroup=str(label),selector=name,
                                          n=len(ix),accepted=int(take.sum()),coverage=float(take.mean()),base_failure=float(loss[ix].mean()),
                                          selected_failure=float(loss[ix][take].mean()) if take.any() else None))
    pred=df[['query_id','source_pmid','hit10','ndcg10']].copy()
    for name,score in scores.items():pred[name]=score
    pred.to_csv(ROOT/f'results/test_{method}_{view}_predictions.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(ROOT/f'results/test_{method}_{view}_curves.csv',index=False)
    pd.DataFrame(subgroup_rows).to_csv(ROOT/f'results/test_{method}_{view}_subgroups.csv',index=False)
    (ROOT/f'results/test_{method}_{view}_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['method','view','base','msf_gain','bootstrap95','probability_calibration']},indent=2),flush=True)
    return result

def summarize():
    os.environ.setdefault('MPLCONFIGDIR',str(CACHE/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_results=[];rows=[]
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    labels={'logistic':'Logistic model','selected_univariate':'Best single feature','raw_top_score':'Top score','relative_gap':'Relative gap','length_only':'Length only'}
    for i,method in enumerate(['bm25','tfidf']):
        for j,view in enumerate(['full','prefix128']):
            r=json.loads((ROOT/f'results/test_{method}_{view}_analysis.json').read_text());all_results.append(r)
            for name,x in r['selectors'].items():
                op=x['operating_points']['0.5']
                rows.append(dict(method=method,view=view,selector=name,base_hit10=r['base']['hit10'],msf=x['msf'],
                                 coverage_at_frozen50=op['coverage'],failure_at_frozen50=op['failure'],auroc=x['auroc']))
            curves=pd.read_csv(ROOT/f'results/test_{method}_{view}_curves.csv');ax=axes[i,j]
            for name,style in [('logistic','-'),('selected_univariate','--'),('raw_top_score',':'),('relative_gap','-.')]:
                c=curves[curves.selector==name];ax.plot(c.coverage,c.failure,style,label=labels[name],linewidth=1.8)
            ax.axhline(1-r['base']['hit10'],color='gray',linestyle='--',linewidth=1,label='Random expectation')
            ax.set_title(f'{method.upper()} | '+('Complete narrative' if view=='full' else 'First 128 tokens'))
            ax.grid(alpha=.2);ax.set_ylim(0,1);ax.set_xlim(.2,1)
            if i==1:ax.set_xlabel('Fraction of queries retained')
            if j==0:ax.set_ylabel('No labeled match in top 10')
    handles,ll=axes[0,0].get_legend_handles_labels();fig.legend(handles,ll,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.09,1,1));(ROOT/'artifacts').mkdir(exist_ok=True)
    fig.savefig(ROOT/'artifacts/risk_coverage.png',dpi=200);fig.savefig(ROOT/'artifacts/risk_coverage.svg');plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT/'results/summary.csv',index=False)
    (ROOT/'results/summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','evaluate','summarize']);p.add_argument('--method',choices=['bm25','tfidf'],default='bm25');p.add_argument('--view',choices=['full','prefix128'],default='full');args=p.parse_args()
    if args.action=='fit':fit(args.method)
    elif args.action=='evaluate':evaluate(args.method,args.view)
    else:summarize()
