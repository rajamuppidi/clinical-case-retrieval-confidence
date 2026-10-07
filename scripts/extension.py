"""Post hoc, development-fitted query-form robustness analysis."""
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.special import logit
from analyze import CACHE, ROOT, FEATURES, ece, model, msf, read

VIEWS = {'bm25': ['prefix64', 'prefix128', 'headtail128', 'highidf128', 'prefix256'],
         'tfidf': ['prefix128','highidf128']}
N_BOOT = 1000

def assess(y, p, threshold, groups, ids):
    take = p >= threshold
    loss = 1-y
    return dict(n=len(y), accepted=int(take.sum()), coverage=float(take.mean()),
                selected_failure=float(loss[take].mean()) if take.any() else None,
                brier=float(brier_score_loss(y,p)), ece10=ece(y,p),
                auroc=float(roc_auc_score(y,p)), msf=msf(p,loss,ids),
                predicted_success=float(p.mean()), observed_success=float(y.mean()))

def bootstrap_delta(df, p_a, thr_a, p_b, thr_b):
    """Source-article bootstrap for failure(b) minus failure(a)."""
    groups=list(df.groupby('source_pmid',sort=True).indices.values())
    y=df.hit10.to_numpy(); rng=np.random.default_rng(20261007)
    out=[]
    for _ in range(N_BOOT):
        ix=np.concatenate([groups[g] for g in rng.integers(len(groups),size=len(groups))])
        take_a=p_a[ix]>=thr_a; take_b=p_b[ix]>=thr_b
        if take_a.any() and take_b.any():
            out.append(float((1-y[ix][take_b]).mean()-(1-y[ix][take_a]).mean()))
    return [float(z) for z in np.quantile(out,[.025,.975])]

def run(method,view):
    with (CACHE/f'{method}_selectors.pkl').open('rb') as f: bundle=pickle.load(f)
    dev=read('dev',method,view)
    training=dev[dev.partition!='calibration']; cal=dev[dev.partition=='calibration']
    test=read('test',method,view)
    full_model=bundle['model']
    adapted_model=model().fit(training[FEATURES],training.hit10)
    p_full_cal=full_model.predict_proba(cal[FEATURES])[:,1]
    p_full_test=full_model.predict_proba(test[FEATURES])[:,1]
    p_adapt_cal=adapted_model.predict_proba(cal[FEATURES])[:,1]
    p_adapt_test=adapted_model.predict_proba(test[FEATURES])[:,1]
    # Calibrate only on the held-out development calibration partition.
    z=logit(np.clip(p_full_cal,1e-6,1-1e-6)).reshape(-1,1)
    platt=LogisticRegression(C=1e6,max_iter=1000).fit(z,cal.hit10)
    p_platt_test=platt.predict_proba(logit(np.clip(p_full_test,1e-6,1-1e-6)).reshape(-1,1))[:,1]
    static_thr=float(bundle['thresholds']['logistic']['0.5'])
    adapted_thr=float(np.quantile(p_adapt_cal,.5))
    same_model_new_thr=float(np.quantile(p_full_cal,.5))
    y=test.hit10.to_numpy();ids=test.query_id.to_numpy()
    results={
       'original_fixed': assess(y,p_full_test,static_thr,test.source_pmid,ids),
       'original_view_threshold':assess(y,p_full_test,same_model_new_thr,test.source_pmid,ids),
       'view_refit':assess(y,p_adapt_test,adapted_thr,test.source_pmid,ids),
       'platt_probabilities':dict(brier=float(brier_score_loss(y,p_platt_test)),ece10=ece(y,p_platt_test),predicted_success=float(p_platt_test.mean()),observed_success=float(y.mean()))
    }
    results['view_refit_vs_original_at_view_threshold_failure_gain']=results['original_view_threshold']['selected_failure']-results['view_refit']['selected_failure']
    results['view_refit_vs_original_at_view_threshold_failure_gain_bootstrap95']=bootstrap_delta(test,p_adapt_test,adapted_thr,p_full_test,same_model_new_thr)
    result=dict(method=method,view=view,development_fit_n=len(training),development_calibration_n=len(cal),
                test_n=len(test),test_shortened_n=int((test.tokens_used<test.tokens_full).sum()),
                test_mean_fraction_tokens_retained=float((test.tokens_used/test.tokens_full.clip(lower=1)).mean()),
                thresholds=dict(original_fixed=static_thr,original_view_threshold=same_model_new_thr,view_refit=adapted_thr),
                platt_slope=float(platt.coef_[0,0]),platt_intercept=float(platt.intercept_[0]),results=results)
    path=ROOT/f'results/extension_{method}_{view}.json';path.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    all_results=[]
    for method,views in VIEWS.items():
        for view in views:
            r=run(method,view);all_results.append(r)
            print(method,view,r['results'],flush=True)
    (ROOT/'results/extension_summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
    rows=[]
    for r in all_results:
        for name,v in r['results'].items():
            if name in ['original_fixed','original_view_threshold','view_refit']:
                rows.append(dict(method=r['method'],view=r['view'],rule=name,**v))
    pd.DataFrame(rows).to_csv(ROOT/'results/extension_summary.csv',index=False)
