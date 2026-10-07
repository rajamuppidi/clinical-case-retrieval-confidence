"""Exact lexical retrieval and outcome-blind confidence features."""
from pathlib import Path
import argparse, csv, hashlib, json, pickle, re, time, resource
from collections import Counter, defaultdict
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/cache'
TOKEN = re.compile(r'(?u)\b\w\w+\b')
FEATURES = ['top_score', 'relative_gap', 'score_per_term', 'score_cv100',
            'nqc_corpus_mean', 'mean_idf', 'log_tokens', 'oov_fraction',
            'unique_fraction', 'top10_top1_ratio']

def tokens(text):
    return TOKEN.findall(text.lower())

def view_tokens(term_list, view, idf=None, vocab=None):
    if view == 'full':
        return term_list
    if view == 'prefix64':
        return term_list[:64]
    if view == 'prefix128':
        return term_list[:128]
    if view == 'prefix256':
        return term_list[:256]
    if view == 'headtail128':
        return term_list if len(term_list) <= 128 else term_list[:64] + term_list[-64:]
    if view == 'highidf128':
        if len(term_list) <= 128:
            return term_list
        if idf is None or vocab is None:
            raise ValueError('highidf128 requires corpus IDF and vocabulary')
        chosen=sorted(sorted(range(len(term_list)),key=lambda i:(-float(idf[vocab[term_list[i]]]) if term_list[i] in vocab else 0.,i))[:128])
        return [term_list[i] for i in chosen]
    raise ValueError(view)

def pmid(uid):
    return uid.rsplit('-', 1)[0]

def partition(uid):
    bucket = int(hashlib.sha256(('clinical-confidence-v1:' + pmid(uid)).encode()).hexdigest()[:8], 16) % 10
    return 'fit' if bucket < 6 else 'validation' if bucket < 8 else 'calibration'

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def load_qrels(split):
    qrels = defaultdict(dict)
    with open(ROOT / f'data/raw/PPR/qrels_{split}.tsv') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            qrels[str(row['query-id'])][str(row['corpus-id'])] = int(row['score'])
    return dict(qrels)

def normalize_text(text):
    return ' '.join(text.lower().split())

def audit():
    corpus = load_jsonl(ROOT / 'data/raw/PPR/corpus.jsonl')
    sets = {'corpus': corpus, **{s: load_jsonl(ROOT / f'data/raw/queries/{s}_queries.jsonl') for s in ['dev', 'test']}}
    ids = {s: {str(x['_id']) for x in rows} for s, rows in sets.items()}
    groups = {s: {pmid(x) for x in values} for s, values in ids.items()}
    hashes = {s: {hashlib.sha256(normalize_text(x['text']).encode()).hexdigest() for x in rows} for s, rows in sets.items()}
    report = {'counts': {s: {'rows': len(rows), 'unique_ids': len(ids[s]), 'source_articles': len(groups[s]),
                           'unique_normalized_texts': len(hashes[s])} for s, rows in sets.items()}, 'overlaps': {}, 'qrels': {}}
    for a,b in [('corpus','dev'),('corpus','test'),('dev','test')]:
        report['overlaps'][a+'_'+b] = {'ids':len(ids[a]&ids[b]),'source_articles':len(groups[a]&groups[b]),
                                      'normalized_texts':len(hashes[a]&hashes[b])}
        assert not (ids[a]&ids[b]) and not (groups[a]&groups[b]), f'Split identity overlap: {a},{b}'
    for s in ['dev','test']:
        qrels=load_qrels(s)
        assert set(qrels)<=ids[s]
        assert all(set(r)<=ids['corpus'] for r in qrels.values())
        report['qrels'][s]={'eligible_queries':len(qrels),'excluded_no_judgments':len(sets[s])-len(qrels),
                            'eligible_articles':len({pmid(q) for q in qrels}),
                            'pairs':sum(map(len,qrels.values())),
                            'grades':sorted({v for rs in qrels.values() for v in rs.values()}),
                            'partitions':dict(Counter(partition(q) for q in qrels)) if s=='dev' else None}
    assert all(len(rows)==len(ids[s]) for s,rows in sets.items())
    (ROOT/'results/data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report

def build_index():
    start=time.perf_counter()
    CACHE.mkdir(parents=True,exist_ok=True)
    corpus=load_jsonl(ROOT/'data/raw/PPR/corpus.jsonl')
    vectorizer=CountVectorizer(lowercase=True,token_pattern=TOKEN.pattern,dtype=np.float32)
    counts=vectorizer.fit_transform(x['text'] for x in corpus).tocsr()
    ids=[str(x['_id']) for x in corpus]
    del corpus
    lengths=np.asarray(counts.sum(axis=1)).ravel()
    n=counts.shape[0]
    df=np.asarray((counts>0).sum(axis=0)).ravel()
    idf=np.log1p((n-df+0.5)/(df+0.5)).astype(np.float32)
    bm=counts.copy()
    # Process rows in bounded blocks to avoid multiple full-size expanded arrays.
    for i in range(0,n,2048):
        end=min(i+2048,n); a,b=bm.indptr[i],bm.indptr[end]
        norm=np.repeat(1.2*(0.25+0.75*lengths[i:end]/lengths.mean()),np.diff(bm.indptr[i:end+1]))
        bm.data[a:b]=bm.data[a:b]*2.2/(bm.data[a:b]+norm)*idf[bm.indices[a:b]]
    transformer=TfidfTransformer(norm='l2',use_idf=True,smooth_idf=True,sublinear_tf=True)
    tfidf=transformer.fit_transform(counts).astype(np.float32)
    sparse.save_npz(CACHE/'bm25.npz',bm,compressed=False)
    sparse.save_npz(CACHE/'tfidf.npz',tfidf,compressed=False)
    with (CACHE/'index.pkl').open('wb') as f:
        pickle.dump(dict(vectorizer=vectorizer,transformer=transformer,ids=ids,idf=idf),f)
    report=dict(documents=n,vocabulary=counts.shape[1],nonzero_counts=counts.nnz,
                seconds=time.perf_counter()-start,max_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (ROOT/'results/index_resources.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

def top_indices(scores,k=1000):
    """Deterministic score-descending, index-ascending ranking including cutoff ties."""
    k=min(k,len(scores))
    cutoff=np.partition(scores,len(scores)-k)[len(scores)-k]
    greater=np.flatnonzero(scores>cutoff)
    equal=np.flatnonzero(scores==cutoff)[:k-len(greater)]
    ix=np.concatenate([greater,equal])
    return ix[np.lexsort((ix,-scores[ix]))]

def metrics(ranked_ids,rels):
    hits=np.array([rels.get(uid,0)>0 for uid in ranked_ids],dtype=float)
    npos=sum(v>0 for v in rels.values())
    pos=np.flatnonzero(hits)
    ideal=np.sum(1/np.log2(np.arange(min(npos,10))+2))
    return dict(hit10=float(hits[:10].sum()>0),ndcg10=float(np.sum(hits[:10]/np.log2(np.arange(min(len(hits),10))+2))/ideal),
                precision10=float(hits[:10].sum()/10),mrr1000=float(1/(pos[0]+1)) if len(pos) else 0.,
                recall1000=float(hits.sum()/npos))

def retrieve(split,method,view,batch_size=16):
    start=time.perf_counter()
    with (CACHE/'index.pkl').open('rb') as f: index=pickle.load(f)
    matrix=sparse.load_npz(CACHE/f'{method}.npz').tocsc()
    vectorizer=index['vectorizer']; vocab=vectorizer.vocabulary_
    qrels=load_qrels(split)
    rows=sorted((x for x in load_jsonl(ROOT/f'data/raw/queries/{split}_queries.jsonl') if str(x['_id']) in qrels),key=lambda x:str(x['_id']))
    outpath=ROOT/f'results/{split}_{method}_{view}.csv'
    result_rows=[]
    with outpath.open('w',newline='') as f:
        writer=None
        for offset in range(0,len(rows),batch_size):
            batch=rows[offset:offset+batch_size]
            original_tokens=[tokens(x['text']) for x in batch]
            tt=[view_tokens(t,view,index['idf'],vocab) for t in original_tokens]
            counts=vectorizer.transform([' '.join(t) for t in tt])
            if method=='bm25':
                qq=counts.copy(); qq.data[:]=1.
            else:
                qq=index['transformer'].transform(counts)
            score_batch=(matrix@qq.T).toarray()
            for j,x in enumerate(batch):
                scores=score_batch[:,j]; ix=top_indices(scores); top=scores[ix]
                qid=str(x['_id']); terms=tt[j]
                invocab=[vocab[t] for t in set(terms) if t in vocab]
                unique=len(invocab); eps=1e-12
                feats=dict(top_score=float(top[0]),relative_gap=float((top[0]-top[1])/max(float(top[0]),eps)),
                           score_per_term=float(top[0]/max(unique,1)),score_cv100=float(np.std(top[:100])/max(float(np.mean(top[:100])),eps)),
                           nqc_corpus_mean=float(np.std(top[:100])/max(float(np.mean(scores)),eps)),
                           mean_idf=float(np.mean(index['idf'][invocab])) if invocab else 0.,
                           log_tokens=float(np.log1p(len(terms))),oov_fraction=sum(t not in vocab for t in terms)/max(len(terms),1),
                           unique_fraction=len(set(terms))/max(len(terms),1),top10_top1_ratio=float(np.mean(top[:10])/max(float(top[0]),eps)))
                row=dict(query_id=qid,source_pmid=pmid(qid),partition=partition(qid) if split=='dev' else 'test',
                         tokens_full=len(original_tokens[j]),tokens_used=len(terms),qrel_count=len(qrels[qid]),
                         **feats,**metrics([index['ids'][i] for i in ix],qrels[qid]),
                         top10_ids='|'.join(index['ids'][i] for i in ix[:10]))
                if writer is None: writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
            if offset%256==0:
                f.flush(); print(f'{split} {method} {view}: {min(offset+batch_size,len(rows))}/{len(rows)} ({time.perf_counter()-start:.1f}s)',flush=True)
    runtime=dict(split=split,method=method,view=view,queries=len(rows),seconds=time.perf_counter()-start,
                 max_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (ROOT/f'results/{split}_{method}_{view}_resources.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print(runtime,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['audit','index','retrieve'])
    p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--method',choices=['bm25','tfidf'],default='bm25')
    p.add_argument('--view',choices=['full','prefix64','prefix128','prefix256','headtail128','highidf128'],default='full');args=p.parse_args()
    if args.action=='audit':audit()
    elif args.action=='index':build_index()
    else:retrieve(args.split,args.method,args.view)
