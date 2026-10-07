"""Download pinned public benchmark files, atomically, without credentials."""
from pathlib import Path
import hashlib, json, urllib.request, concurrent.futures, datetime
ROOT=Path(__file__).resolve().parents[1]
REPO='zhengyun21/PMC-Patients-ReCDS'
REV='a27717bb27679cf0860305997685547ca01b3dd1'
FILES=['PPR/corpus.jsonl','PPR/qrels_dev.tsv','PPR/qrels_test.tsv','queries/dev_queries.jsonl','queries/test_queries.jsonl','README.md']
EXPECTED_SHA256={'PPR/corpus.jsonl': '33c2f31a02816f20de1cadae2dba4ab97518398519aeb1e46ebcb6dca5038058', 'PPR/qrels_dev.tsv': '50f4cef71b3619a81053943af5870f56120a46d1e97d7f1efa831f6fc9099575', 'PPR/qrels_test.tsv': '19c00c9daed954c8a9da9d836b888ab5f7bd55383c99e2255cd654ff5020a54d', 'queries/dev_queries.jsonl': 'e3afe1b5f6832a4c691267453153aaec6af5fd7cdbef567be484a7a1fb48d48b', 'queries/test_queries.jsonl': '0501920965ef79f53c6efb30d1f8f7b7ba6ff3564da1913db5a515d365cef8b8', 'README.md': 'a6f32da3c4453172f753b09d5cbf24308aecc86e9f2162f704bac2f78379a090'}
def fetch(rel):
    dest=ROOT/'data/raw'/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{rel}?download=true'
    if not dest.exists():
        temp=dest.with_suffix(dest.suffix+'.part')
        with urllib.request.urlopen(url, timeout=120) as source, temp.open('wb') as sink:
            while chunk:=source.read(1024*1024): sink.write(chunk)
        temp.replace(dest)
    sha=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
    if sha != EXPECTED_SHA256[rel]:
        raise ValueError(f'Checksum mismatch for {rel}; expected pinned source bytes')
    record=dict(path=str(dest.relative_to(ROOT)),url=url,sha256=sha,bytes=dest.stat().st_size)
    print(json.dumps(record),flush=True)
    return record
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: records=list(pool.map(fetch,FILES))
    (ROOT/'data/manifest.json').write_text(json.dumps(dict(repository=REPO,revision=REV,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records),indent=2)+'\n')
