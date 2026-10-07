# Clinical case retrieval experiments

Code for lexical case retrieval, query shortening, and confidence calibration on the public PMC-Patients-ReCDS patient-to-patient benchmark.

## Run

Use Python 3.12 and install the pinned packages in `requirements.txt`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_study.py
.venv/bin/python scripts/run_extension.py
```

The scripts download the public benchmark at a fixed revision with SHA-256 checks, build BM25 and TF-IDF indexes, run eight unit checks, and write outputs locally to ignored `data/`, `results/`, and `artifacts/` directories. The corpus download is approximately 449 MB. Allow several gigabytes of memory for indexing.

The original comparison uses full queries and first-128-token queries. The extension compares additional query views and fits query-form-specific confidence models. **The extension is exploratory:** its rules were developed after the original test results were inspected. The official labels are citation-derived and incomplete; they do not measure clinical safety or usefulness.

The source dataset is [PMC-Patients-ReCDS](https://huggingface.co/datasets/zhengyun21/PMC-Patients-ReCDS). See the dataset card for its terms of use.
