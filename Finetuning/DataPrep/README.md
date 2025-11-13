# Data Preparation for Finetuning

This folder contains code and notebooks used to prepare datasets for finetuning models. It shows examples of cleaning, chunking, prompt templating, and exporting data into JSONL formats ready for supervised finetuning.

Contents
- `Data_prep.ipynb` — Notebook with step-by-step data cleaning and preparation pipelines.
- `scripts/` — Utility scripts used by the notebook (if present).
- `raw/` — Raw source files (CSV/JSON/other).
- `processed/` — Processed outputs such as `*_processed.jsonl` ready for finetuning.

Quick Start
1. Install required packages (run from repo root):

```bash
pip install -r Finetuning/DataPrep/requirements.txt || pip install pandas datasets jsonlines
```

2. Open the notebook `Finetuning/DataPrep/Data_prep.ipynb` in JupyterLab or VS Code and run the cells.

Key Steps Demonstrated
- Loading raw data with `pandas` or `datasets`.
- Cleaning text fields (normalization, whitespace, removing PII where necessary).
- Splitting long documents into semantic chunks with overlap.
- Creating prompt templates (Q/A, instruction/response, or text-only formats).
- Exporting final training files as JSONL for training pipelines.

Example: saving JSONL

```python
import jsonlines
with jsonlines.open('processed/finetune_ready.jsonl', 'w') as writer:
    writer.write_all(prepared_examples)
```

Notes & Best Practices
- Prefer streaming large pretrained corpora with `datasets` to avoid downloading everything locally.
- Remove sensitive information before committing processed datasets to the repository. Use `.gitignore` or Git LFS for large files.
- Keep processed data reproducible by including the exact preprocessing parameters in the notebook.

If you'd like, I can also:
- Add a small `requirements.txt` in this folder if missing.
- Stage and push the `Finetuning/DataPrep/` folder now (I will do that next if you confirm).