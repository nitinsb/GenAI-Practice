# Finetuning: whereItFits

This folder contains educational materials and a hands-on notebook showing where finetuning fits relative to pretraining and how to prepare instruction-style datasets for supervised finetuning.

Notebooks
- `02_Where_finetuning_fits_in_lab_student.ipynb` — Demonstrates dataset contrasts, prompt templates, dataset formatting, and saving to JSONL for finetuning.

Overview
- Purpose: show difference between large-scale pretraining corpora (e.g., C4) and small, task-specific finetuning datasets. Provide practical steps for formatting and saving instruction/response pairs.
- Input / Output: Reads `lamini_docs.jsonl`, demonstrates prompt templates, builds `lamini_docs_processed.jsonl` for finetuning.

Quick Start
1. Ensure required Python packages installed (e.g., `datasets`, `pandas`, `jsonlines`).

```bash
# from repo root
pip install -r Finetuning/whereItFits/requirements.txt  # if provided
# or
pip install datasets pandas jsonlines
```

2. Run the notebook `02_Where_finetuning_fits_in_lab_student.ipynb` in JupyterLab or VS Code.

Key Code Snippets
- Loading the pretrained dataset (example uses C4):

```python
from datasets import load_dataset
pretrained_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
```

- Reading the company finetuning dataset (example `lamini_docs.jsonl`):

```python
import pandas as pd
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
```

- Creating prompt/response pairs and saving to JSONL for finetuning:

```python
import jsonlines

# Build question/answer formatted dataset
prompt_template_qa = """### Question:\n{question}\n\n### Answer:\n{answer}"""

examples = instruction_dataset_df.to_dict()
finetuning_dataset_question_answer = []
for i in range(len(examples.get("question", []))):
    question = examples["question"][i]
    answer = examples["answer"][i]
    text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
    finetuning_dataset_question_answer.append({"question": text_with_prompt_template_qa, "answer": answer})

with jsonlines.open('lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(finetuning_dataset_question_answer)
```

Note: The notebook contains a short code line that loads `lamini/lamini_docs` from the Hugging Face Hub. The correct code (no stray quotes) is:

```python
finetuning_dataset_name = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_name)
print(finetuning_dataset)
```

Troubleshooting
- If `load_dataset` fails due to network or authentication, ensure you have Internet access and (if necessary) HF token configured in `HUGGINGFACE_HUB_TOKEN`.
- For large pretrained datasets (like C4), prefer streaming mode to avoid downloading huge data.

Learning Outcomes
- Understand how finetuning datasets differ from pretraining corpora.
- Learn common prompt templates for instruction/response style finetuning.
- Produce a JSONL file suitable for supervised finetuning pipelines.

If you want, I can also sanitize the notebook cell that contains the mis-escaped strings and save a corrected notebook file; tell me if you'd like me to fix that cell in-place.