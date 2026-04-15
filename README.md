# xling-rag: Enhancing Factuality in Low-Resource Languages via Cross-Lingual RAG

**CS505 NLP Final Project — Boston University**

Aryan Sharma · Saniya Sekhon · Visista Jayanti · Pamela Oliveira

---

## Overview

Large language models perform poorly on factual question answering in low-resource languages like Telugu due to sparse training data and limited retrieval corpora. This project investigates whether **cross-lingual retrieval-augmented generation (RAG)** — retrieving relevant English Wikipedia passages in response to a Telugu query — can reduce hallucinations and improve factual accuracy compared to monolingual RAG and no-retrieval baselines.

The core question: can a multilingual model read English evidence and still produce a correct answer in Telugu?

---

## Research Hypothesis

Cross-lingual RAG (retrieving English passages for a Telugu query) will outperform both monolingual RAG and direct prompting on factual accuracy and hallucination rate, because the English retrieval corpus is substantially richer than the Telugu one.

---

## Pipeline

```
Question (Telugu)
      ↓
[BGE-M3 encoder] → multilingual vector
      ↓
FAISS index search → top-k English Wikipedia passages
      ↓
Prompt: Telugu question + raw English passages
      ↓
[mT0-base generator]
      ↓
Answer (Telugu)
```

---

## Experimental Conditions

| Condition | Description | Status |
|-----------|-------------|--------|
| **A — No retrieval** | Direct prompting with the Telugu question only | ✅ Done |
| **B — Monolingual RAG** | Retrieve from Telugu Wikipedia passages | 🔲 In progress |
| **C — Cross-lingual RAG** | Retrieve from English Wikipedia passages (untranslated) | ✅ Done |
| **D — Cross-lingual RAG + Translation** | Same as C, but passages translated to Telugu via NLLB-200-600M | 🔲 Planned |

---

## Dataset

**TyDi QA** (Clark et al., 2020) — a multilingual QA dataset built from Wikipedia across 11 typologically diverse languages.

- `question` + `answers` from the **Telugu split** → test queries and gold labels
- English Wikipedia (50k articles, chunked) → retrieval corpus indexed with FAISS

---

## Models

| Role | Model |
|------|-------|
| Multilingual encoder | `BAAI/bge-m3` |
| Generator | `bigscience/mt0-base` |
| Translation (ablation only) | `facebook/nllb-200-distilled-600M` |

---

## Evaluation Metrics

- **Character 3-gram Recall** — language-agnostic overlap metric suited to morphologically rich languages
- **Response Language Correctness (RLC)** — fraction of responses generated in Telugu (detects English fallback)
- **LLM-as-a-Judge** — GPT-4o + Claude Sonnet majority vote for semantic accuracy
- **Hallucination Rate** — manual annotation of outputs for unsupported factual claims

---

## Project Structure

```
crosslingual-rag-factuality/
├── data/
│   ├── tydiqa_telugu_train/     # Cached Telugu TyDi QA split
│   ├── tydiqa_english_train/    # Cached English TyDi QA split
│   └── index/
│       ├── wiki_bge.faiss       # FAISS index over Wikipedia chunks
│       └── wiki_passages.npy    # Aligned passage texts
├── scripts/
│   ├── index.py                 # Build TyDi QA English FAISS index
│   ├── build_wiki_index.py      # Build Wikipedia FAISS index (run on SCC)
│   ├── baseline.py              # Condition A — no retrieval
│   ├── rag_crosslingual.py      # Condition C — cross-lingual RAG
│   └── data.ipynb               # Exploratory notebook
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/your-team/xling-rag.git
cd crosslingual-rag-factuality

python3.10 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> **On SCC:** Set the HuggingFace cache to your project directory to avoid filling your home quota:
> ```bash
> export HF_HOME=/projectnb/cs505am/students/<your_username>/.cache/huggingface
> ```

---

## Running the Pipeline

### Step 1 — Download and cache TyDi QA

```python
from datasets import load_dataset, load_from_disk

tydi = load_dataset("tydiqa", "secondary_task")
telugu = tydi["train"].filter(lambda x: x["id"].startswith("telugu-"))
english = tydi["train"].filter(lambda x: x["id"].startswith("english-"))

telugu.save_to_disk("data/tydiqa_telugu_train")
english.save_to_disk("data/tydiqa_english_train")
```

### Step 2 — Build the Wikipedia FAISS index (run on SCC GPU node)

```bash
python scripts/build_wiki_index.py
```

This downloads 50k English Wikipedia articles, chunks them into 200-word passages with 50-word overlap, encodes them with `BAAI/bge-m3`, and saves the FAISS index to `data/index/`.

> Takes ~30 minutes on a V100 GPU.

### Step 3 — Run Condition A (no retrieval baseline)

```bash
python scripts/baseline.py
```

Outputs predictions to stdout. Results saved to `data/results_baseline.json`.

### Step 4 — Run Condition C (cross-lingual RAG)

```bash
python scripts/rag_crosslingual.py
```

Retrieves top-3 English Wikipedia passages per Telugu query using FAISS, then generates answers with `mt0-base`. Results saved to `data/results_crosslingual_wiki.json`.

---

## Notes

- All scripts must be run from the project root (`crosslingual-rag-factuality/`), not from inside `scripts/`
- `model.max_seq_length = 512` is set explicitly for `bge-m3` to avoid OOM on Apple Silicon
- Wikipedia chunks are aligned with the FAISS index by insertion order — do not shuffle `wiki_passages.npy` independently of the index

---

## References

- Clark et al. (2020). TyDi QA. *TACL*.
- Chen et al. (2024). BGE-M3. *arXiv:2309.07597*.
- Costa-jussà et al. (2022). No Language Left Behind. *arXiv:2207.04672*.
- Ranaldi et al. (2025). CrossRAG. *arXiv:2504.03616*.
- Moon et al. (2025). Quality-Aware Translation Tagging in Multilingual RAG. *MRL 2025*.
```