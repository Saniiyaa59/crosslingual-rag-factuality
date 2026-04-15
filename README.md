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
FAISS index search → top-k English passages (untranslated)
      ↓
Prompt: Telugu question + raw English passages
      ↓
[mT0-base / BLOOMZ-560m generator]
      ↓
Answer (Telugu)
```

---

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **A — No retrieval** | Direct prompting with the Telugu question only |
| **B — Monolingual RAG** | Retrieve from Telugu Wikipedia passages |
| **C — Cross-lingual RAG** | Retrieve from English Wikipedia passages (untranslated) |
| **D — Cross-lingual RAG + Translation** | Same as C, but passages translated to Telugu via NLLB-200-600M (ablation) |

---

## Dataset

**TyDi QA** (Clark et al., 2020) — a multilingual QA dataset built from Wikipedia across 11 typologically diverse languages.

- `question` + `answers` from the **Telugu split** → test queries and gold labels
- `context` from the **English split** → retrieval corpus (FAISS index)

---

## Models

| Role | Model |
|------|-------|
| Multilingual encoder | `BAAI/bge-m3` |
| Generator | `bigscience/mt0-base` or `bigscience/bloomz-560m` |
| Translation (ablation only) | `facebook/nllb-200-distilled-600M` |

---

## Evaluation Metrics

- **Character 3-gram Recall** — language-agnostic overlap metric suited to morphologically rich languages
- **Response Language Correctness (RLC)** — fraction of responses generated in Telugu (detects English fallback)
- **LLM-as-a-Judge** — GPT-4o + Claude Sonnet majority vote for semantic accuracy
- **Hallucination Rate** — manual annotation of outputs for unsupported factual claims

---

## Setup

```bash
# Clone the repo
git clone https://github.com/your-team/xling-rag.git
cd xling-rag

# Create and activate virtual environment (Python 3.10+ required)
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
crosslingual-rag-factuality/
├── data/               # Cached TyDi QA splits and FAISS index
├── scripts/
│   ├── index.py        # Build English FAISS index (bge-m3)
│   ├── build_wiki_index.py  # Build Wikipedia FAISS index
│   ├── baseline.py     # Condition A — no retrieval
│   ├── rag_crosslingual.py  # Condition C — cross-lingual RAG
│   └── data.ipynb      # Exploratory notebook
└── README.md
```

---

## References

- Clark et al. (2020). TyDi QA. *TACL*.
- Chen et al. (2024). BGE-M3. *arXiv:2309.07597*.
- Costa-jussà et al. (2022). No Language Left Behind. *arXiv:2207.04672*.
- Ranaldi et al. (2025). CrossRAG. *arXiv:2504.03616*.
- Moon et al. (2025). Quality-Aware Translation Tagging in Multilingual RAG. *MRL 2025*.