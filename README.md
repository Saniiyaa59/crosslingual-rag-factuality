# xling-rag: Enhancing Factuality in Low-Resource Languages via Cross-Lingual RAG

**CS505 NLP Final Project — Boston University**

Saniya Sekhon · Aryan Sharma · Visista Jayanti · Pamela Oliveira

---

## Overview

Large language models perform poorly on factual question answering in low-resource languages like Telugu due to sparse training data and limited retrieval corpora. This project investigates whether **cross-lingual retrieval-augmented generation (RAG)** — retrieving relevant English Wikipedia passages in response to a Telugu query — can reduce hallucinations and improve factual accuracy compared to monolingual RAG and no-retrieval baselines.

The core question: can a multilingual model read English evidence and still produce a correct answer in Telugu?

---

## Research Hypothesis

Cross-lingual RAG (retrieving English passages for a Telugu query) will outperform both monolingual RAG and direct prompting on factual accuracy and hallucination rate, because the English retrieval corpus is substantially richer than the Telugu one. Combining both corpora (Condition D) is expected to further improve coverage.

---

## Pipeline

### Condition C — Cross-lingual RAG

```
Question (Telugu)
      ↓
[NLLB-200-distilled-600M] → English translation
      ↓
combined_query = english_translation + " " + telugu_question
      ↓
[BGE-M3 encoder] → multilingual vector
      ↓
FAISS index search → top-5 English Wikipedia passages
      ↓
Prompt: Telugu question + English passages
      ↓
[Qwen2.5-7B-Instruct]
      ↓
Answer (Telugu)
```

### Condition D — Combined RAG (English + Telugu indexes)

```
Question (Telugu)
      ↓
[NLLB] → English translation → combined_query
      │
      ├── [BGE-M3] → top-3 English Wikipedia passages
      └── [BGE-M3] → top-3 Telugu Wikipedia passages
                              ↓
                  Prompt: Telugu question + 6 passages
                              ↓
                  [Qwen2.5-7B-Instruct]
                              ↓
                       Answer (Telugu)
```

---

## Experimental Conditions

| Condition | Description | Script | Status |
|-----------|-------------|--------|--------|
| **A — No retrieval** | Direct prompting with the Telugu question only | `baseline.py` | ✅ Done |
| **B — Monolingual RAG** | Retrieve from Telugu Wikipedia; generate with Qwen | `rag_monolingual.py` | ✅ Done |
| **C — Cross-lingual RAG** | Translate query with NLLB, retrieve from English Wikipedia, generate with Qwen | `rag.py` | ✅ Done |
| **D — Combined RAG** | Retrieve from both English and Telugu Wikipedia indexes | `rag_combined.py` | ✅ Done |

---

## Dataset

**TyDi QA** (Clark et al., 2020) — a multilingual QA dataset built from Wikipedia across 11 typologically diverse languages.

- `question` + `answers` from the **Telugu split** → test queries and gold labels
- English Wikipedia (50k articles, chunked) → cross-lingual retrieval corpus (Condition C, D)
- Telugu Wikipedia (20k articles, chunked) → monolingual retrieval corpus (Condition B, D)

---

## Models

| Role | Model |
|------|-------|
| Multilingual encoder | `BAAI/bge-m3` |
| Query translator | `facebook/nllb-200-distilled-600M` |
| Generator | `Qwen/Qwen2.5-7B-Instruct` |

> **Note:** Qwen2.5-7B requires ~14GB GPU memory. Use a 40GB+ GPU (A100/A40) on SCC. On a 16GB V100, switch to `Qwen/Qwen2.5-3B-Instruct` in each script.

---

## Evaluation Metrics

- **Exact Match (EM)** — predicted string exactly matches gold answer after normalization
- **Character 3-gram Recall (Chr-3)** — language-agnostic overlap metric suited to morphologically rich languages
- **Response Language Correctness (RLC)** — fraction of responses generated in Telugu (detects English fallback)
- **LLM-as-a-Judge** — GPT-4o + Claude Sonnet majority vote for semantic accuracy (planned)

---

## Project Structure

```
crosslingual-rag-factuality/
├── data/
│   ├── tydiqa_telugu_train/          # Cached Telugu TyDi QA split
│   ├── tydiqa_english_train/         # Cached English TyDi QA split
│   ├── results_baseline.json         # Condition A outputs
│   ├── results_monolingual.json      # Condition B outputs
│   ├── results_crosslingual_wiki.json # Condition C outputs
│   ├── results_combined.json         # Condition D outputs
│   └── index/
│       ├── wiki_bge.faiss            # FAISS index over English Wikipedia chunks
│       ├── wiki_passages.npy         # Aligned English passage texts
│       ├── telugu_wiki_bge.faiss     # FAISS index over Telugu Wikipedia chunks
│       └── telugu_wiki_passages.npy  # Aligned Telugu passage texts
├── scripts/
│   ├── build_wiki_index.py           # Build English Wikipedia FAISS index
│   ├── build_telugu_index.py         # Build Telugu Wikipedia FAISS index
│   ├── baseline.py                   # Condition A — no retrieval
│   ├── rag.py                        # Condition C — cross-lingual RAG
│   ├── rag_monolingual.py            # Condition B — monolingual Telugu RAG
│   ├── rag_combined.py               # Condition D — combined EN + TE RAG
│   ├── retrieval_utils.py            # Shared load_index / retrieve helpers
│   ├── evaluate.py                   # Compute EM, Chr-3, RLC across conditions
│   ├── index.py                      # Build TyDi QA English FAISS index
│   └── data.ipynb                    # Exploratory notebook
├── report/
│   ├── midway_report.tex
│   └── references.bib
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
from datasets import load_dataset

tydi = load_dataset("tydiqa", "secondary_task")
telugu = tydi["train"].filter(lambda x: x["id"].startswith("telugu-"))
english = tydi["train"].filter(lambda x: x["id"].startswith("english-"))

telugu.save_to_disk("data/tydiqa_telugu_train")
english.save_to_disk("data/tydiqa_english_train")
```

### Step 2 — Build the English Wikipedia FAISS index (SCC GPU node)

```bash
python scripts/build_wiki_index.py
```

Indexes 50k English Wikipedia articles (200-word chunks, 50-word overlap) with `BAAI/bge-m3`. Saves to `data/index/wiki_bge.faiss` and `data/index/wiki_passages.npy`.

> Takes ~30 minutes on a V100 GPU.

### Step 3 — Build the Telugu Wikipedia FAISS index (SCC GPU node)

```bash
python scripts/build_telugu_index.py
```

Indexes 20k Telugu Wikipedia articles with the same chunking and encoder. Saves to `data/index/telugu_wiki_bge.faiss` and `data/index/telugu_wiki_passages.npy`.

> Takes ~15 minutes on a V100 GPU.

### Step 4 — Run all conditions

```bash
# Condition A — no retrieval baseline
python scripts/baseline.py

# Condition B — monolingual Telugu RAG
python scripts/rag_monolingual.py

# Condition C — cross-lingual English RAG (with NLLB query translation)
python scripts/rag.py

# Condition D — combined English + Telugu RAG
python scripts/rag_combined.py
```

All scripts use a **two-phase GPU memory strategy**: Phase 1 runs retrieval (BGE-M3 + NLLB where applicable) and frees GPU memory before Phase 2 loads Qwen for generation. This fits within a 16GB V100 for the 3B model; use a 40GB+ GPU for the 7B model.

### Step 5 — Evaluate

```bash
python scripts/evaluate.py
```

Computes EM, Chr-3, and RLC across all result files.

---

## Key Findings (10-example dev set)

| Condition | EM | RLC | Notes |
|---|---|---|---|
| A — No retrieval | 0/10 | ~0% | Hallucination, English fallback |
| B — Monolingual TE | 1/10 | ~90% | Best for village census questions |
| C — Cross-lingual EN | 0/10 | ~80% | English index missing India-specific articles |
| D — Combined EN+TE | 0/10 | ~85% | Context noise hurt vs. B alone |

Primary failure modes: (1) index coverage — English Wikipedia's first 50k articles skew Western; (2) NLLB mistranslates named entities (e.g., *వేప* → "beech" instead of "neem"); (3) LLM extraction errors even when relevant passage is retrieved.

---

## Notes

- All scripts must be run from the project root (`crosslingual-rag-factuality/`), not from inside `scripts/`
- `model.max_seq_length = 512` is set explicitly for `bge-m3` to avoid truncation issues
- Wikipedia chunks are aligned with the FAISS index by insertion order — do not shuffle `*_passages.npy` independently of the index
- The English index uses `range(50000)` which selects the first 50k articles by ID (earliest-created, skewing Western topics); random sampling would improve coverage for India-specific queries

---

## References

- Clark et al. (2020). TyDi QA. *TACL*.
- Chen et al. (2024). BGE-M3. *arXiv:2402.03216*.
- Costa-jussà et al. (2022). No Language Left Behind. *arXiv:2207.04672*.
- Ranaldi et al. (2025). CrossRAG. *arXiv:2504.03616*.
- Moon et al. (2025). Quality-Aware Translation Tagging in Multilingual RAG. *MRL 2025*.
- Chirkova et al. (2024). RAG in Multilingual Settings. *KnowLLM 2024*.
