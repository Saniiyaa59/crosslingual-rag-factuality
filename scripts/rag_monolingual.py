from datasets import load_from_disk
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from retrieval_utils import load_index, retrieve
import torch
import json
import gc

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    telugu = load_from_disk("data/tydiqa_telugu_train")
    sample = telugu.select(range(10))

    # ── Phase 1: retrieve from Telugu Wikipedia (BGE-M3) ───────────────────
    print("Phase 1: retrieving from Telugu Wikipedia index...")

    retriever = SentenceTransformer("BAAI/bge-m3", device=device)
    retriever.max_seq_length = 512

    index, passages = load_index(
        "data/index/telugu_wiki_bge.faiss",
        "data/index/telugu_wiki_passages.npy"
    )
    print(f"Index loaded: {index.ntotal} passages")

    rows = []
    for row in sample:
        question = row["question"]
        gold = row["answers"]["text"][0]
        retrieved = retrieve(question, retriever, index, passages, k=5)
        rows.append({
            "question": question,
            "gold": gold,
            "retrieved_passages": retrieved,
        })
        print(f"  retrieved for: {question[:60]}...")

    # Free BGE-M3 from GPU before loading Qwen
    del retriever
    gc.collect()
    torch.cuda.empty_cache()
    print("Phase 1 done. GPU memory freed.")

    # ── Phase 2: generate with Qwen ────────────────────────────────────────
    print("Phase 2: loading Qwen and generating...")

    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
    )

    results = []
    for row in rows:
        question = row["question"]
        gold = row["gold"]
        retrieved = row["retrieved_passages"]
        context = "\n\n".join(retrieved)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multilingual QA assistant. "
                    "Use the provided context to answer the question. "
                    "Always answer in the same language as the question. "
                    "Be concise — one sentence or less."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer in Telugu using only information from the context above."
                )
            }
        ]
        out = generator(messages, max_new_tokens=128, do_sample=False,
                        temperature=None, top_p=None, top_k=None)
        prediction = out[0]["generated_text"][-1]["content"].strip()

        results.append({
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "retrieved_passages": retrieved,
        })
        print(f"Q: {question}")
        print(f"Gold: {gold}")
        print(f"Pred: {prediction}")
        print(f"Retrieved: {retrieved[0][:200]}...")
        print("---")

    with open("data/results_monolingual.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(results)} results saved to data/results_monolingual.json")