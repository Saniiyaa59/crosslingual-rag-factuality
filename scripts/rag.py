from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import json
import gc

def load_index(index_path, passages_path):
    index = faiss.read_index(index_path)
    passages = np.load(passages_path, allow_pickle=True).tolist()
    return index, passages

def retrieve(query, retriever, index, passages, k=5):
    query_embedding = retriever.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)
    scores, indices = index.search(query_embedding, k)
    return [passages[i] for i in indices[0]]

def translate(text, model, tokenizer, device, src_lang="tel_Telu", tgt_lang="eng_Latn"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    output = model.generate(**inputs, forced_bos_token_id=forced_bos, max_new_tokens=256)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    telugu = load_from_disk("data/tydiqa_telugu_train")
    sample = telugu.select(range(500))

    # ── Phase 1: translate + retrieve (NLLB + BGE-M3) ──────────────────────
    print("Phase 1: translating and retrieving...")

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)

    retriever = SentenceTransformer("BAAI/bge-m3", device=device)
    retriever.max_seq_length = 512

    index, passages = load_index(
        "data/index/wiki_bge.faiss",
        "data/index/wiki_passages.npy"
    )

    rows = []
    for row in sample:
        question = row["question"]
        gold = row["answers"]["text"][0]
        english_query = translate(question, translator_model, tokenizer, device)
        combined_query = english_query + " " + question
        retrieved = retrieve(combined_query, retriever, index, passages, k=5)
        rows.append({
            "question": question,
            "english_query": english_query,
            "gold": gold,
            "retrieved_passages": retrieved,
        })
        print(f"  translated: {english_query}")

    # Free NLLB + BGE-M3 from GPU before loading Qwen
    del translator_model, tokenizer, retriever
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
        context = "\n\n".join(row["retrieved_passages"])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multilingual QA assistant. "
                    "Use the provided context to answer the question. "
                    "Always answer in the same language as the question, "
                    "even if the context is in a different language. "
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
            "english_query": row["english_query"],
            "gold": row["gold"],
            "prediction": prediction,
            "retrieved_passages": row["retrieved_passages"],
        })
        print(f"Q: {question}")
        print(f"EN: {row['english_query']}")
        print(f"Gold: {row['gold']}")
        print(f"Pred: {prediction}")
        print(f"Retrieved: {row['retrieved_passages'][0][:200]}...")
        print("---")

    with open("data/results_crosslingual_wiki.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)