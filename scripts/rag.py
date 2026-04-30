from datasets import load_from_disk
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import json

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

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    telugu = load_from_disk("data/tydiqa_telugu_train")
    sample = telugu.select(range(10))

    # Load query translator (Telugu → English, used before retrieval)
    translator = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang="tel_Telu",
        tgt_lang="eng_Latn",
        device=0 if device == "cuda" else -1,
        max_length=256,
    )

    # Load retriever
    retriever = SentenceTransformer("BAAI/bge-m3", device=device)
    retriever.max_seq_length = 512

    # Load index
    index, passages = load_index(
        "data/index/wiki_bge.faiss",
        "data/index/wiki_passages.npy"
    )

    # Load Qwen generator (no license gate, strong multilingual)
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
    )

    results = []

    for row in sample:
        question = row["question"]
        gold = row["answers"]["text"][0]

        # Translate Telugu query to English
        english_query = translator(question)[0]["translation_text"]

        # Combined query preserves named entities that NLLB mistranslates
        combined_query = english_query + " " + question
        retrieved = retrieve(combined_query, retriever, index, passages, k=5)
        context = "\n\n".join(retrieved)

        # Llama generates the answer in Telugu from English context
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
        out = generator(messages, max_new_tokens=128, do_sample=False)
        prediction = out[0]["generated_text"][-1]["content"].strip()

        results.append({
            "question": question,
            "english_query": english_query,
            "gold": gold,
            "prediction": prediction,
            "retrieved_passages": retrieved
        })
        print(f"Q: {question}")
        print(f"EN: {english_query}")
        print(f"Gold: {gold}")
        print(f"Pred: {prediction}")
        print(f"Retrieved: {retrieved[0][:200]}...")
        print("---")

    with open("data/results_crosslingual_wiki.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
