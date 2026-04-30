from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import json

def load_index(index_path, passages_path):
    index = faiss.read_index(index_path)
    passages = np.load(passages_path, allow_pickle=True).tolist()
    return index, passages

def retrieve(query, retriever, index, passages, k=3):
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

    # Load generator
    model_name = "bigscience/mt0-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    results = []

    for row in sample:
        question = row["question"]
        gold = row["answers"]["text"][0]

        # Translate Telugu query to English before retrieval
        english_query = translator(question)[0]["translation_text"]

        # Retrieve top-3 Wikipedia passages using the English query
        retrieved = retrieve(english_query, retriever, index, passages, k=3)
        context = "\n\n".join(retrieved)

        # Prompt with retrieved context; ask explicitly for a Telugu answer
        prompt = (
            f"Answer the following question in Telugu using the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Telugu Answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

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