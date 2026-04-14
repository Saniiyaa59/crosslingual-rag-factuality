from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
    # Load data
    telugu = load_from_disk("data/tydiqa_telugu_train")
    sample = telugu.select(range(10))

    # Load retriever
    retriever = SentenceTransformer("BAAI/bge-m3")
    retriever.max_seq_length = 512

    # Load index
    index, passages = load_index(
        "data/index/english_bge.faiss",
        "data/index/english_passages.npy"
    )

    # Load generator
    model_name = "bigscience/mt0-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    results = []

    for row in sample:
        question = row["question"]
        gold = row["answers"]["text"][0]

        # Retrieve top-3 English passages
        retrieved = retrieve(question, retriever, index, passages, k=3)
        context = "\n\n".join(retrieved)

        # Prompt with retrieved context
        prompt = f"Answer the following question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "retrieved_passages": retrieved
        })
        print(f"Q: {question}")
        print(f"Gold: {gold}")
        print(f"Pred: {prediction}")
        print(f"Retrieved: {retrieved[0][:200]}...")
        print("---")

    question = "వేప చెట్టు యొక్క శాస్త్రీయ నామం ఏమిటి?"  # neem tree scientific name
    retrieved = retrieve(question, retriever, index, passages, k=3)
    for p in retrieved:
        print(p[:300])
        print("---")

    # Save results
    with open("data/results_crosslingual.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)