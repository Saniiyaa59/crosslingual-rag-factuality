from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from retrieval_utils import load_index, retrieve
import torch
import json

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data — same as rag.py
    telugu = load_from_disk("data/tydiqa_telugu_train")
    sample = telugu.select(range(10))

    # Load retriever — same model and settings as rag.py
    retriever = SentenceTransformer("BAAI/bge-m3", device=device)
    retriever.max_seq_length = 512

    # Load Telugu index built by build_telugu_index.py
    index, passages = load_index(
        "data/index/telugu_wiki_bge.faiss",
        "data/index/telugu_wiki_passages.npy"
    )
    print(f"Index loaded: {index.ntotal} passages")

    # Load generator — same model as baseline.py and rag.py
    model_name = "bigscience/mt0-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    results = []

    for row in sample:
        question = row["question"]
        gold = row["answers"]["text"][0]

        # Retrieve top-3 Telugu Wikipedia passages
        retrieved = retrieve(question, retriever, index, passages, k=3)
        context = "\n\n".join(retrieved)

        # Prompt — same structure as rag.py
        prompt = f"Answer the following question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "retrieved_passages": retrieved
        })

        # Same print format as rag.py
        print(f"Q: {question}")
        print(f"Gold: {gold}")
        print(f"Pred: {prediction}")
        print(f"Retrieved: {retrieved[0][:200]}...")
        print("---")

    # Save results — same pattern as rag.py
    with open("data/results_monolingual.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(results)} results saved to data/results_monolingual.json")