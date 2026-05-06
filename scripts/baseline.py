from datasets import load_from_disk
from transformers import pipeline
import torch
import json

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    telugu = load_from_disk("data/tydiqa_telugu_train")
    sample = telugu.select(range(500))

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

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multilingual QA assistant. "
                    "Answer the question using your knowledge. "
                    "Always answer in the same language as the question. "
                    "Be concise — one sentence or less."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Answer in Telugu."
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
        })
        print(f"Q: {question}")
        print(f"Gold: {gold}")
        print(f"Pred: {prediction}")
        print("---")

    with open("data/results_baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(results)} results saved to data/results_baseline.json")