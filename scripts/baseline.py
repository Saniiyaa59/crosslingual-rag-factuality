from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

if __name__ == "__main__":
    # Load Telugu test data
    telugu = load_from_disk("data/tydiqa_telugu_train")
    print(telugu)

    # Use a small sample to test the pipeline first
    sample = telugu.select(range(10))

    # Load model
    model_name = "bigscience/mt0-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    results = []

    for row in sample:
        question = row["question"]
        gold = row["answers"]["text"][0]

        # Prompt — no retrieval, just the question
        prompt = f"Answer the following question: {question}"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "question": question,
            "gold": gold,
            "prediction": prediction
        })
        print(f"Q: {question}")
        print(f"Gold: {gold}")
        print(f"Pred: {prediction}")
        print("---")