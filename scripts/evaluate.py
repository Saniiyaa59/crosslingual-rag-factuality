from langdetect import detect

def compute_rlc(results, target_lang="te"):
    correct_lang = 0
    for r in results:
        try:
            detected = detect(r["prediction"])
            if detected == target_lang:
                correct_lang += 1
        except:
            pass  # langdetect fails on very short strings
    return correct_lang / len(results)


import json

if __name__ == "__main__":
    with open("./data/results_crosslingual.json") as f:
        results = json.load(f)

    rlc = compute_rlc(results)
    print(f"RLC: {rlc:.2f}")