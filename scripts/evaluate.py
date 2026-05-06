import json
import re
import unicodedata
from pathlib import Path

try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("Warning: langdetect not installed — RLC will be skipped.")


# ── Text normalization ──────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


# ── Metrics ────────────────────────────────────────────────────────────────

def exact_match(prediction: str, gold: str) -> int:
    return int(normalize(prediction) == normalize(gold))


def char_ngram_recall(prediction: str, gold: str, n: int = 3) -> float:
    pred_norm = normalize(prediction)
    gold_norm = normalize(gold)

    def ngrams(s, n):
        return [s[i:i+n] for i in range(len(s) - n + 1)]

    gold_ng = ngrams(gold_norm, n)
    if not gold_ng:
        return float(normalize(prediction) == normalize(gold))

    pred_ng_set = set(ngrams(pred_norm, n))
    matched = sum(1 for g in gold_ng if g in pred_ng_set)
    return matched / len(gold_ng)


def answer_in_prediction(prediction: str, gold: str) -> int:
    return int(normalize(gold) in normalize(prediction))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(gold_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def response_language_correct(prediction: str, target_lang: str = "te") -> int:
    if not HAS_LANGDETECT:
        return -1
    try:
        return int(detect(prediction) == target_lang)
    except Exception:
        return 0


# ── Per-condition evaluation ────────────────────────────────────────────────

def evaluate(results: list, label: str) -> dict:
    em_scores, chr3_scores, aip_scores, f1_scores, rlc_scores = [], [], [], [], []

    for r in results:
        pred = r.get("prediction", "")
        gold = r.get("gold", "")

        em_scores.append(exact_match(pred, gold))
        chr3_scores.append(char_ngram_recall(pred, gold, n=3))
        aip_scores.append(answer_in_prediction(pred, gold))
        f1_scores.append(token_f1(pred, gold))

        rlc = response_language_correct(pred)
        if rlc >= 0:
            rlc_scores.append(rlc)

    n = len(results)
    metrics = {
        "condition": label,
        "n": n,
        "EM":   sum(em_scores) / n if n else 0,
        "Chr3": sum(chr3_scores) / n if n else 0,
        "AIP":  sum(aip_scores) / n if n else 0,
        "F1":   sum(f1_scores) / n if n else 0,
    }
    if rlc_scores:
        metrics["RLC"] = sum(rlc_scores) / len(rlc_scores)
    else:
        metrics["RLC"] = None

    return metrics


# ── Entry point ─────────────────────────────────────────────────────────────

CONDITION_FILES = [
    ("A — No retrieval",      "data/results_baseline.json"),
    ("B — Monolingual TE",    "data/results_monolingual.json"),
    ("C — Cross-lingual EN",  "data/results_crosslingual_wiki.json"),
    ("D — Combined EN+TE",    "data/results_combined.json"),
]

if __name__ == "__main__":
    all_metrics = []

    for label, path in CONDITION_FILES:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {path} not found")
            continue
        with open(p, encoding="utf-8") as f:
            results = json.load(f)
        metrics = evaluate(results, label)
        all_metrics.append(metrics)

        # Per-example breakdown
        print(f"\n{'='*60}")
        print(f"  {label}  (n={metrics['n']})")
        print(f"{'='*60}")
        for r in results:
            pred = r.get("prediction", "")
            gold = r.get("gold", "")
            em = exact_match(pred, gold)
            c3 = char_ngram_recall(pred, gold)
            aip = answer_in_prediction(pred, gold)
            f1 = token_f1(pred, gold)
            q_short = r.get("question", "")[:50]
            print(f"  Q: {q_short}")
            print(f"     Gold:  {gold[:80]}")
            print(f"     Pred:  {pred[:80]}")
            print(f"     EM={em}  Chr3={c3:.3f}  AIP={aip}  F1={f1:.3f}")
            print()

    # Summary table
    if all_metrics:
        print("\n" + "="*80)
        print(f"{'Condition':<28} {'N':>4}  {'EM':>6}  {'Chr-3':>7}  {'AIP':>6}  {'F1':>6}  {'RLC':>6}")
        print("-"*80)
        for m in all_metrics:
            rlc_str = f"{m['RLC']:.3f}" if m["RLC"] is not None else "  n/a"
            print(f"{m['condition']:<28} {m['n']:>4}  {m['EM']:>6.3f}  {m['Chr3']:>7.3f}  {m['AIP']:>6.3f}  {m['F1']:>6.3f}  {rlc_str:>6}")
        print("="*80)
        print("AIP = Answer In Prediction (gold span appears anywhere in output)")