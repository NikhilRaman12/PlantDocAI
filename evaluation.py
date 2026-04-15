import json
import time
import math
from generator import ResponseGenerator

# RAGAS LLM-based metrics require a valid OpenAI key to run without hanging.
# Set RAGAS_AVAILABLE = True and configure an LLM client when a key is available.
RAGAS_AVAILABLE = False

EVAL_QUERIES = [
    "Symptoms of Fusarium wilt in tomato",
    "Management of bacterial blight in rice",
    "How to control powdery mildew in cucurbits?",
    "Integrated pest management for cotton bollworm",
    "Nutrient deficiency signs in groundnut"
]

GROUND_TRUTHS = {
    "Symptoms of Fusarium wilt in tomato": "Fusarium wilt in tomato typically shows lower leaf yellowing, one-sided wilting, vascular browning in stem, stunted growth, and eventual plant collapse.",
    "Management of bacterial blight in rice": "Manage bacterial blight in rice with resistant varieties, balanced fertilization (avoid excess nitrogen), clean field sanitation, proper spacing, and recommended bactericide practices where advised.",
    "How to control powdery mildew in cucurbits?": "Control powdery mildew in cucurbits by using resistant varieties, reducing canopy humidity, removing infected plant parts, and applying suitable fungicides at recommended intervals.",
    "Integrated pest management for cotton bollworm": "IPM for cotton bollworm includes pheromone traps, scouting and threshold-based action, conserving natural enemies, use of biocontrols, and need-based insecticide rotation.",
    "Nutrient deficiency signs in groundnut": "Groundnut nutrient deficiency can include chlorosis, poor growth, leaf discoloration, and specific symptoms such as boron deficiency causing distorted young leaves and poor pod development."
}

def calculate_cost(prompt_tokens, completion_tokens, rate_prompt=0.0000015, rate_completion=0.000002):
    return (prompt_tokens * rate_prompt) + (completion_tokens * rate_completion)


def _sanitize_for_json(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value

def run_evaluation():
    generator = ResponseGenerator()
    results = []
    ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for query in EVAL_QUERIES:
        try:
            start = time.time()
            response = generator.run(query)
            end = time.time()

            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = prompt_tokens + completion_tokens
            cost = calculate_cost(prompt_tokens, completion_tokens)
            retrieved_contexts = generator.retrieve_documents(query) or []
            ground_truth = GROUND_TRUTHS.get(query, "information not available")

            ragas_data["question"].append(query)
            ragas_data["answer"].append(response.content)
            ragas_data["contexts"].append(retrieved_contexts)
            ragas_data["ground_truth"].append(ground_truth)

            results.append({
                "query": query,
                "answer": response.content,
                "contexts_count": len(retrieved_contexts),
                "latency_seconds": round(end - start, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(cost, 6)
            })

        except Exception as e:
            results.append({"query": query, "error": str(e)})

    if RAGAS_AVAILABLE:
        try:
            metrics = [ContextPrecisionWithReference(), ContextRecall()]
            hf_dataset = HFDataset.from_dict(ragas_data)
            ragas_results = evaluate(hf_dataset, metrics=metrics)
            for i, r in enumerate(results):
                if "error" not in r:
                    r["context_precision"] = float(ragas_results["context_precision"][i])
                    r["context_recall"] = float(ragas_results["context_recall"][i])
        except Exception as e:
            print(f"RAGAS evaluation skipped: {e}")
            for r in results:
                if "error" not in r:
                    r["context_precision"] = None
                    r["context_recall"] = None
    else:
        print("RAGAS not available — skipping metric evaluation")
        for r in results:
            if "error" not in r:
                r["context_precision"] = None
                r["context_recall"] = None

    valid_results = [r for r in results if "error" not in r]
    precision_values = [r["context_precision"] for r in valid_results if r.get("context_precision") is not None]
    recall_values = [r["context_recall"] for r in valid_results if r.get("context_recall") is not None]

    summary = {
        "avg_latency": round(sum(r["latency_seconds"] for r in valid_results) / len(valid_results), 3) if valid_results else 0,
        "avg_cost_usd": round(sum(r["estimated_cost_usd"] for r in valid_results) / len(valid_results), 6) if valid_results else 0,
        "avg_context_precision": float(sum(precision_values) / len(precision_values)) if precision_values else None,
        "avg_context_recall": float(sum(recall_values) / len(recall_values)) if recall_values else None
    }

    return _sanitize_for_json({"results": results, "summary": summary})

if __name__ == "__main__":
    eval_output = run_evaluation()
    with open("results.json", "w") as f:
        json.dump(eval_output, f, indent=2, allow_nan=False)
    print("Evaluation complete. Results + summary saved to results.json")
