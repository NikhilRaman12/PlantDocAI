import json
import re
import time
import types


# ─────────────────────────────────────────────
# SAFE LOADER  (avoids circular import)
# ─────────────────────────────────────────────
def load_generator():
    with open("generator.py", "r", encoding="utf-8") as f:
        code = f.read()
    code = code.replace("from generator import ResponseGenerator", "")
    mod = types.ModuleType("generator_fixed")
    exec(code, mod.__dict__)
    return mod.ResponseGenerator


# ─────────────────────────────────────────────
# CONFIG — one query per knowledge-base PDF
# ─────────────────────────────────────────────
TOP_K = 4
COST_PER_1K_TOKENS = 0.5   # Groq free-tier placeholder

QUERIES = [
    # Banana.pdf
    {
        "query": "Symptoms and management of Panama disease in banana",
        "source": "Banana.pdf",
        "concepts": ["banana", "fusarium", "panama", "yellowing", "wilting"]
    },
    # biopesticides.pdf
    {
        "query": "How does neem-based biopesticide control crop pests?",
        "source": "biopesticides.pdf",
        "concepts": ["neem", "azadirachtin", "biopesticide", "pest", "spray"]
    },
    # cotton.pdf
    {
        "query": "Management of cotton bollworm using integrated pest management",
        "source": "cotton.pdf",
        "concepts": ["cotton", "bollworm", "ipm", "pheromone", "insecticide"]
    },
    # cucurbitaceous.pdf
    {
        "query": "Control of powdery mildew and downy mildew in cucurbits",
        "source": "cucurbitaceous.pdf",
        "concepts": ["cucurbit", "powdery mildew", "downy mildew", "fungicide", "spray"]
    },
    # diseasecausing_agents.pdf
    {
        "query": "Types of plant disease causing agents and their characteristics",
        "source": "diseasecausing_agents.pdf",
        "concepts": ["bacteria", "fungi", "virus", "pathogen", "disease"]
    },
    # diseases_horticulture.pdf
    {
        "query": "Common fungal diseases in horticulture crops and their treatment",
        "source": "diseases_horticulture.pdf",
        "concepts": ["horticulture", "fungal", "fruit", "vegetable", "disease"]
    },
    # disease_fieldcrops.pdf
    {
        "query": "Symptoms of rust and smut diseases in wheat and field crops",
        "source": "disease_fieldcrops.pdf",
        "concepts": ["wheat", "rust", "smut", "field", "disease"]
    },
    # insectpest_fieldcrops.pdf
    {
        "query": "Major insect pests attacking field crops and control measures",
        "source": "insectpest_fieldcrops.pdf",
        "concepts": ["aphid", "thrips", "pest", "field", "control"]
    },
    # IPM-Vegetables.pdf
    {
        "query": "Integrated pest management practices for vegetable crops",
        "source": "IPM-Vegetables.pdf",
        "concepts": ["ipm", "vegetable", "biological", "monitoring", "trap"]
    },
    # Organic-Farming.pdf
    {
        "query": "Role of compost and biofertilizers in organic farming",
        "source": "Organic-Farming.pdf",
        "concepts": ["organic", "compost", "biofertilizer", "soil", "farming"]
    },
    # pesticides_list_updated.pdf
    {
        "query": "Recommended fungicides and insecticides with dosage for crop protection",
        "source": "pesticides_list_updated.pdf",
        "concepts": ["fungicide", "insecticide", "dosage", "pesticide", "chemical"]
    },
    # plant_disease_management.pdf
    {
        "query": "Cultural and chemical methods for plant disease management",
        "source": "plant_disease_management.pdf",
        "concepts": ["cultural", "chemical", "management", "disease", "prevention"]
    },
    # Rice.pdf
    {
        "query": "Symptoms and management of rice blast and bacterial blight",
        "source": "Rice.pdf",
        "concepts": ["rice", "blast", "bacterial blight", "sheath", "fungicide"]
    },
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _clean(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def precision_at_k(docs, concepts: list, k: int) -> float:
    if not docs:
        return 0.0
    hits = sum(
        1 for doc in docs[:k]
        if any(c in _clean(doc.page_content) for c in concepts)
    )
    return round(hits / k, 3)


def recall_at_k(docs, concepts: list) -> float:
    if not docs:
        return 0.0
    all_text = " ".join(_clean(d.page_content) for d in docs)
    hits = sum(1 for c in concepts if c in all_text)
    return round(hits / len(concepts), 3)


def relevance_score(response: str, concepts: list) -> float:
    text = _clean(response)
    hits = sum(1 for c in concepts if c in text)
    return round(hits / len(concepts), 3)


def final_score(p: float, r: float, rel: float) -> float:
    return round(p * 0.4 + r * 0.3 + rel * 0.3, 3)


def estimate_tokens(text: str) -> int:
    return len(text.split())


def compute_cost(tokens: int) -> float:
    return round((tokens / 1000) * COST_PER_1K_TOKENS, 4)


def detect_mode(generator, context: str) -> str:
    if generator.retriever is None:
        return "LLM_ONLY"
    if not context:
        return "FALLBACK"
    if len(context.strip()) < 100:
        return "HYBRID"
    return "RAG"


# ─────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────
def run_evaluation():
    ResponseGenerator = load_generator()
    generator = ResponseGenerator()
    results = []

    for item in QUERIES:
        query = item["query"]
        concepts = item["concepts"]
        source = item["source"]

        print(f"\n{'='*60}")
        print(f"Query  : {query}")
        print(f"Source : {source}")
        print("="*60)

        try:
            docs = []
            context = ""
            if generator.retriever:
                docs = generator.vectorstore.similarity_search(query, k=TOP_K)
                context = generator.get_context(query) or ""

            start = time.time()
            response_obj = generator.run(query)
            latency = round(time.time() - start, 3)

            response = response_obj.content
            tokens = estimate_tokens(response)

            p   = precision_at_k(docs, concepts, TOP_K)
            r   = recall_at_k(docs, concepts)
            rel = relevance_score(response, concepts)

            detected_mode = getattr(response_obj, "mode", None) or detect_mode(generator, context)

            results.append({
                "query":            query,
                "source_pdf":       source,
                "mode":             detected_mode,
                "precision_at_k":   p,
                "recall_at_k":      r,
                "relevance":        rel,
                "final_score":      final_score(p, r, rel),
                "latency_sec":      latency,
                "tokens":           tokens,
                "cost_usd":         compute_cost(tokens),
                "contexts_count":   len(docs),
                "response_preview": response[:300] + "..."
            })

            print(f"  Mode={results[-1]['mode']}  P@K={p}  R@K={r}  Rel={rel}  Score={results[-1]['final_score']}  {latency}s")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"query": query, "source_pdf": source, "error": str(exc)})

    return results


def summarise(results: list) -> dict:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {}

    def avg(key):
        return round(sum(r[key] for r in valid) / len(valid), 3)

    return {
        "total_queries":        len(results),
        "successful":           len(valid),
        "avg_precision_at_k":   avg("precision_at_k"),
        "avg_recall_at_k":      avg("recall_at_k"),
        "avg_relevance":        avg("relevance"),
        "avg_final_score":      avg("final_score"),
        "avg_latency_sec":      avg("latency_sec"),
        "avg_cost_usd":         avg("cost_usd"),
    }

if __name__ == "__main__":
    results = run_evaluation()
    summary = summarise(results)

    output = {"results": results, "summary": summary}

    for output_file in ("evaluation_results.json", "results.json"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k:<25} {v}")
    print("\nEvaluation complete. Saved to evaluation_results.json and results.json")
