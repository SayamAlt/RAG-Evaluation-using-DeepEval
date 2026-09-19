from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
import asyncio
from src.generator import prompt, llm  # Reuse the exact prompt and model

load_dotenv()  # Load the environment variables

# Stop before StrOutputParser() so that the AIMessage (with usage metadata) arrives
measured_chain = prompt | llm

# Define config
QUESTIONS = [
    "What charge did Orlando Taylor dispute with Andrus Bank, and how much was it?",
    "What reason did United Care give for denying the $1,000 claim?",
    "What steps did the retail agent take to resolve John Rena's duplicate charge?",
    "How did agent Candice at Hotel California collect payment for a room reservation?",
    "What two extra charges appeared on Mr. Garfield's Bright Star Services bill that month?"
]

# Invoke the RAG pipeline
rag_pipeline = asyncio.run(RAGPipeline.create(fetch_k=20, top_k=5))

REPEATS = 3  # Cost is stable, so fewer repeats needed than latency

# Pricing: gpt-4o-mini, USD per 1M tokens (verified Aug 2026)
PRICE_INPUT_PER_1M = 0.15          # cache-miss input
PRICE_CACHED_INPUT_PER_1M = 0.075  # cached (repeated prefix) input - half price
PRICE_OUTPUT_PER_1M = 0.60         # output (4x input - long answers dominate)

# Business projection knobs - Configure these as per business needs
QUERIES_PER_DAY = 2000  # expected traffic requests
USD_TO_INR = 95.0       # approximate; set to the current rate

# Budget (the "SLO" for cost): the offline pass/fail line
COST_BUDGET_PER_QUERY_USD = 0.0015  # e.g. must stay under ~0.13 INR / query

# Token measurement
async def measure_tokens(pipeline, question):
    docs = await pipeline.retriever.invoke(question)
    context_text = "\n".join([doc.page_content for doc in docs])

    message = measured_chain.invoke({"question": question, "context": context_text})
    usage = message.usage_metadata or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    # cached prefix tokens, if the provider reports them
    details = usage.get("input_token_details") or {}
    cached_tokens = details.get("cache_read", 0) or 0

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens
    }

# Determine the total cost for I/P tokens, O/P tokens, and cached tokens
def determine_total_cost_usd(input_tokens, output_tokens, cached_tokens):
    uncached_input = max(input_tokens - cached_tokens, 0)
    input_cost = uncached_input / 1_000_000 * PRICE_INPUT_PER_1M
    cached_input_cost = cached_tokens / 1_000_000 * PRICE_CACHED_INPUT_PER_1M
    output_cost = output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M

    return {
        "input": input_cost,
        "cached": cached_input_cost,
        "output": output_cost,
        "total": input_cost + cached_input_cost + output_cost
    }

# Benchmark loop
async def benchmark(pipeline):
    rows = []
    print("Measuring token usage...")

    for question in QUESTIONS:
        for _ in range(REPEATS):
            tokens = await measure_tokens(pipeline, question)
            cost = determine_total_cost_usd(tokens["input"], tokens["output"], tokens["cached"])
            rows.append({**tokens, **{f"cost_{k}": v for k, v in cost.items()}})

    return rows

# Aggregate + Report
def avg(rows, key):
    return sum(row[key] for row in rows) / len(rows)

def generate_report(rows):
    num_rows = len(rows)
    avg_input = avg(rows, "input")
    avg_output = avg(rows, "output")
    avg_cached = avg(rows, "cached")
    avg_cost = avg(rows, "cost_total")
    min_cost = min(row["cost_total"] for row in rows)
    max_cost = max(row["cost_total"] for row in rows)

    # Split: how much of the bill belongs to input and output
    avg_cost_input = avg(rows, "cost_input") + avg(rows, "cost_cached")
    avg_cost_output = avg(rows, "cost_output")
    output_share = 100 * avg_cost_output / avg_cost if avg_cost else 0

    print("\n" + "=" * 70)
    print(f"COST  (gpt-4o-mini @ ${PRICE_INPUT_PER_1M}/${PRICE_OUTPUT_PER_1M} per 1M in/out)")
    print("=" * 70)
    print(f"Number of samples: {num_rows}")
    print(f"avg input tokens:  {avg_input:8.0f} ({avg_cached:.0f} cached)")
    print(f"avg output tokens: {avg_output:8.0f}")
    print("-" * 70)
    print(f"avg cost / query:  ${avg_cost:.6f}   (Rs {avg_cost * USD_TO_INR:.4f})")
    print(f"min / max:         ${min_cost:.6f} / ${max_cost:.6f}"
          f" <- tight range = cost is stable, unlike latency")
    print(f"input vs output:   {100 - output_share:.0f}% input / {output_share:.0f}% output"
          f" (output is 4x the rate -> long answers dominate)")
    print("-" * 70)

    # Projection: the number a founder actually cares about
    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"projection @ {QUERIES_PER_DAY}/day:")
    print(f"  per day:   ${daily:8.2f}  (Rs {daily * USD_TO_INR:8.2f})")
    print(f"  per month: ${monthly:8.2f}  (Rs {monthly * USD_TO_INR:8.2f})")
    print("=" * 70)

    # Budget verdict (the offline pass/fail)
    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  "
          f"${avg_cost:.6f} [{verdict}]")
    print("=" * 70)
    print("note: production caching of the (large, fixed) system prompt can push "
          "the real bill BELOW this estimate - watch the 'cached' count grow online.")

# Entrypoint
async def main():
    rows = await benchmark(rag_pipeline)
    generate_report(rows)

if __name__ == "__main__":
    asyncio.run(main())