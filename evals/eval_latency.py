import math, time, asyncio
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
from src.generator import generate_answer, generate_streaming_answer

load_dotenv() # Load environment variables

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

REPEATS = 5  # Number of times to repeat the evaluation for each question
WARMUP_RUNS = 2  # Number of warmup runs to perform before measuring latency

MEASURE_TTFT = True  # Measure Time To First Token (TTFT) for streaming responses
STAGE_LEVEL = True  # Measure latency at each stage of the RAG pipeline (retrieval, reranking, generation)

# SLOs/budgets (A latency number is meaningless and insignificant without a target budget or SLO to compare against)
SLO_P95_MS = 3000  # end-to-end latency SLO (95th percentile) in milliseconds
SLO_TTFT_P95_MS = 1200  # Time To First Token SLO (95th percentile) in milliseconds

async def run_end_to_end_pipeline(pipeline, question):
    result = await pipeline.invoke(question)
    return result["answer"]

# Stage-level (non-streaming) - reuses the RAG pipeline's retriever and generator timing each component.
async def run_stages(pipeline, question):
    t0 = time.perf_counter()
    docs = await pipeline.retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()
    answer = await generate_answer(question, context)
    t2 = time.perf_counter()
    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000
    }

# Stage-level (streaming) - same retrieval, but stream generation and record the clock the instant the FIRST response token arrives.
async def run_stages_streaming(pipeline, question):
    t0 = time.perf_counter()
    docs = await pipeline.retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()

    first_token_time = None
    pieces = []

    for token in generate_streaming_answer(question, context):
        if first_token_time is None:
            first_token_time = time.perf_counter()  # Clocks the first non-empty chunk

        pieces.append(token)

    t2 = time.perf_counter()
    answer = "".join(pieces)
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else float("nan")
    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms
    }

# Percentile helpers
def percentile(values, p):
    values = [value for value in values if not math.isnan(value)]

    if not values:
        return float("nan")

    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100.0)
    low, high = math.floor(k), math.ceil(k)

    if low == high:
        return sorted_values[int(k)]
    return sorted_values[low] * (high - k) + sorted_values[high] * (k - low)

# Benchmark loop
async def benchmark(pipeline):
    # Warmup - Run and DISCARD, so cold start does not pollute stats
    print(f"Warming up ({WARMUP_RUNS} runs, discarded)...")

    for idx in range(WARMUP_RUNS):
        await run_end_to_end_pipeline(pipeline, QUESTIONS[idx % len(QUESTIONS)])

    total_ms, retrieval_ms, generation_ms, ttft_ms = [], [], [], []
    answer_lengths = []

    # Measured runs - each question REPEATS multiple times, pool all samples
    print("Measuring...")

    for question in QUESTIONS:
        for _ in range(REPEATS):
            start = time.perf_counter()

            if MEASURE_TTFT:
                answer, stage = await run_stages_streaming(pipeline, question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
                ttft_ms.append(stage["ttft"])
            elif STAGE_LEVEL:
                answer, stage = await run_stages(pipeline, question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
            else:
                answer = await run_end_to_end_pipeline(pipeline, question)

            elapsed_ms = (time.perf_counter() - start) * 1000

            total_ms.append(elapsed_ms)
            answer_lengths.append(len(answer or ""))

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths
    }

# Aggregation and reporting
def summarize(samples):
    filtered_samples = [sample for sample in samples if not math.isnan(sample)]
    return {
        "n": len(filtered_samples),
        "mean": sum(filtered_samples) / len(filtered_samples),
        "p50": percentile(filtered_samples, 50),
        "p95": percentile(filtered_samples, 95),
        "p99": percentile(filtered_samples, 99),
        "min": min(filtered_samples),
        "max": max(filtered_samples)
    }

def print_row(label, s):
    print(f"{label:<12} | n={s['n']:<3} "
          f"mean={s['mean']:7.1f}  p50={s['p50']:7.1f}  "
          f"p95={s['p95']:7.1f}  p99={s['p99']:7.1f}  "
          f"min={s['min']:7.1f}  max={s['max']:7.1f}")

def slo_line(label, p95, budget):
    verdict = "PASS" if p95 <= budget else "FAIL"
    print(f"SLO: {label:<22} p95 <= {budget:>5} ms -> p95 = {p95:7.0f} ms [{verdict}]")

def generate_report(results):
    print("\n" + "=" * 78)
    print("LATENCY (milliseconds)")
    print("=" * 78)
    print(f"{'stage':<12} | {'samples':<5} {'mean':>11} {'p50':>11} "
          f"{'p95':>11} {'p99':>11} {'min':>11} {'max':>11}")
    print("-" * 78)

    total = summarize(results["total"])
    print_row("end-to-end", total)
    if results["ttft"]:
        print_row("ttft", summarize(results["ttft"]))   # perceived: query -> first token
    if results["retrieval"]:
        print_row("retrieval", summarize(results["retrieval"]))
        print_row("generation", summarize(results["generation"]))

    avg_len = sum(results["answer_len"]) / len(results["answer_len"])
    print("-" * 78)
    print(f"avg answer length: {avg_len:.0f} chars "
          f"(latency scales with output length -- keep in mind when comparing configs)")

    # SLO verdicts: the teaching contrast lives here
    print("=" * 78)
    slo_line("full answer", total["p95"], SLO_P95_MS)
    if results["ttft"]:
        slo_line("first token (perceived)", summarize(results["ttft"])["p95"], SLO_TTFT_P95_MS)
    print("=" * 78)

# Entrypoint
async def main():
    results = await benchmark(rag_pipeline)
    generate_report(results)

if __name__ == "__main__":
    asyncio.run(main())