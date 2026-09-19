import math, time, asyncio
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
from src.generator import prompt, llm, generate_answer, generate_streaming_answer

load_dotenv()

QUESTIONS = [
    "What charge did Orlando Taylor dispute with Andrus Bank, and how much was it?",
    "What reason did United Care give for denying the $1,000 claim?",
    "What steps did the retail agent take to resolve John Rena's duplicate charge?",
    "How did agent Candice at Hotel California collect payment for a room reservation?",
    "What two extra charges appeared on Mr. Garfield's Bright Star Services bill that month?"
]

measured_chain = prompt | llm

LATENCY_REPEATS = 5
WARMUP_RUNS = 2
MEASURE_TTFT = True
STAGE_LEVEL = True
SLO_P95_MS = 6000
SLO_TTFT_P95_MS = 5000

COST_REPEATS = 3
PRICE_INPUT_PER_1M = 0.15
PRICE_CACHED_INPUT_PER_1M = 0.075
PRICE_OUTPUT_PER_1M = 0.60
QUERIES_PER_DAY = 2000
USD_TO_INR = 95.0
COST_BUDGET_PER_QUERY_USD = 0.0015

RELIABILITY_REPEATS = 5
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5

async def run_end_to_end_pipeline(rag, question):
    result = await rag.invoke(question)
    return result["answer"]

async def run_stages(rag, question):
    t0 = time.perf_counter()
    docs = await rag.retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()
    answer = await generate_answer(question, context)
    t2 = time.perf_counter()
    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000
    }

async def run_stages_streaming(rag, question):
    t0 = time.perf_counter()
    docs = await rag.retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()

    first_token_time = None
    pieces = []

    for token in generate_streaming_answer(question, context):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        pieces.append(token)

    t2 = time.perf_counter()
    answer = "".join(pieces)
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else float("nan")
    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms
    }

def percentile(values, p):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100.0)
    low, high = math.floor(k), math.ceil(k)
    if low == high:
        return sorted_values[int(k)]
    return sorted_values[low] * (high - k) + sorted_values[high] * (k - low)

async def benchmark_latency(rag):
    print(f"Warming up ({WARMUP_RUNS} runs, discarded)...")
    for idx in range(WARMUP_RUNS):
        await run_end_to_end_pipeline(rag, QUESTIONS[idx % len(QUESTIONS)])

    total_ms, retrieval_ms, generation_ms, ttft_ms = [], [], [], []
    answer_lengths = []

    print("Measuring...")
    for question in QUESTIONS:
        for _ in range(LATENCY_REPEATS):
            start = time.perf_counter()

            if MEASURE_TTFT:
                answer, stage = await run_stages_streaming(rag, question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
                ttft_ms.append(stage["ttft"])
            elif STAGE_LEVEL:
                answer, stage = await run_stages(rag, question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
            else:
                answer = await run_end_to_end_pipeline(rag, question)

            total_ms.append((time.perf_counter() - start) * 1000)
            answer_lengths.append(len(answer or ""))

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths
    }

def summarize_latency(samples):
    filtered = [s for s in samples if not math.isnan(s)]
    return {
        "n": len(filtered),
        "mean": sum(filtered) / len(filtered),
        "p50": percentile(filtered, 50),
        "p95": percentile(filtered, 95),
        "p99": percentile(filtered, 99),
        "min": min(filtered),
        "max": max(filtered)
    }

def print_row(label, s):
    print(f"{label:<12} | n={s['n']:<3} "
          f"mean={s['mean']:7.1f}  p50={s['p50']:7.1f}  "
          f"p95={s['p95']:7.1f}  p99={s['p99']:7.1f}  "
          f"min={s['min']:7.1f}  max={s['max']:7.1f}")

def slo_line(label, p95, budget):
    verdict = "PASS" if p95 <= budget else "FAIL"
    print(f"SLO: {label:<22} p95 <= {budget:>5} ms -> p95 = {p95:7.0f} ms [{verdict}]")

def generate_latency_report(results):
    print("\n" + "=" * 78)
    print("LATENCY (milliseconds)")
    print("=" * 78)
    print(f"{'stage':<12} | {'samples':<5} {'mean':>11} {'p50':>11} "
          f"{'p95':>11} {'p99':>11} {'min':>11} {'max':>11}")
    print("-" * 78)

    total = summarize_latency(results["total"])
    print_row("end-to-end", total)
    if results["ttft"]:
        print_row("ttft", summarize_latency(results["ttft"]))
    if results["retrieval"]:
        print_row("retrieval", summarize_latency(results["retrieval"]))
        print_row("generation", summarize_latency(results["generation"]))

    avg_len = sum(results["answer_len"]) / len(results["answer_len"])
    print("-" * 78)
    print(f"avg answer length: {avg_len:.0f} chars "
          f"(latency scales with output length -- keep in mind when comparing configs)")

    print("=" * 78)
    slo_line("full answer", total["p95"], SLO_P95_MS)
    if results["ttft"]:
        slo_line("first token (perceived)", summarize_latency(results["ttft"])["p95"], SLO_TTFT_P95_MS)
    print("=" * 78)

def run_latency(rag, verbose=True):
    results = asyncio.run(benchmark_latency(rag))
    if verbose:
        generate_latency_report(results)
    total = summarize_latency(results["total"])
    return {
        "e2e_p95_ms": total["p95"],
        "e2e_p99_ms": total["p99"],
        "ttft_p95_ms": summarize_latency(results["ttft"])["p95"] if results["ttft"] else float("nan"),
        "slo_pass": total["p95"] <= SLO_P95_MS
    }

async def measure_tokens(rag, question):
    docs = await rag.retriever.invoke(question)
    context_text = "\n".join([doc.page_content for doc in docs])

    message = measured_chain.invoke({"question": question, "context": context_text})
    usage = message.usage_metadata or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    details = usage.get("input_token_details") or {}
    cached_tokens = details.get("cache_read", 0) or 0

    return {"input": input_tokens, "output": output_tokens, "cached": cached_tokens}

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

async def benchmark_cost(rag):
    rows = []
    print("Measuring token usage...")
    for question in QUESTIONS:
        for _ in range(COST_REPEATS):
            tokens = await measure_tokens(rag, question)
            cost = determine_total_cost_usd(tokens["input"], tokens["output"], tokens["cached"])
            rows.append({**tokens, **{f"cost_{k}": v for k, v in cost.items()}})
    return rows

def avg(rows, key):
    return sum(row[key] for row in rows) / len(rows)

def generate_cost_report(rows):
    avg_input = avg(rows, "input")
    avg_output = avg(rows, "output")
    avg_cached = avg(rows, "cached")
    avg_cost = avg(rows, "cost_total")
    min_cost = min(row["cost_total"] for row in rows)
    max_cost = max(row["cost_total"] for row in rows)

    avg_cost_output = avg(rows, "cost_output")
    output_share = 100 * avg_cost_output / avg_cost if avg_cost else 0

    print("\n" + "=" * 70)
    print(f"COST  (gpt-4o-mini @ ${PRICE_INPUT_PER_1M}/${PRICE_OUTPUT_PER_1M} per 1M in/out)")
    print("=" * 70)
    print(f"Number of samples: {len(rows)}")
    print(f"avg input tokens:  {avg_input:8.0f} ({avg_cached:.0f} cached)")
    print(f"avg output tokens: {avg_output:8.0f}")
    print("-" * 70)
    print(f"avg cost / query:  ${avg_cost:.6f}   (Rs {avg_cost * USD_TO_INR:.4f})")
    print(f"min / max:         ${min_cost:.6f} / ${max_cost:.6f}"
          f" <- tight range = cost is stable, unlike latency")
    print(f"input vs output:   {100 - output_share:.0f}% input / {output_share:.0f}% output"
          f" (output is 4x the rate -> long answers dominate)")
    print("-" * 70)

    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"projection @ {QUERIES_PER_DAY}/day:")
    print(f"  per day:   ${daily:8.2f}  (Rs {daily * USD_TO_INR:8.2f})")
    print(f"  per month: ${monthly:8.2f}  (Rs {monthly * USD_TO_INR:8.2f})")
    print("=" * 70)

    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  "
          f"${avg_cost:.6f} [{verdict}]")
    print("=" * 70)
    print("note: production caching of the (large, fixed) system prompt can push "
          "the real bill BELOW this estimate - watch the 'cached' count grow online.")

def run_cost(rag, verbose=True):
    rows = asyncio.run(benchmark_cost(rag))
    if verbose:
        generate_cost_report(rows)
    avg_cost = avg(rows, "cost_total")
    return {
        "avg_cost_usd": avg_cost,
        "avg_input_tokens": avg(rows, "input"),
        "avg_output_tokens": avg(rows, "output"),
        "budget_pass": avg_cost <= COST_BUDGET_PER_QUERY_USD
    }

class Reliability:

    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0

async def call_with_retries(fn, reliability):
    reliability.calls += 1

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await fn()
            reliability.successes += 1
            return result
        except Exception as e:
            if attempt < MAX_RETRIES:
                reliability.retries += 1
                await asyncio.sleep(BACKOFF_BASE_S * (2 ** attempt))
            else:
                reliability.failures += 1
                print(f"FAILED after {MAX_RETRIES} retries: {e}")
                return None

async def benchmark_reliability(rag):
    reliability = Reliability()
    print("Measuring reliability...")

    for question in QUESTIONS:
        for _ in range(RELIABILITY_REPEATS):
            q = question
            await call_with_retries(
                lambda q=q: rag.invoke(q),
                reliability
            )

    return reliability

def generate_reliability_report(reliability):
    success_rate = 100 * reliability.successes / reliability.calls if reliability.calls else 0
    error_rate   = 100 * reliability.failures  / reliability.calls if reliability.calls else 0
    retry_rate   = 100 * reliability.retries   / reliability.calls if reliability.calls else 0

    print("\n" + "=" * 60)
    print("RELIABILITY")
    print("=" * 60)
    print(f"Total requests:      {reliability.calls}")
    print(f"Successful requests: {reliability.successes}")
    print(f"Failed requests:     {reliability.failures}")
    print(f"Retried requests:    {reliability.retries}")
    print("-" * 60)
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Error rate:   {error_rate:.2f}%")
    print(f"Retry rate:   {retry_rate:.2f}%")
    print("=" * 60)

def run_reliability(rag, verbose=True):
    reliability = asyncio.run(benchmark_reliability(rag))
    if verbose:
        generate_reliability_report(reliability)
    success_rate = 100 * reliability.successes / reliability.calls if reliability.calls else 0
    error_rate   = 100 * reliability.failures  / reliability.calls if reliability.calls else 0
    retry_rate   = 100 * reliability.retries   / reliability.calls if reliability.calls else 0
    return {
        "success_rate": success_rate,
        "error_rate": error_rate,
        "retry_rate": retry_rate,
        "total_calls": reliability.calls
    }

def run_operations(rag=None, verbose=True):
    rag = rag or asyncio.run(RAGPipeline.create(fetch_k=20, top_k=5))

    latency     = run_latency(rag, verbose=verbose)
    cost        = run_cost(rag, verbose=verbose)
    reliability = run_reliability(rag, verbose=verbose)

    snapshot = {}
    snapshot.update({f"latency.{k}": v     for k, v in latency.items()})
    snapshot.update({f"cost.{k}": v        for k, v in cost.items()})
    snapshot.update({f"reliability.{k}": v for k, v in reliability.items()})
    return snapshot

def main():
    snapshot = run_operations(verbose=True)

    print("\n" + "=" * 70)
    print("OPERATIONS SNAPSHOT  (feeds regression testing)")
    print("=" * 70)
    for key, value in snapshot.items():
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key:<36} {shown}")
    print("=" * 70)

if __name__ == "__main__":
    main()