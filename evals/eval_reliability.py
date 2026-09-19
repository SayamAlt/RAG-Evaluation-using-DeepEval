import asyncio
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

load_dotenv()  # Load environment variables

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

REPEATS = 5

MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5


# Reliability tracker
class Reliability:

    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0

# Retry wrapper
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

                # Exponential backoff - use asyncio.sleep to avoid blocking the event loop
                await asyncio.sleep(BACKOFF_BASE_S * (2 ** attempt))
            else:
                reliability.failures += 1
                print(f"FAILED after {MAX_RETRIES} retries: {e}")
                return None

# Benchmark loop
async def benchmark(pipeline):
    reliability = Reliability()
    print("Measuring reliability...")

    for question in QUESTIONS:
        for _ in range(REPEATS):
            q = question  # capture loop variable explicitly to avoid closure issues
            await call_with_retries(
                lambda q=q: pipeline.invoke(q),
                reliability
            )

    return reliability

# Generate the final report
def generate_report(reliability):
    success_rate = (
        100 * reliability.successes / reliability.calls
        if reliability.calls else 0
    )

    error_rate = (
        100 * reliability.failures / reliability.calls
        if reliability.calls else 0
    )

    retry_rate = (
        100 * reliability.retries / reliability.calls
        if reliability.calls else 0
    )

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

# Entrypoint
async def main():
    reliability = await benchmark(rag_pipeline)
    generate_report(reliability)

if __name__ == "__main__":
    asyncio.run(main())