import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from src.reranker import RerankingRetriever

load_dotenv()

GOLDEN_PATH = "goldens/retriever_golden_dataset.json"
LLM_JUDGE = "gpt-4.1-mini"
THRESHOLD = 0.7

# Load the golden set - the fixed, human-authored truth
with open(GOLDEN_PATH) as f:
    golden_dataset = json.load(f)
    
# Run the retriever on each question to fill retrieval context, then build one test case per golden.
retriever = RerankingRetriever()

test_cases = []

for golden_data in golden_dataset:
    retrieved_docs = retriever.invoke(golden_data["query"])
    retrieval_context = [doc.page_content for doc in retrieved_docs]
    
    test_cases.append(
        LLMTestCase(
            input=golden_data["query"],
            expected_output=golden_data["expected_output"],
            retrieval_context=retrieval_context,
            actual_output="{generator not evaluated in this run}"
        )
    )
    
# Metrics - recall and precision
metrics = [
    ContextualPrecisionMetric(threshold=THRESHOLD, model=LLM_JUDGE, include_reason=True),
    ContextualRecallMetric(threshold=THRESHOLD, model=LLM_JUDGE, include_reason=True)
]

# Evaluate every metric on each test case in a batched & parallel manner to produce a printed report
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "reranker", # vs "reranked" which is an alternative
        "embedding_model": "text-embedding-3-large",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 3,
        "judge_model": LLM_JUDGE,
        "golden_set": GOLDEN_PATH
    }
)