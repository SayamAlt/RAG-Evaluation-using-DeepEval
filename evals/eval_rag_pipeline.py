import json
from src.rag_pipeline import RAGPipeline
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)
from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = "goldens/faithfulness_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Load queries
with open(GOLDEN_DATASET_PATH) as f:
    golden_dataset = json.load(f)
    
# Run the full RAG Pipeline for each query to build a test case from LIVE output
rag_pipeline = RAGPipeline()
test_cases = []

for gd in golden_dataset:
    result = rag_pipeline.invoke(gd["question"]) # RAG Pipeline: Retrieve -> Rerank -> Generate

    test_cases.append(
        LLMTestCase(
            input=gd["question"],
            actual_output=result["answer"],
            retrieval_context=result["context"]
        )
    )
    
# RAG Traid Metrics
metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
]

# Evaluate the RAG pipeline
evaluate(
    test_cases=test_cases, 
    metrics=metrics,
    hyperparameters={
        "retriever": "reranker",
        "embedding_model": "text-embedding-3-large",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "fetch_k": 10,
        "top_k": 5,
        "generator_model": "gpt-4o-mini",
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_DATASET_PATH,
    }
)