import time, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langsmith import Client
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)

load_dotenv()

PROJECT = "RAG Evaluation"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7
SAMPLE_RATE = 1
POLL_SECONDS = 60

METRIC_DATASET = {
    "faithfulness":         "rag-triad",
    "answer_relevancy":     "application-level",
    "contextual_relevancy": "rag-triad",
}

client = Client()

def _sampled(run):
    return hash(str(run.id)) % 100 < SAMPLE_RATE * 100

def _existing_keys(run):
    fb = client.list_feedback(run_ids=[run.id])
    return {f.key for f in fb}

def _get_or_create_dataset(name):
    existing = list(client.list_datasets(dataset_name=name))
    if existing:
        return existing[0]
    return client.create_dataset(name, description=f"RAG pipeline golden dataset: {name}")

def _flag_to_dataset(run, key, query, answer, context, score, reason):
    dataset_name = METRIC_DATASET.get(key)
    if not dataset_name:
        return
    try:
        dataset = _get_or_create_dataset(dataset_name)
        inputs = {"query": query}
        if context:
            inputs["context"] = context
        outputs = {"actual_output": answer}
        client.create_examples(
            inputs=[inputs],
            outputs=[outputs],
            metadata=[{
                "source_run_id": str(run.id),
                "metric": key,
                "score": score,
                "reason": reason,
                "flagged": "bad_performance",
            }],
            dataset_id=dataset.id,
        )
        print(f"  [flag] {key}={score:.3f} → added to dataset '{dataset_name}'")
    except Exception as e:
        print(f"  [flag] failed to add to dataset '{dataset_name}': {e}")

def score_recent_traces():
    runs = client.list_runs(
        project_name=PROJECT,
        is_root=True,
        run_type="chain"
    )
    for run in runs:
        if not _sampled(run):
            continue
        outputs = run.outputs or {}
        answer = outputs.get("answer")
        context = outputs.get("context")
        query = (run.inputs or {}).get("query", "")
        if not answer or not context:
            continue
        already = _existing_keys(run)
        jobs = [
            (
                "faithfulness",
                FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
                dict(input=query, actual_output=answer, retrieval_context=context)
            ),
            (
                "answer_relevancy",
                AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
                dict(input=query, actual_output=answer)
            ),
            (
                "contextual_relevancy",
                ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
                dict(input=query, actual_output=answer, retrieval_context=context)
            )
        ]
        for key, metric, tc_kwargs in jobs:
            if key in already:
                continue
            try:
                metric.measure(LLMTestCase(**tc_kwargs))
                client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=metric.score,
                    comment=metric.reason
                )
                if metric.score < THRESHOLD:
                    _flag_to_dataset(run, key, query, answer, context, metric.score, metric.reason)
            except Exception as e:
                print(f"[{key}] failed on run {run.id}: {e}")

if __name__ == "__main__":
    while True:
        score_recent_traces()
        time.sleep(POLL_SECONDS)