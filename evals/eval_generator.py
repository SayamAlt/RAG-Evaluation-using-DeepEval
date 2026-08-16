from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from src.generator import generate
from dotenv import load_dotenv
import json
from deepeval import evaluate

load_dotenv()

GOLDEN_DATASET_PATH = "goldens/faithfulness_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Load the faithfulness golden dataset (question + golden_context)
with open(GOLDEN_DATASET_PATH) as f:
    golden_dataset = json.load(f)

# Run the generator on the golden dataset, build one test case per record
test_cases = []

for gd in golden_dataset:
    query = gd["question"]
    context = [gd["golden_context"]]
    answer = generate(query, context)

    test_cases.append(
        LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=context
        )
    )

# Key metrics - decomposes actual output into claims and evaluates each claim with reference to context
metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True
    )
]

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "generator_model": "gpt-4o-mini",
        "temperature": 0,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_DATASET_PATH
    }
)