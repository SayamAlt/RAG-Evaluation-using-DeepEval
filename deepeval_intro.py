from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv()

# Test cases
tc1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris."
)

tc2 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine."
)

# One metric, judged by an LLM (pinned for reproducibility)
metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True)

# Run both test cases through the metric with a printed report
evaluate(test_cases=[tc1, tc2], metrics=[metric])