import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from src.rag_pipeline import RAGPipeline
from deepeval.metrics.g_eval import Rubric

load_dotenv()

GOLDEN_DATASET_PATH = "goldens/correctness_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Load the golden dataset
with open(GOLDEN_DATASET_PATH) as f:
    golden_dataset = json.load(f)
    
# Invoke the RAG Pipeline for each query to build a test case from LIVE output
rag_pipeline = RAGPipeline()
test_cases = []

for gd in golden_dataset:
    result = rag_pipeline.invoke(gd["question"]) # RAG Pipeline: retrieve -> rerank -> generate
    
    test_cases.append(
        LLMTestCase(
            input=gd["question"],
            actual_output=result["answer"],
            expected_output=gd["expected_output"]
        )
    )
    
# Define the correctness metric (graded G-Eval: partial credit, not pass/fail)
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output",
        "A claim is wrong ONLY if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
        "When comparing identifiers, codes, account numbers, or formatted sequences (e.g., policy numbers, order IDs), disregard differences in spacing, hyphenation, or punctuation — these are voice-transcription formatting artifacts, not factual errors.",
        "When the expected output describes a sequential or multi-step process, an actual output that correctly states one or more steps from that sequence without contradicting any step has NOT made a factual error — treat the unstated steps as omissions (a completeness concern), not inaccuracies.",
        "A factually accurate answer must score at least 0.9 even if it is shorter, less detailed, or covers fewer points than the expected output — an answer whose stated claims match the expected output without any contradiction scores at least 0.9.",
        "Do NOT deduct for brevity, missing elaboration, fewer examples, or omitted points as omissions are not errors here.",
        "Additional correct information must NEVER lower the score.",
        "When the actual output is a fallback phrase such as 'I do not have enough information in the provided context' but the expected output contains a specific factual answer, treat this as a complete factual failure and assign a score in the (0,4) range.",
        "When the actual output uses a vague or generic term (e.g., 'additional fee', 'extra charge') where the expected output uses a specific named term (e.g., 'data overage charge'), this is an imprecision — NOT a contradiction. Do not score below 7 for this pattern alone, since the generic term does not assert something false. Only score (0,4) when a factually wrong claim is explicitly stated.",
        "Reserve low scores for answers that state something contradictory or factually incorrect."
    ],
    rubric=[
        Rubric(score_range=(9,10), expected_outcome="All stated claims are factually correct and consistent with the expected output. No contradictions exist. Brevity or omission of additional details does not lower this score — a correct, concise subset of the expected facts still qualifies."),
        Rubric(score_range=(5,8), expected_outcome="Contains at least one stated claim that is directly inaccurate, contradicts the expected output, or uses a clearly wrong specific value (e.g., wrong charge type, wrong named item, wrong classification). Omitting information that was not stated does NOT qualify — only wrong statements count."),
        Rubric(score_range=(0,4), expected_outcome="Contains one or more claims that directly and clearly contradict the expected output, or the answer is entirely wrong or opposite to the expected output.")
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False # Graded scale where log probabilities of top k output tokens are normalized to 0-1 scale and then a weighted average is taken of them to compute against a threshold
)

completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points in the expected output, but anchor them to what the question is specifically asking — details in the expected output that are tangential to the question (e.g., date context when the question asks for an amount, reasons when the question asks for an action, supplemental channels when the question asks for a primary action) are supporting details, not key points.",
        "For questions targeting a specific outcome or final step within a multi-step process, treat that outcome as the primary key point; prerequisite or intermediate steps present in the expected output are secondary supporting context.",
        "When the expected output describes a step-by-step procedure, treat the main action steps as key points and their procedural sub-steps (navigation paths, button labels, sub-actions within a step) as supporting details — penalize omission of main steps, not omission of sub-steps.",
        "Check how many of the identified key points are addressed in the actual output.",
        "Penalize the actual output for each primary key point that it omits or ONLY partially covers; do NOT penalize for omitting supporting details.",
        "An actual output that covers all primary key points must score at least 0.7, even if secondary supporting details (timing qualifiers, reasons, supplemental channels, navigation sub-steps) are absent.",
        "For questions that ask how long something will take to be completed, delivered, or credited — the duration or timeline of that specific final action is the SOLE primary key point. Any preceding processing steps present in the expected output (e.g., 'will be processed first, then credited') are secondary supporting context, not primary key points. An answer that correctly states the final timeline must score at least 0.7.",
        "Judge key point coverage only. Do NOT lower the score because a covered point is stated incorrectly — factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output."
    ],
    rubric=[
        Rubric(score_range=(9,10), expected_outcome="Addresses all primary key points and most secondary supporting details from the expected output."),
        Rubric(score_range=(7,8), expected_outcome="Addresses all primary key points (the direct answers to what the question asks) but omits secondary supporting details such as reasons, timing qualifiers, navigation sub-steps, or supplemental channels. An answer that directly answers the question asked always falls in this range even if contextual elaboration is missing."),
        Rubric(score_range=(4,6), expected_outcome="Addresses most but not all primary key points — at least one primary key point is missing or only partially answered."),
        Rubric(score_range=(0,3), expected_outcome="Misses most or all primary key points; the answer fails to address what the question directly asked.")
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False
)

style = GEval(
    name="Style",
    evaluation_steps=[
        "Check that the response answers the question directly without restating or echoing the question back.",
        "Check that the response is concise — it should not pad the answer with filler phrases, hedging language, pleasantries, or unnecessary elaboration beyond what the question requires.",
        "Check that the response is written in complete, grammatically correct sentences. A noun phrase alone is not an acceptable answer.",
        "Check that the response uses a neutral, objective third-person voice consistent with a customer support reporting style (e.g., 'The agent set up...', 'The refund will be...'). It should not be conversational, apologetic, or first-person.",
        "Check that the response uses precise, specific terminology where applicable — it should not substitute vague generic terms (e.g., 'additional fees', 'standard room') when a specific term is clearly implied by the question.",
        "Do NOT penalize the response for being short or for omitting details — brevity is a style strength, not a flaw.",
        "Opening with a nominal paraphrase of the question subject (e.g., 'The [noun from question] was...') is at most a minor deviation — do NOT score below 0.7 for this pattern alone. Reserve scores below 0.7 for responses that fully restate the entire question, use first-person voice, or contain substantial filler or padding."
    ],
    rubric=[
        Rubric(score_range=(9,10), expected_outcome="Clearly in complete sync with the desired style: direct, concise, complete sentence, neutral third-person voice, precise terminology, no question restating, no filler."),
        Rubric(score_range=(5,8), expected_outcome="Mostly in alignment with the desired style but has one minor deviation — e.g., slight hedging, mildly conversational phrasing, or a vague term where a specific one was available."),
        Rubric(score_range=(0,4), expected_outcome="Completely misaligned with the desired style — e.g., restates the question, uses first-person or apologetic tone, contains significant filler or padding, answers with a noun phrase instead of a complete sentence.")
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False
)
# Perform evaluation on the test cases to generate a final report
evaluate(
    test_cases=test_cases,
    metrics=[correctness, completeness, style],
    hyperparameters={
        "retriever": "reranker",
        "embedding_model": "text-embedding-3-large",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "fetch_k": 20,
        "top_k": 10,
        "generator_model": "gpt-4o-mini",
        "temperature": 0,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_DATASET_PATH
    }
)