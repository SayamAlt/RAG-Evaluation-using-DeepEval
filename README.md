# RAG Evaluation

This project tests how well a retriever finds the right chunks of text when you ask it a question.

The data is 10 call center mock call videos from YouTube, saved as subtitle files. They cover tech support, health insurance, hotel bookings, billing disputes, telco issues, and banking.

---

## What the project does

You ask a question. The system searches a vector database of call center transcripts to find the most relevant text chunks. Then two metrics tell you how good those chunks are.

That's it.

---

## The three pieces

### 1. Retriever (`src/retriever.py`)

Reads all the subtitle files, splits them into chunks of about 1000 characters, and stores them in ChromaDB using OpenAI embeddings.

Call `build_retriever()` to get a LangChain retriever you can query directly.

### 2. Reranker (`src/reranker.py`)

The retriever is fast but rough. The reranker makes it more accurate.

It works in two steps. First it pulls 10 candidate chunks using cosine similarity. Then it scores each one against the query using a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and keeps the top 5. The cross-encoder reads the query and chunk together, so it catches relevance that cosine similarity misses.

### 3. Evaluator (`evals/eval_retriever.py`)

Runs the reranker on every question in the golden dataset and measures two things:

**Contextual Precision** - are the chunks it returned actually useful for this question?

**Contextual Recall** - do the returned chunks contain enough to answer the question properly?

Both are scored by `gpt-4.1-mini` acting as a judge, with a pass threshold of 0.7.

---

## Golden dataset

`goldens/retriever_golden_dataset.json` has 20 questions. Each question came from a real chunk of transcript. For every question, the file stores the ideal answer and the original chunk it came from.

`goldens/golden_dataset_generator.py` made them. It samples random chunks, sends each one to `gpt-4o-mini`, and gets back a question and answer. All 20 run in parallel so it finishes in about 10 seconds.

---

## Setup

Add your OpenAI key to a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

Install dependencies:

```bash
uv sync
```

---

## Run the eval

```bash
python evals/eval_retriever.py
```

This builds the vector store (or reuses an existing one in `chroma_store/`), runs every golden query through the reranker, and prints a report with precision and recall scores for each test case.

---

## Project layout

```
data/                                       10 call center subtitle files (.vtt)

src/
  retriever.py                              loads VTTs, builds ChromaDB vector store
  reranker.py                               two-stage retrieval with cross-encoder

goldens/
  retriever_golden_dataset.json             20 question and answer pairs with source context
  golden_dataset_generator.py              generates new golden pairs from the transcript chunks

evals/
  eval_retriever.py                         runs ContextualPrecision and ContextualRecall

deepeval_intro.py                           small demo showing how deepeval works
```

---

## Dependencies

- `langchain-openai` + `langchain-chroma` for the vector store
- `sentence-transformers` for the cross-encoder reranker
- `deepeval` for the evaluation metrics
- `openai` for embeddings and the LLM judge