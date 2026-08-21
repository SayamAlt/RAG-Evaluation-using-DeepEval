# RAG Evaluation

This project tests how well each part of a RAG system works. It covers the retriever, the generator, and the full pipeline end to end.

The data is 10 call center mock call videos from YouTube, saved as subtitle files. They cover tech support, health insurance, hotel bookings, billing disputes, telco issues, and banking.

![RAG Evaluation](https://www.comet.com/site/wp-content/uploads/2026/02/RAG-Evaluation-1024x576.png)
![RAG Pipeline](https://miro.medium.com/v2/resize:fit:13216/1*cO88iZwmqxLcoNDi-nI4Tg.png)
![RAG Triad](https://assets.zilliz.com/RAG_Triad_639f6454ae.png)

---

## What the project does

You ask a question. The system searches a vector database of call center transcripts to find the most relevant text chunks. A cross-encoder reranker then picks the best ones. A generator reads those chunks and writes a grounded answer. Then metrics tell you how good the retrieval and the answer are.

---

## The five pieces

### 1. Retriever (`src/retriever.py`)

Reads all the subtitle files, splits them into chunks of about 1000 characters, and stores them in ChromaDB using OpenAI embeddings.

Call `build_retriever()` to get a LangChain retriever you can query directly.

### 2. Reranker (`src/reranker.py`)

The retriever is fast but rough. The reranker makes it more accurate.

It works in two steps. First it pulls 10 candidate chunks using cosine similarity. Then it scores each one against the query using a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and keeps the top 5. The cross-encoder reads the query and chunk together, so it catches relevance that cosine similarity misses.

### 3. Generator (`src/generator.py`)

Takes a question and the retrieved chunks, and writes a short grounded answer using `gpt-4o-mini`.

It follows a faithfulness-first prompt. It only uses information from the chunks. If the chunks do not contain enough to answer, it says so rather than guessing.

### 4. RAG Pipeline (`src/rag_pipeline.py`)

Connects the reranker and the generator into one call.

Pass a question to `RAGPipeline().invoke(query)` and get back the query, the retrieved chunks, and the final answer.

### 5. Evaluators (`evals/`)

Three separate eval scripts cover different parts of the system.

**`evals/eval_retriever.py`** runs the reranker on every question in the retriever golden dataset and measures two things:

**Contextual Precision** - are the chunks it returned actually useful for this question?

**Contextual Recall** - do the returned chunks contain enough to answer the question properly?

**`evals/eval_generator.py`** feeds each golden context directly into the generator, skipping the retriever, and measures two things:

**Faithfulness** - does the answer stick to what the context actually says, with no made-up claims?

**Answer Relevancy** - does the answer actually address the question asked?

**`evals/eval_rag_pipeline.py`** runs the full pipeline end to end and measures the RAG triad:

**Contextual Relevancy** - are the retrieved chunks relevant to the question?

**Faithfulness** - is the generated answer grounded in those chunks?

**Answer Relevancy** - does the answer address the question?

**`evals/eval_application.py`** runs application-level evaluation using G-Eval, a framework where an LLM judge scores each answer against a human-curated golden dataset. It measures three things:

**Correctness** - do the stated facts match the expected answer without contradictions? This is graded strictly on truth, not completeness. Omissions are not penalised; only factually wrong or contradictory claims lower the score. Identifier formatting differences from voice transcription (e.g. spacing in policy numbers) are ignored.

**Completeness** - does the answer cover the primary key points the question is asking for? Primary key points are anchored to what the question directly asks. Supporting context (reasons, timing qualifiers, navigation sub-steps) is treated as secondary and not penalised when absent. An answer that covers all primary key points must score at least 0.7.

**Style** - is the answer written the way a customer support report should read? It should be direct, concise, in complete sentences, use a neutral third-person voice, and use precise terminology rather than generic labels. Brevity is a strength, not a flaw.

All three metrics use rubric-based G-Eval with graded scoring (`strict_mode=False`), meaning scores are continuous rather than pass/fail. The judge model is `gpt-4.1-mini` and the pass threshold is 0.7.

All metrics are scored by `gpt-4.1-mini` acting as a judge, with a pass threshold of 0.7.

---

## Golden datasets

`goldens/retriever_golden_dataset.json` has 20 questions for retriever evaluation. Each question came from a real chunk of transcript. For every question, the file stores the ideal answer and the original chunk it came from.

`goldens/golden_dataset_generator.py` made them. It samples random chunks, sends each one to `gpt-4o-mini`, and gets back a question and answer. All 20 run in parallel so it finishes in about 10 seconds.

`goldens/faithfulness_golden_dataset.json` has 20 questions for generator and pipeline evaluation. Each question is paired with the exact chunk it came from, stored as `golden_context`, along with the source `video_id`.

`goldens/faithfulness_dataset_generator.py` made them. It loads all chunks directly from the ChromaDB store, exports them to `goldens/chunks.json`, samples 20, and sends each one to `gpt-4o-mini` to get a question. All 20 run in parallel.

`goldens/correctness_golden_dataset.json` has 20 questions for application-level evaluation. Each question was written by hand against a specific call center transcript, with a human-verified expected answer. Every entry cites the source VTT file and video ID so results can be traced back to the original call. Two questions per source file cover different aspects of the same conversation (e.g. a charge amount and a follow-up action).

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

## Run the evals

Retriever eval:

```bash
python -m evals.eval_retriever
```

Generator eval:

```bash
python -m evals.eval_generator
```

Full RAG pipeline eval:

```bash
python -m evals.eval_rag_pipeline
```

Application-level eval (G-Eval, correctness golden dataset):

```bash
python -m evals.eval_application
```

Each script builds the vector store if one does not exist yet, runs every golden query, and prints a report with scores for each test case.

---

## Project layout

```
data/                                       10 call center subtitle files (.vtt)

src/
  retriever.py                              loads VTTs, builds ChromaDB vector store
  reranker.py                               two-stage retrieval with cross-encoder
  generator.py                              generates grounded answers from retrieved chunks
  rag_pipeline.py                           end-to-end pipeline: retrieve, rerank, generate

goldens/
  retriever_golden_dataset.json             20 question and answer pairs for retriever eval
  golden_dataset_generator.py              generates retriever golden pairs from transcript chunks
  faithfulness_golden_dataset.json          20 questions with source chunks for generator eval
  faithfulness_dataset_generator.py        generates faithfulness golden pairs from ChromaDB
  chunks.json                               all chunks exported from the vector store
  correctness_golden_dataset.json           20 hand-curated Q&A pairs for application-level eval

evals/
  eval_retriever.py                         runs ContextualPrecision and ContextualRecall
  eval_generator.py                         runs Faithfulness and AnswerRelevancy on the generator
  eval_rag_pipeline.py                      runs the RAG triad on the full pipeline
  eval_application.py                       runs G-Eval (Correctness, Completeness, Style) on live pipeline output

deepeval_intro.py                           small demo showing how deepeval works
```

---

## Dependencies

- `langchain-openai` + `langchain-chroma` for the vector store
- `sentence-transformers` for the cross-encoder reranker
- `deepeval` for the evaluation metrics
- `openai` for embeddings and the LLM judge