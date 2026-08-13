import asyncio, glob, json, random
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

load_dotenv()

client = AsyncOpenAI()

SYSTEM_PROMPT = (
    "You are a dataset curator. Given a call-centre transcript excerpt, "
    "produce exactly one realistic question a caller might ask and an ideal agent answer. "
    "The question must be answerable solely from the excerpt. "
    'Return ONLY valid JSON: {"question": "...", "answer": "..."}'
)

def load_chunks():
    texts = []

    for path in glob.glob("data/*.vtt"):
        with open(path) as f:
            lines = []
            for line in f:
                line = line.strip()
                # mirror retriever.py filter: skip blank, header, timestamps, cue tags, music
                if not line or line == "WEBVTT" or "-->" in line or "<c>" in line or "[MUSIC]" in line:
                    continue
                lines.append(line)
        texts.append(" ".join(lines))

    splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))

async def generate_golden(idx: int, chunk: str) -> dict:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript excerpt:\n{chunk}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "id": f"g{idx:03d}",
        "query": data["question"],
        "expected_output": data["answer"],
        # Ground-truth context: the chunk the Q&A was derived from.
        # ContextualRecall checks expected_output can be attributed to retrieval_context;
        # ContextualPrecision checks retrieved nodes are ranked by relevance.
        "retrieval_context": [chunk],
    }

async def main():
    chunks = load_chunks()
    samples = random.sample(chunks, min(12, len(chunks)))

    tasks = [generate_golden(i, chunk) for i, chunk in enumerate(samples, 1)]
    rows = await asyncio.gather(*tasks)   # all parallel — ~5-10s total

    out_path = "goldens/retriever_golden_dataset.json"
    with open(out_path, "w") as f:
        json.dump(list(rows), f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} goldens -> {out_path}")

if __name__ == "__main__":
    asyncio.run(main())