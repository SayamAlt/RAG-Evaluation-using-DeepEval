import asyncio, json, random
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

client = AsyncOpenAI()

CHROMA_DIR = "chroma_store"
CHUNKS_PATH = "goldens/chunks.json"
OUTPUT_PATH = "goldens/faithfulness_golden_dataset.json"
N_SAMPLES = 20

SYSTEM_PROMPT = (
    "You are a dataset curator. Given a call-centre transcript excerpt, "
    "produce exactly one realistic question that is directly and fully answerable from the excerpt alone. "
    'Return ONLY valid JSON: {"question": "..."}'
)


def export_chunks() -> list[dict]:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    result = store._collection.get(include=["documents", "metadatas"])
    chunks = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(chunks)} chunks -> {CHUNKS_PATH}")
    return chunks


async def generate_question(idx: int, chunk_text: str, video_id: str) -> dict:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript excerpt:\n{chunk_text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "id": f"f{idx:03d}",
        "question": data["question"],
        "golden_context": chunk_text,
        "video_id": video_id,
    }


async def main():
    chunks = export_chunks()
    samples = random.sample(chunks, min(N_SAMPLES, len(chunks)))
    tasks = [generate_question(i, c["text"], c["metadata"].get("video_id", "")) for i, c in enumerate(samples, 1)]
    rows = await asyncio.gather(*tasks)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(list(rows), f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} faithfulness goldens -> {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
