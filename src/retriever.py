import os, re, glob
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
DESTINATION_DIR = "chroma_store"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def load_vector_store():
    if os.path.exists(DESTINATION_DIR) and os.listdir(DESTINATION_DIR):
        return Chroma(persist_directory=DESTINATION_DIR, embedding_function=embeddings)

    docs = []

    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        
        for line in open(path):
            line = line.strip()
            
            if not line or line == "WEBVTT" or "-->" in line or "<c>" in line or "[MUSIC]" in line:
                continue
            
            lines.append(line)
            
        text = " ".join(lines)

        filename = os.path.basename(path)
        match = re.search(r"\[([A-Za-z0-9_-]+)\]\.en\.vtt$", filename)
        video_id = match.group(1) if match else ""
        title = re.sub(r"\s*\[[A-Za-z0-9_-]+\]\.en\.vtt$", "", filename)

        docs.append(Document(
            page_content=text,
            metadata={"source": path, "title": title, "video_id": video_id}
        ))

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    ).split_documents(docs)
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DESTINATION_DIR)

def build_retriever():
    return load_vector_store().as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    retriever = build_retriever()
    results = retriever.invoke("What is regression testing?")
    print(f"Results: {results}")