from src.generator import generate
from src.reranker import RerankingRetriever

class RAGPipeline:
    
    def __init__(self, fetch_k=20, top_k=10):
        # 1 retriever instance which loads vector store and reranker model
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)
        
    def invoke(self, query: str) -> dict:
        # Retrieve: over-fetch then rerank results down to top_k documents
        retrieved_docs = self.retriever.invoke(query)
        retrieved_context = [doc.page_content for doc in retrieved_docs]
        answer = generate(query=query, context=retrieved_context)
        return {
            "query": query,
            "context": retrieved_context,
            "answer": answer
        }
        
if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    result = rag_pipeline.invoke("What should a customer do if there is an outage in their area?")
    print("Query:", result["query"])
    print("Answer:", result["answer"])
    print("\nRetrieved chunks:")
    
    for idx, chunk in enumerate(result["context"]):
        print(f"{idx}: {chunk[:100]}...")