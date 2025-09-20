from typing import List
from src.model.gpt_generator import GPT2Generator
from src.retrieval.retriever import Retriever


class RAGSystem:
    def __init__(self, retriever_model="all-MiniLM-L6-v2", gpt2_model="gpt2"):
        """
        Initialize the RAG system

        Args:
            retriever_model: Name of the retriever model
            gpt2_model: Name of the GPT-2 model
        """
        self.retriever = Retriever(retriever_model)
        self.generator = GPT2Generator(gpt2_model)

    def add_documents(self, documents: List[str]):
        """Add documents to the retrieval system"""
        self.retriever.add_documents(documents)
        print(f"Added {len(documents)} documents to the retriever.")

    def query(self, question: str, top_k: int = 3, max_length: int = 200) -> dict:
        """
        Perform a RAG query

        Args:
            question: User question
            top_k: Number of documents to retrieve
            max_length: Maximum length of generated response

        Returns:
            Dictionary containing retrieved documents and generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.search(question, top_k)

        # Build context
        if retrieved_docs:
            context = "Context:\n" + "\n".join(f"- {doc}" for doc in retrieved_docs)
            context += f"\n\nQuestion: {question}\nAnswer:"
        else:
            context = f"Question: {question}\nAnswer:"

        # Generate response
        response = self.generator.generate_response(context, max_length)

        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "response": response,
            "context": context,
        }
