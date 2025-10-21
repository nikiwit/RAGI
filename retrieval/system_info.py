"""
System information management for identity queries.
"""

from typing import Dict
from config import config

class SystemInformation:
    """Manages information about the RAG system itself."""
    
    @classmethod
    def get_system_info(cls) -> Dict[str, str]:
        """
        Get system information dictionary for answering identity questions.
        
        Returns:
            Dictionary mapping question patterns to responses
        """
        return {
            # Basic identity information
            "who are you": "I'm RAGI (Retrieval-Augmented Generation Interface), an AI assistant designed to help you find information from your document collection using advanced semantic search and natural language processing.",
            "what are you": "I'm an AI-powered retrieval system that uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on your documents.",
            "your name": "You can call me RAGI. I'm here to help you find information in your knowledge base.",

            # Technical information about the system
            "model": f"I'm powered by {config.LLM_MODEL_NAME} for generating responses, and I use the {config.EMBEDDING_MODEL_NAME} embedding model to understand and retrieve relevant information from your documents.",
            "llm": f"I'm using {config.LLM_MODEL_NAME} as my language model to generate responses based on information retrieved from your document collection.",
            "embedding": f"I use the {config.EMBEDDING_MODEL_NAME} embedding model to convert text into numerical vectors for semantic search capabilities.",
            "how do you work": "I use Retrieval Augmented Generation (RAG) to find relevant information in your documents and create helpful responses. First, I analyze your question, then search for relevant documents using semantic search, and finally generate a response based on the retrieved information.",
            "technology": f"I'm built using the LangChain framework with {config.LLM_MODEL_NAME} as my language model and {config.EMBEDDING_MODEL_NAME} for embeddings. I use ChromaDB as my vector database to store and retrieve information efficiently.",
            "version": f"I'm running RAGI version 1.1, a streamlined Retrieval Augmented Generation system optimized for general-purpose knowledge retrieval.",

            # Development information
            "who made you": "I'm an open-source RAG system that you can customize for your specific use case.",
            "what can you do": "I can answer questions based on any documents you've added to my knowledge base. I support PDFs, Word documents, text files, Markdown, PowerPoint, and EPUB formats.",
            "your purpose": "My purpose is to help you quickly find accurate information from your document collection using AI-powered semantic search.",
        }
    
    @classmethod
    def get_response_for_identity_query(cls, query: str) -> str:
        """
        Get appropriate response for an identity query.
        
        Args:
            query: The user's query string
            
        Returns:
            Response string appropriate for the identity query
        """
        query_lower = query.lower()
        system_info = cls.get_system_info()
        
        # Check for model-specific questions 
        if any(term in query_lower for term in ["model", "llm", "language model"]):
            return system_info.get("model", system_info.get("llm"))
        
        # Check for embedding/vector questions
        if any(term in query_lower for term in ["embedding", "vector", "semantic"]):
            return system_info.get("embedding")
        
        # Check for technology questions
        if any(term in query_lower for term in ["tech", "technology", "stack", "framework", "built with"]):
            return system_info.get("technology")
        
        # Check for other identity questions
        for key, value in system_info.items():
            if key in query_lower:
                return value
        
        # Default response
        return f"I'm RAGI, a Retrieval-Augmented Generation assistant. I'm powered by {config.LLM_MODEL_NAME} with {config.EMBEDDING_MODEL_NAME} embeddings for retrieval. I help you find information in your document collection using AI-powered semantic search."