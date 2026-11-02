"""Vector store management for document retrieval."""

import sys
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import FAISS_INDEX_DIR, EMBEDDING_MODEL, RETRIEVAL_K

logger = logging.getLogger(__name__)


def load_vector_store():
    if not FAISS_INDEX_DIR.exists():
        logger.error(f"No index found at '{FAISS_INDEX_DIR}/'")
        sys.exit(
            f"No index found at '{FAISS_INDEX_DIR}/'.\n"
            "Run 'python ingest.py' first to build the database."
        )
    
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Vector store loaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        sys.exit(1)


def search_docs(vector_store, query: str) -> str:
    try:
        results = vector_store.similarity_search(query, k=RETRIEVAL_K)
        
        if not results:
            logger.info(f"No results found for query: {query}")
            return "No relevant information found in the documents."

        formatted = []
        for doc in results:
            source = doc.metadata.get("source_file", "unknown source")
            page = doc.metadata.get("page", "?")
            formatted.append(f"[{source}, page {page}]\n{doc.page_content}")
        
        result = "\n\n---\n\n".join(formatted)
        logger.info(f"Found {len(results)} results for query: {query}")
        return result
        
    except Exception as e:
        logger.error(f"Error during document search: {e}")
        return f"Error searching documents: {str(e)}"
