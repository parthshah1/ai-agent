"""Vector store management for document indexing and retrieval."""

import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class VectorStore:
    """Manages document embeddings and similarity search."""

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the vector store.
        
        Args:
            embedding_model: HuggingFace model name for embeddings
            cache_dir: Directory to cache the vector store
        """
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        # Convert cache_dir to Path object
        if cache_dir is None:
            self.cache_dir = Path.home() / ".filecoin-qa" / "vectorstore"
        else:
            self.cache_dir = Path(cache_dir)
            
        self.embeddings = self._init_embeddings()
        self.store: Optional[FAISS] = None

    def _init_embeddings(self) -> Embeddings:
        """Initialize the embedding model."""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"}
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not self.store:
            self.store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.store.add_documents(documents)

    def save(self) -> None:
        """Save the vector store to disk."""
        if self.store:
            self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(self.cache_dir))

    def load(self) -> bool:
        """Load the vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if self.cache_dir.exists():
            self.store = FAISS.load_local(
                str(self.cache_dir),
                self.embeddings
            )
            return True
        return False

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter for results
        
        Returns:
            List of similar documents
        """
        if not self.store:
            raise RuntimeError("Vector store not initialized. Add documents first.")
        
        return self.store.similarity_search(
            query,
            k=k,
            filter=filter
        ) 