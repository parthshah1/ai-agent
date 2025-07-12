"""Vector store management for document indexing and retrieval with source prioritization."""

import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class SourceType(Enum):
    """Enum for different source types with their priority weights."""
    FIP = 1.0  # Highest priority - current protocol state
    CODE = 0.9  # Implementation truth
    SPEC = 0.7  # May be outdated
    ISSUE = 0.6  # Discussion context
    OTHER = 0.5  # Default weight

class VectorStore:
    """Manages document embeddings and similarity search with source prioritization."""

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

    def _get_source_type(self, metadata: Dict[str, Any]) -> SourceType:
        """Determine source type and priority from document metadata.
        
        Args:
            metadata: Document metadata containing source info
            
        Returns:
            SourceType enum indicating priority
        """
        path = metadata.get("file_path", "").lower()
        doc_type = metadata.get("type", "").lower()
        
        # Check for FIPs
        if "fip" in path or "fips" in path:
            return SourceType.FIP
            
        # Check for code files
        if any(path.endswith(ext) for ext in [".go", ".rs", ".ts", ".js", ".py"]):
            return SourceType.CODE
            
        # Check for spec files
        if "spec" in path or "specs" in path:
            return SourceType.SPEC
            
        # Check for issues/PRs
        if doc_type in ["issue", "pr"]:
            return SourceType.ISSUE
            
        return SourceType.OTHER

    def _prepare_document(self, document: Document) -> Document:
        """Prepare document by adding source type and weight metadata.
        
        Args:
            document: Input document
            
        Returns:
            Document with enhanced metadata
        """
        source_type = self._get_source_type(document.metadata)
        document.metadata.update({
            "source_type": source_type.name,
            "source_weight": source_type.value,
            "indexed_at": datetime.utcnow().isoformat()
        })
        return document

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store with source prioritization.
        
        Args:
            documents: List of documents to add
        """
        # Prepare documents with source weights
        prepared_docs = [self._prepare_document(doc) for doc in documents]
        
        if not self.store:
            self.store = FAISS.from_documents(prepared_docs, self.embeddings)
        else:
            self.store.add_documents(prepared_docs)

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
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        return False

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[Document]:
        """Search for similar documents with source prioritization.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter for results
            min_score: Minimum similarity score threshold
            
        Returns:
            List of similar documents, ordered by weighted relevance
        """
        if not self.store:
            raise RuntimeError("Vector store not initialized. Add documents first.")
        
        # Get more results than needed to allow for weighting
        raw_results = self.store.similarity_search_with_score(
            query,
            k=k * 2,  # Get extra results for filtering
            filter=filter
        )
        
        # Apply source weights to scores
        weighted_results = [
            (doc, score * doc.metadata.get("source_weight", SourceType.OTHER.value))
            for doc, score in raw_results
        ]
        
        # Filter by minimum score and sort by weighted score
        filtered_results = [
            doc for doc, score in weighted_results
            if score >= min_score
        ]
        
        # Return top k results after weighting
        return filtered_results[:k] 