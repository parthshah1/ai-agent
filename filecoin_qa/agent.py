"""Filecoin QA agent implementation."""

import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import SecretStr

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from filecoin_qa.loaders.github import FilecoinGitHubLoader, split_documents
from filecoin_qa.indexing.store import VectorStore, SourceType

# Default repositories to index
DEFAULT_REPOS = [
    "filecoin-project/lotus",  # Reference implementation
    "filecoin-project/builtin-actors",  # Core protocol actors
    "filecoin-project/FIPs",  # Filecoin Improvement Proposals
    "filecoin-project/specs"  # Original specs (may be outdated)
]

class FilecoinQAAgent:
    """Agent for answering questions about Filecoin with source prioritization."""

    def __init__(
        self,
        repos: Optional[List[str]] = None,
        github_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        model_name: str = "gpt-4-turbo-preview"  # Using GPT-4 for better reasoning
    ):
        """Initialize the QA agent.
        
        Args:
            repos: List of repositories to index
            github_token: GitHub personal access token
            openai_api_key: OpenAI API key
            cache_dir: Directory to cache vector store
            model_name: OpenAI model to use
        """
        self.repos = repos or os.getenv("REPOS", ",".join(DEFAULT_REPOS)).split(",")
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.github_token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN env var.")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        # Initialize components
        self.vector_store = VectorStore(cache_dir=cache_dir)
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,  # Keep it factual
            api_key=self.openai_api_key
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize QA chain only if vector store is loaded
        self.qa_chain = None
        if self.vector_store.store is not None:
            self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create the QA chain with source-aware prompts."""
        if self.vector_store.store is None:
            raise RuntimeError("Vector store not initialized. Run index_repositories() first.")
            
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
        Given the following conversation and a follow up question, rephrase the follow up question
        to be a standalone question that captures all relevant context from the chat history.

        Chat History:
        {chat_history}

        Follow Up Input: {question}
        Standalone question:""")

        qa_prompt_template = """You are an expert on the Filecoin project and its protocol and codebase.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        The context comes from different sources with varying authority:
        1. FIPs (Filecoin Improvement Proposals) - Most authoritative for current protocol rules
        2. Code (Implementation) - Source of truth for actual behavior
        3. Specs - May be outdated but provide good background
        4. Issues/PRs - Provide discussion context
        
        If sources conflict, prefer FIPs over specs, and implementation over documentation.
        Always cite your sources using markdown links.
        
        For each fact in your answer, indicate which source supports it.
        If a spec says one thing but a FIP changed it, explain the change.
        
        Context:
        {context}

        Question: {question}
        
        Answer in markdown format. Include code snippets where relevant.
        If this is the first question in a conversation and no specific question is asked,
        respond with a helpful welcome message explaining what kind of questions I can answer about Filecoin.
        
        Helpful Answer:"""

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.store.as_retriever(
                search_kwargs={
                    "k": int(os.getenv("TOP_K", "5")),
                    "min_score": 0.3  # Minimum relevance threshold
                }
            ),
            memory=self.memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(qa_prompt_template)},
            return_source_documents=True,
            return_generated_question=True
        )

    def index_repositories(self, force: bool = False) -> None:
        """Index or update the repository content.
        
        Args:
            force: Force reindexing even if cache exists
        """
        if not force and self.vector_store.load():
            self.qa_chain = self._create_qa_chain()
            return
        
        all_documents = []
        for repo in self.repos:
            loader = FilecoinGitHubLoader(repo, access_token=self.github_token)
            documents = loader.load_all()
            chunks = split_documents(documents)
            all_documents.extend(chunks)
        
        self.vector_store.add_documents(all_documents)
        self.vector_store.save()
        
        # Create QA chain with new vector store
        self.qa_chain = self._create_qa_chain()

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question about Filecoin.
        
        Args:
            question: The question to ask
        
        Returns:
            Dict containing the answer, source documents, and their weights
        """
        if not self.vector_store.store or not self.qa_chain:
            raise RuntimeError("No indexed documents. Run index_repositories() first.")
        
        result = self.qa_chain({"question": question})
        
        # Include source weights in response
        sources = []
        for doc in result["source_documents"]:
            source_type = doc.metadata.get("source_type", "OTHER")
            source_weight = doc.metadata.get("source_weight", SourceType.OTHER.value)
            sources.append({
                "content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "source_type": source_type,
                    "source_weight": source_weight
                }
            })
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "generated_question": result.get("generated_question")
        } 