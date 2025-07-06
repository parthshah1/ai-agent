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
from filecoin_qa.indexing.store import VectorStore

# Default repositories to index
DEFAULT_REPOS = [
    "filecoin-project/lotus",
    "filecoin-project/specs-actors",
    "filecoin-project/FIPs"
]

class FilecoinQAAgent:
    """Agent for answering questions about Filecoin."""

    def __init__(
        self,
        repos: Optional[List[str]] = None,
        github_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        model_name: str = "gpt-4-turbo-preview"
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
            temperature=0.0,
            api_key=self.openai_api_key
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize QA chain only if vector store is loaded
        self.qa_chain = None
        if self.vector_store.store is not None:
            self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create the QA chain with custom prompts."""
        if self.vector_store.store is None:
            raise RuntimeError("Vector store not initialized. Run index_repositories() first.")
            
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
        Given the following conversation and a follow up question, rephrase the follow up question
        to be a standalone question that captures all relevant context from the chat history.

        Chat History:
        {chat_history}

        Follow Up Input: {question}
        Standalone question:""")

        qa_prompt_template = """You are an expert on the Filecoin project and its codebase.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always include references to the source code, documentation, or issues that you used to answer the question.
        You should always use the latest version of the codebase, you can read filecoin specs but those are outdated. FIP's are the most up to date.
        

        Context:
        {context}

        Question: {question}
        
        Answer in markdown format. Include code snippets where relevant, and always link to the source files or issues.
        Helpful Answer:"""

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.store.as_retriever(
                search_kwargs={"k": int(os.getenv("TOP_K", "5"))}
            ),
            memory=self.memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(qa_prompt_template)},
            return_source_documents=True
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
            Dict containing the answer and source documents
        """
        if not self.vector_store.store or not self.qa_chain:
            raise RuntimeError("No indexed documents. Run index_repositories() first.")
        
        result = self.qa_chain({"question": question})
        
        return {
            "answer": result["answer"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        } 