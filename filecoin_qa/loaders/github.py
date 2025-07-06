"""GitHub repository content loader."""

import os
import logging
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path

from langchain_community.document_loaders import GithubFileLoader, GitHubIssuesLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from github.MainClass import Github

logger = logging.getLogger(__name__)

class FilecoinGitHubLoader:
    """Loads content from Filecoin GitHub repositories."""

    def __init__(
        self,
        repo: str,
        branch: str = "master",
        access_token: Optional[str] = None,
        file_filter: Optional[List[str]] = None,
    ):
        """Initialize the loader.
        
        Args:
            repo: Repository name in format "owner/repo"
            branch: Branch name to load from
            access_token: GitHub personal access token
            file_filter: List of file extensions to load (e.g. [".go", ".md"])
        """
        self.repo = repo
        self.branch = branch
        self.access_token = access_token or os.getenv("GITHUB_TOKEN")
        self.file_filter = file_filter or [".go", ".md", ".ts", ".js", ".py"]
        
        if not self.access_token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN env var.")
        
        # Initialize GitHub client
        self.github_client = Github(self.access_token)

    def _should_load_file(self, path: str) -> bool:
        """Check if file should be loaded based on extension."""
        return any(path.endswith(ext) for ext in self.file_filter)

    def load_files(self) -> List[Document]:
        """Load files from the repository.
        
        Returns:
            List of successfully loaded documents. Files that fail to load are skipped.
        """
        if not self.access_token:
            raise ValueError("GitHub token is required")
        
        documents = []
        try:
            # Get repository and tree
            github_repo = self.github_client.get_repo(self.repo)
            tree = github_repo.get_git_tree(self.branch, recursive=True)
            
            # Process each file in the tree
            for entry in tree.tree:
                if entry.type == "blob" and self._should_load_file(entry.path):
                    try:
                        # Load individual file content
                        file_content = github_repo.get_contents(entry.path, ref=self.branch)
                        if isinstance(file_content, list):
                            continue  # Skip directories
                            
                        doc = Document(
                            page_content=file_content.decoded_content.decode('utf-8'),
                            metadata={
                                "source": entry.path,
                                "file_path": entry.path,
                                "repo": self.repo,
                                "type": "file",
                                "url": f"https://github.com/{self.repo}/blob/{self.branch}/{entry.path}"
                            }
                        )
                        documents.append(doc)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load file {entry.path} from {self.repo}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to access repository {self.repo}: {str(e)}")
            # Don't raise - return any documents we managed to load
            
        return documents

    def load_issues(
        self,
        include_prs: bool = True,
        state: Literal["open", "closed", "all"] = "all"
    ) -> List[Document]:
        """Load issues and optionally PRs from the repository."""
        if not self.access_token:
            raise ValueError("GitHub token is required")
            
        try:
            loader = GitHubIssuesLoader(
                repo=self.repo,
                access_token=self.access_token,
                include_prs=include_prs,
                state=state
            )
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "repo": self.repo,
                    "type": "issue" if not doc.metadata.get("is_pull_request") else "pr"
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load issues from {self.repo}: {str(e)}")
            return []

    def load_all(self) -> List[Document]:
        """Load both files and issues/PRs.
        
        Returns:
            Combined list of all successfully loaded documents.
            If either files or issues fail to load, returns documents from the successful operation.
        """
        documents = []
        
        try:
            documents.extend(self.load_files())
        except Exception as e:
            logger.error(f"Failed to load files from {self.repo}: {str(e)}")
            
        try:
            documents.extend(self.load_issues())
        except Exception as e:
            logger.error(f"Failed to load issues from {self.repo}: {str(e)}")
            
        return documents

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks for embedding.
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents) 