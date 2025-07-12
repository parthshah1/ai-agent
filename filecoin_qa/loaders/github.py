"""GitHub repository content loader with source-aware chunking."""

import os
import logging
import re
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import GithubFileLoader, GitHubIssuesLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from github.MainClass import Github

logger = logging.getLogger(__name__)

def split_documents(documents: List[Document], chunk_size: int = 1000) -> List[Document]:
    """Split documents into chunks with source-aware splitting.
    
    Args:
        documents: List of documents to split
        chunk_size: Target size for each chunk
        
    Returns:
        List of chunked documents
    """
    # Custom splitters for different file types
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    
    markdown_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n## ", "\n# ", "\n### ", "\n\n", "\n", " "],
        keep_separator=True
    )
    
    chunks = []
    for doc in documents:
        try:
            # Choose splitter based on file type
            is_markdown = doc.metadata.get("file_path", "").endswith((".md", ".mdx"))
            splitter = markdown_splitter if is_markdown else code_splitter
            
            # Split the document
            doc_chunks = splitter.split_documents([doc])
            
            # Enhance chunk metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                    "processed_at": datetime.utcnow().isoformat()
                })
            
            chunks.extend(doc_chunks)
            
        except Exception as e:
            logger.warning(f"Failed to split document {doc.metadata.get('file_path')}: {str(e)}")
            # Keep original document if splitting fails
            chunks.append(doc)
            
    return chunks

class FilecoinGitHubLoader:
    """Loads content from Filecoin GitHub repositories with source type awareness."""

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
        self.file_filter = file_filter or [
            # Code files
            ".go", ".rs", ".ts", ".js", ".py",
            # Documentation
            ".md", ".mdx",
            # Config files that might have comments
            ".toml", ".yaml", ".json"
        ]
        
        if not self.access_token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN env var.")
        
        # Initialize GitHub client
        self.github_client = Github(self.access_token)

    def _should_load_file(self, path: str) -> bool:
        """Check if file should be loaded based on extension and path."""
        # Skip test files and examples unless in FIPs repo
        if not "FIPs" in self.repo and any(x in path.lower() for x in ["/test/", "/tests/", "/example"]):
            return False
            
        return any(path.endswith(ext) for ext in self.file_filter)

    def _extract_fip_metadata(self, content: str, path: str) -> Dict[str, Any]:
        """Extract metadata from FIP document.
        
        Args:
            content: FIP document content
            path: File path
            
        Returns:
            Dictionary of FIP metadata
        """
        metadata = {}
        
        # Try to extract FIP number
        fip_match = re.search(r"FIP-(\d+)", path)
        if fip_match:
            metadata["fip_number"] = int(fip_match.group(1))
            
        # Try to extract status
        status_match = re.search(r"status:\s*(\w+)", content, re.IGNORECASE)
        if status_match:
            metadata["status"] = status_match.group(1).lower()
            
        # Try to extract title
        title_match = re.search(r"title:\s*(.+)$", content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
            
        return metadata

    def load_files(self) -> List[Document]:
        """Load files from the repository with enhanced metadata.
        
        Returns:
            List of successfully loaded documents
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
                            
                        content = file_content.decoded_content.decode('utf-8')
                        
                        # Build base metadata
                        metadata = {
                            "source": entry.path,
                            "file_path": entry.path,
                            "repo": self.repo,
                            "type": "file",
                            "url": f"https://github.com/{self.repo}/blob/{self.branch}/{entry.path}",
                            "last_modified": file_content.last_modified,
                            "size": file_content.size
                        }
                        
                        # Add FIP-specific metadata if applicable
                        if "FIPs" in self.repo and entry.path.endswith(".md"):
                            fip_metadata = self._extract_fip_metadata(content, entry.path)
                            metadata.update(fip_metadata)
                        
                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                        documents.append(doc)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load file {entry.path} from {self.repo}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to access repository {self.repo}: {str(e)}")
            
        return documents

    def load_issues(
        self,
        include_prs: bool = True,
        state: Literal["open", "closed", "all"] = "all",
        labels: Optional[List[str]] = None
    ) -> List[Document]:
        """Load issues and optionally PRs from the repository.
        
        Args:
            include_prs: Whether to include pull requests
            state: Issue state to include
            labels: List of labels to filter by
        """
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
            
            # Add enhanced metadata
            for doc in documents:
                is_pr = doc.metadata.get("is_pull_request", False)
                doc.metadata.update({
                    "repo": self.repo,
                    "type": "pr" if is_pr else "issue",
                    "processed_at": datetime.utcnow().isoformat()
                })
                
                # Filter by labels if specified
                if labels:
                    doc_labels = doc.metadata.get("labels", [])
                    if not any(label in doc_labels for label in labels):
                        continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load issues from {self.repo}: {str(e)}")
            return []

    def load_all(self) -> List[Document]:
        """Load both files and issues/PRs with comprehensive metadata.
        
        Returns:
            Combined list of all successfully loaded documents
        """
        documents = []
        
        try:
            # Load files first
            documents.extend(self.load_files())
        except Exception as e:
            logger.error(f"Failed to load files from {self.repo}: {str(e)}")
            
        try:
            # Load issues with specific labels for FIPs
            if "FIPs" in self.repo:
                documents.extend(self.load_issues(
                    labels=["fip", "Final", "Last Call", "Active"]
                ))
            else:
                # For other repos, focus on merged PRs and important issues
                documents.extend(self.load_issues(
                    state="closed",
                    labels=["merged", "important", "documentation"]
                ))
        except Exception as e:
            logger.error(f"Failed to load issues from {self.repo}: {str(e)}")
            
        return documents 