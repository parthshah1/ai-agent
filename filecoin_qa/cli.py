"""Command-line interface for the Filecoin QA tool."""

import os
import sys
from typing import Optional, List
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from .agent import FilecoinQAAgent

app = typer.Typer(
    name="filecoin-qa",
    help="AI-powered assistant for answering questions about Filecoin"
)
console = Console()

def setup_agent(
    repos: Optional[List[str]] = None,
    force_reindex: bool = False
) -> FilecoinQAAgent:
    """Set up and initialize the QA agent.
    
    Args:
        repos: List of repositories to index
        force_reindex: Force reindexing of repositories
    
    Returns:
        Initialized QA agent
    """
    # Check for required environment variables
    if not os.getenv("GITHUB_TOKEN"):
        console.print(
            "[red]Error:[/red] GitHub token not found. Please set GITHUB_TOKEN environment variable."
        )
        sys.exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]Error:[/red] OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )
        sys.exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Initializing QA agent...")
        agent = FilecoinQAAgent(repos=repos)
        
        progress.add_task(description="Indexing repositories...")
        agent.index_repositories(force=force_reindex)
    
    return agent

@app.command()
def ask(
    question: str = typer.Argument(None, help="Question to ask about Filecoin"),
    repos: Optional[List[str]] = typer.Option(
        None,
        "--repo", "-r",
        help="GitHub repository to index (can be specified multiple times)"
    ),
    force_reindex: bool = typer.Option(
        False,
        "--force-reindex",
        help="Force reindexing of repositories"
    )
):
    """Ask a question about Filecoin."""
    agent = setup_agent(repos=repos, force_reindex=force_reindex)
    
    # If no question provided, enter interactive mode
    if not question:
        console.print("\n🤖 [bold]Filecoin QA Assistant[/bold]\n")
        console.print(
            "Ask me anything about Filecoin's code, documentation, or issues.\n"
            "Type 'exit' or press Ctrl+C to quit.\n"
        )
        
        while True:
            try:
                question = Prompt.ask("\n[bold blue]?[/bold blue]")
                if not question:  # Skip empty questions
                    continue
                if question.lower() in ("exit", "quit"):
                    console.print("\nGoodbye! 👋")
                    break
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    progress.add_task(description="Thinking...")
                    result = agent.ask(question)
                
                console.print("\n[bold green]Answer:[/bold green]")
                console.print(Markdown(result["answer"]))
                
                if result["sources"]:
                    console.print("\n[bold]Sources:[/bold]")
                    for source in result["sources"]:
                        url = source["metadata"].get("url", "N/A")
                        console.print(f"- {url}")
                
            except KeyboardInterrupt:
                console.print("\nGoodbye! 👋")
                break
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {str(e)}")
                continue  # Continue the loop even after errors
    
    # Single question mode
    else:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Thinking...")
                result = agent.ask(question)
            
            console.print(Markdown(result["answer"]))
            
            if result["sources"]:
                console.print("\n[bold]Sources:[/bold]")
                for source in result["sources"]:
                    url = source["metadata"].get("url", "N/A")
                    console.print(f"- {url}")
        
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            sys.exit(1)

def main():
    """Entry point for the CLI."""
    app() 