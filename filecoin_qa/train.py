"""Script to index repositories and test the QA system."""

import os
import time
from typing import List, Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .agent import FilecoinQAAgent

app = typer.Typer()
console = Console()

TEST_QUESTIONS = [
    "How does Lotus handle block validation?",
    "What is the process for sector sealing?",
    "How are storage deals implemented?",
    "What's the purpose of WindowPost?",
    "How does proof verification work?",
]

def index_repositories(
    repos: Optional[List[str]] = None,
    force: bool = False
) -> FilecoinQAAgent:
    """Index the specified repositories.
    
    Args:
        repos: List of repositories to index
        force: Force reindexing even if cache exists
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task = progress.add_task("Initializing QA agent...", total=None)
        agent = FilecoinQAAgent(repos=repos)
        
        progress.update(task, description="Indexing repositories...")
        agent.index_repositories(force=force)
        
        progress.update(task, description="Indexing complete!")
        
    return agent

def run_test_questions(agent: FilecoinQAAgent) -> None:
    """Run a set of test questions through the agent.
    
    Args:
        agent: Initialized QA agent
    """
    console.print("\n[bold]Running test questions:[/bold]\n")
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        console.print(f"\n[bold blue]Question {i}:[/bold blue] {question}")
        
        start_time = time.time()
        result = agent.ask(question)
        duration = time.time() - start_time
        
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(result["answer"])
        
        if result["sources"]:
            console.print("\n[bold]Sources:[/bold]")
            for source in result["sources"]:
                url = source["metadata"].get("url", "N/A")
                console.print(f"- {url}")
        
        console.print(f"\n[dim]Time taken: {duration:.2f} seconds[/dim]")
        console.print("\n" + "-" * 80)

@app.command()
def train(
    repos: Optional[List[str]] = typer.Option(
        None,
        "--repo", "-r",
        help="GitHub repository to index (can be specified multiple times)"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force reindexing of repositories"
    ),
    test: bool = typer.Option(
        True,
        help="Run test questions after indexing"
    )
):
    """Index repositories and optionally run test questions."""
    try:
        agent = index_repositories(repos=repos, force=force)
        
        if test:
            run_test_questions(agent)
        
        console.print("\n[bold green]✓[/bold green] Training complete!")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 