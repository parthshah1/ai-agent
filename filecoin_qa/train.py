"""Script to index repositories and evaluate the QA system."""

import os
import time
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .agent import FilecoinQAAgent
from .indexing.store import SourceType

app = typer.Typer()
console = Console()

# Test questions with expected source types and key points
TEST_QUESTIONS = [
    {
        "question": "What is the current maximum sector size in Filecoin?",
        "expected_sources": [SourceType.FIP, SourceType.CODE],
        "key_points": ["sector size", "maximum", "current value"]
    },
    {
        "question": "How does WindowPost verification work?",
        "expected_sources": [SourceType.CODE, SourceType.SPEC],
        "key_points": ["WindowPost", "verification", "proof"]
    },
    {
        "question": "What changes did FIP-0045 introduce?",
        "expected_sources": [SourceType.FIP],
        "key_points": ["FIP-0045", "changes", "implementation"]
    },
    {
        "question": "How are storage deals implemented in Lotus?",
        "expected_sources": [SourceType.CODE],
        "key_points": ["storage deal", "implementation", "Lotus"]
    },
    {
        "question": "What was the rationale for changing the proof system?",
        "expected_sources": [SourceType.FIP, SourceType.ISSUE],
        "key_points": ["proof system", "rationale", "change"]
    }
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

def evaluate_source_coverage(
    result: Dict[str, Any],
    expected_sources: List[SourceType]
) -> Dict[str, Any]:
    """Evaluate if the answer uses expected source types.
    
    Args:
        result: QA result containing answer and sources
        expected_sources: List of expected source types
        
    Returns:
        Evaluation metrics
    """
    found_sources = set()
    for source in result["sources"]:
        source_type = source["metadata"].get("source_type")
        if source_type:
            found_sources.add(source_type)
    
    expected = set(s.name for s in expected_sources)
    found = set(str(s) for s in found_sources)
    
    return {
        "expected_sources": list(expected),
        "found_sources": list(found),
        "missing_sources": list(expected - found),
        "extra_sources": list(found - expected),
        "coverage_score": len(found & expected) / len(expected) if expected else 1.0
    }

def evaluate_key_points(
    result: Dict[str, Any],
    key_points: List[str]
) -> Dict[str, Any]:
    """Evaluate if the answer covers expected key points.
    
    Args:
        result: QA result containing answer
        key_points: List of key points to check for
        
    Returns:
        Evaluation metrics
    """
    answer_text = result["answer"].lower()
    found_points = []
    missing_points = []
    
    for point in key_points:
        if point.lower() in answer_text:
            found_points.append(point)
        else:
            missing_points.append(point)
    
    return {
        "total_points": len(key_points),
        "found_points": found_points,
        "missing_points": missing_points,
        "coverage_score": len(found_points) / len(key_points) if key_points else 1.0
    }

def evaluate_source_weights(result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate if sources are properly weighted.
    
    Args:
        result: QA result containing sources with weights
        
    Returns:
        Weight analysis metrics
    """
    weights = {}
    for source in result["sources"]:
        source_type = source["metadata"].get("source_type", "OTHER")
        weight = source["metadata"].get("source_weight", 0.0)
        weights[source_type] = weights.get(source_type, []) + [weight]
    
    # Calculate average weight per source type
    avg_weights = {
        st: sum(w) / len(w)
        for st, w in weights.items()
    }
    
    # Check if weights follow our priority rules
    correct_priority = True
    if "FIP" in avg_weights and "SPEC" in avg_weights:
        correct_priority &= avg_weights["FIP"] > avg_weights["SPEC"]
    if "CODE" in avg_weights and "SPEC" in avg_weights:
        correct_priority &= avg_weights["CODE"] > avg_weights["SPEC"]
    
    return {
        "weights_by_source": avg_weights,
        "correct_priority": correct_priority
    }

def run_evaluation(agent: FilecoinQAAgent) -> Dict[str, Any]:
    """Run comprehensive evaluation of the QA system.
    
    Args:
        agent: Initialized QA agent
        
    Returns:
        Evaluation results and metrics
    """
    results = {
        "questions": [],
        "metrics": {
            "total_questions": len(TEST_QUESTIONS),
            "avg_source_coverage": 0.0,
            "avg_key_point_coverage": 0.0,
            "correct_priority_ratio": 0.0,
            "avg_response_time": 0.0
        }
    }
    
    console.print("\n[bold]Running evaluation questions:[/bold]\n")
    
    total_time = 0
    correct_priority_count = 0
    
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        console.print(f"\n[bold blue]Question {i}:[/bold blue] {test_case['question']}")
        
        start_time = time.time()
        result = agent.ask(test_case["question"])
        duration = time.time() - start_time
        total_time += duration
        
        # Evaluate the result
        source_eval = evaluate_source_coverage(result, test_case["expected_sources"])
        points_eval = evaluate_key_points(result, test_case["key_points"])
        weights_eval = evaluate_source_weights(result)
        
        if weights_eval["correct_priority"]:
            correct_priority_count += 1
        
        # Store detailed results
        question_result = {
            "question": test_case["question"],
            "answer": result["answer"],
            "source_evaluation": source_eval,
            "key_points_evaluation": points_eval,
            "weights_evaluation": weights_eval,
            "response_time": duration
        }
        results["questions"].append(question_result)
        
        # Display results
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(result["answer"])
        
        console.print("\n[bold]Evaluation:[/bold]")
        console.print(f"Source coverage: {source_eval['coverage_score']:.2f}")
        console.print(f"Key points coverage: {points_eval['coverage_score']:.2f}")
        console.print(f"Correct source priority: {weights_eval['correct_priority']}")
        console.print(f"\n[dim]Time taken: {duration:.2f} seconds[/dim]")
        console.print("\n" + "-" * 80)
    
    # Calculate aggregate metrics
    results["metrics"].update({
        "avg_source_coverage": sum(q["source_evaluation"]["coverage_score"] for q in results["questions"]) / len(TEST_QUESTIONS),
        "avg_key_point_coverage": sum(q["key_points_evaluation"]["coverage_score"] for q in results["questions"]) / len(TEST_QUESTIONS),
        "correct_priority_ratio": correct_priority_count / len(TEST_QUESTIONS),
        "avg_response_time": total_time / len(TEST_QUESTIONS)
    })
    
    return results

def display_evaluation_summary(results: Dict[str, Any]) -> None:
    """Display a summary table of evaluation results.
    
    Args:
        results: Evaluation results dictionary
    """
    metrics = results["metrics"]
    
    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Questions", str(metrics["total_questions"]))
    table.add_row("Avg Source Coverage", f"{metrics['avg_source_coverage']:.2%}")
    table.add_row("Avg Key Point Coverage", f"{metrics['avg_key_point_coverage']:.2%}")
    table.add_row("Correct Priority Ratio", f"{metrics['correct_priority_ratio']:.2%}")
    table.add_row("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")
    
    console.print("\n")
    console.print(table)

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
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Save evaluation results to JSON file"
    )
):
    """Index repositories and run evaluation."""
    try:
        agent = index_repositories(repos=repos, force=force)
        results = run_evaluation(agent)
        
        display_evaluation_summary(results)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\nResults saved to: {output_path}")
        
        console.print("\n[bold green]✓[/bold green] Evaluation complete!")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 