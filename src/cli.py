"""CLI commands for pipeline management."""

import os
import sys
from rich.console import Console
from rich.prompt import Confirm

from src.tracking.display import ProgressDisplay
from src.tracking.state import PipelineStateManager, PipelineStatus


console = Console()
display = ProgressDisplay(console)


def status():
    """Show pipeline progress."""
    if not os.path.exists("data/pipeline_state.json"):
        console.print("[yellow]No pipeline state found. Run email-ingest first.[/yellow]")
        return

    state = PipelineStateManager()
    display.show_overview(state)

    errors = state.get_failed_emails()
    if errors:
        console.print()
        display.show_errors(state)


def resume():
    """Resume a paused or failed pipeline run."""
    if not os.path.exists("data/pipeline_state.json"):
        console.print("[yellow]No pipeline state found. Run email-ingest first.[/yellow]")
        return

    state = PipelineStateManager()

    if state.status == PipelineStatus.RUNNING.value:
        console.print("[yellow]Pipeline is already running.[/yellow]")
        return

    display.show_resume_info(state)

    if not Confirm.ask("Resume pipeline?"):
        return

    import json
    from datetime import datetime
    from src.preprocessing.pipeline import PreprocessingPipeline

    raw_dir = "data/raw_emails"
    emails = []
    for filename in sorted(os.listdir(raw_dir)):
        if filename.endswith(".json"):
            with open(f"{raw_dir}/{filename}") as f:
                email = json.load(f)
                email["date"] = datetime.fromisoformat(email["date"])
                emails.append(email)

    state.set_status(PipelineStatus.RUNNING)
    pipeline = PreprocessingPipeline(state_manager=state)
    pipeline.run(emails)


def reset():
    """Clear pipeline state and start fresh."""
    if not os.path.exists("data/pipeline_state.json"):
        console.print("[yellow]No pipeline state to reset.[/yellow]")
        return

    state = PipelineStateManager()
    progress = state.get_progress()

    console.print(f"[bold red]This will reset the pipeline state.[/bold red]")
    console.print(f"  Current progress: {progress['overall_pct']}%")
    console.print(f"  Total emails: {progress['total_emails']}")
    console.print()

    if not Confirm.ask("Are you sure? This cannot be undone."):
        return

    state.reset()
    console.print("[green]Pipeline state reset. Run email-ingest to start fresh.[/green]")
