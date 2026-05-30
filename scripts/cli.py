"""CLI commands for pipeline management."""

import os
import sys

from rich.console import Console

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.display import ProgressDisplay
from src.tracking.state import PipelineStateManager, PipelineStatus


console = Console()
display = ProgressDisplay(console)


def cmd_status():
    """Show pipeline progress."""
    state = PipelineStateManager()
    if state.total_emails == 0:
        console.print("[yellow]No pipeline state found. Run 'email-ingest' first.[/yellow]")
        return
    display.show_overview(state)


def cmd_errors():
    """Show failed emails."""
    state = PipelineStateManager()
    display.show_errors(state)


def cmd_resume():
    """Resume a paused/failed pipeline."""
    state = PipelineStateManager()
    if state.total_emails == 0:
        console.print("[yellow]No pipeline state found. Run 'email-ingest' first.[/yellow]")
        return

    current = state.status
    if current == PipelineStatus.RUNNING.value:
        console.print("[yellow]Pipeline is already running.[/yellow]")
        return

    console.print(f"[bold]Resuming pipeline (was: {current})[/bold]")
    display.show_resume_info(state)

    state.set_status(PipelineStatus.RUNNING)
    console.print("[green]Pipeline set to running. Run 'email-preprocess' to continue processing.[/green]")


def cmd_pause():
    """Signal pipeline to pause."""
    state = PipelineStateManager()
    state.set_status(PipelineStatus.PAUSED)
    console.print("[yellow]Pipeline will pause after the current email completes.[/yellow]")


def cmd_reset():
    """Clear all pipeline state and start fresh."""
    state = PipelineStateManager()
    if state.total_emails == 0:
        console.print("[yellow]No pipeline state to reset.[/yellow]")
        return

    # Confirm
    console.print(f"[red]This will clear the state for {state.total_emails} emails.[/red]")
    confirm = input("Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        console.print("Cancelled.")
        return

    state.reset()
    console.print("[green]Pipeline state cleared. Ready for a fresh run.[/green]")


def main():
    """Dispatch CLI commands."""
    commands = {
        "status": cmd_status,
        "errors": cmd_errors,
        "resume": cmd_resume,
        "pause": cmd_pause,
        "reset": cmd_reset,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        console.print("Usage: email-cli <command>")
        console.print("Commands: " + ", ".join(commands.keys()))
        return

    commands[sys.argv[1]]()


if __name__ == "__main__":
    main()
