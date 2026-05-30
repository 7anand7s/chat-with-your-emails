"""Rich-based progress display for the pipeline."""

from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.tracking.state import PipelineStateManager, STAGE_ORDER


# Stage display names and colors
STAGE_DISPLAY = {
    "fetched": ("Fetched", "cyan"),
    "cleaned": ("Cleaned", "blue"),
    "llm_extracted": ("LLM Extracted", "yellow"),
    "vlm_processed": ("VLM Processed", "magenta"),
    "embedded": ("Embedded", "green"),
    "stored": ("Stored", "bold green"),
}

STATUS_COLORS = {
    "running": "yellow",
    "paused": "cyan",
    "completed": "green",
    "failed": "red",
}


class ProgressDisplay:
    """Display pipeline progress using Rich."""

    def __init__(self, console: Console = None):
        self.console = console or Console()

    def show_overview(self, state: PipelineStateManager):
        """Show a full progress overview table."""
        progress = state.get_progress()
        total = progress["total_emails"]
        status = progress["status"]
        status_color = STATUS_COLORS.get(status, "white")

        # Build the progress table
        table = Table(title="Pipeline Progress", show_lines=True)
        table.add_column("Stage", style="bold", min_width=18)
        table.add_column("Completed", justify="right", min_width=12)
        table.add_column("Failed", justify="right", min_width=10)
        table.add_column("Remaining", justify="right", min_width=12)
        table.add_column("Progress", min_width=20)

        stages_data = progress["stages"]

        for stage_key, (display_name, color) in STAGE_DISPLAY.items():
            stage_info = stages_data.get(stage_key, {"completed": 0, "failed": 0})
            completed = stage_info["completed"]
            failed = stage_info["failed"]
            remaining = max(0, total - completed - failed)

            # Progress bar
            if total > 0:
                pct = completed / total
                bar_width = 20
                filled = int(bar_width * pct)
                bar = f"[{color}]{'█' * filled}[/{color}]{'░' * (bar_width - filled)}"
                pct_str = f" {pct:.0%}"
            else:
                bar = "░" * 20
                pct_str = " 0%"

            completed_str = f"[{color}]{completed}[/{color}]" if completed > 0 else "0"
            failed_str = f"[red]{failed}[/red]" if failed > 0 else "0"
            remaining_str = str(remaining) if remaining > 0 else "[dim]-[/dim]"

            table.add_row(display_name, completed_str, failed_str, remaining_str, bar + pct_str)

        # Summary panel
        overall_pct = progress["overall_pct"]
        total_errors = progress["total_errors"]
        started = progress.get("started_at", "")
        updated = progress.get("updated_at", "")

        summary_lines = [
            f"[bold {status_color}]Status: {status.upper()}[/bold {status_color}]",
            f"Total emails: {total}",
            f"Overall progress: [bold]{overall_pct}%[/bold]",
        ]
        if total_errors > 0:
            summary_lines.append(f"[red]Errors: {total_errors} emails[/red]")
        if started:
            summary_lines.append(f"Started: {started[:19]}")
        if updated:
            summary_lines.append(f"Updated: {updated[:19]}")

        self.console.print()
        self.console.print(Panel("\n".join(summary_lines), title="Pipeline Summary"))
        self.console.print(table)
        self.console.print()

    def show_current(self, subject: str, sender: str, stage: str, index: int, total: int):
        """Show current email being processed."""
        display_name, color = STAGE_DISPLAY.get(stage, (stage, "white"))
        self.console.print(
            f"  [{color}][{index}/{total}][/{color}] {display_name}: "
            f"[bold]{subject[:50]}[/bold] — {sender[:30]}"
        )

    def show_errors(self, state: PipelineStateManager):
        """Show a table of failed emails."""
        errors = state.get_failed_emails()

        if not errors:
            self.console.print("[green]No errors![/green]")
            return

        table = Table(title=f"Failed Emails ({len(errors)})", show_lines=True)
        table.add_column("Subject", min_width=30)
        table.add_column("Sender", min_width=20)
        table.add_column("Stage", min_width=15)
        table.add_column("Error", min_width=40)
        table.add_column("Retries", justify="right")

        for err in errors:
            table.add_row(
                err["subject"][:50],
                err["sender"][:30],
                err["stage"],
                err["error"][:80] if err["error"] else "",
                str(err["retries"]),
            )

        self.console.print(table)

    def show_resume_info(self, state: PipelineStateManager):
        """Show what will be resumed."""
        progress = state.get_progress()
        total = progress["total_emails"]
        stages = progress["stages"]

        stored = stages.get("stored", {}).get("completed", 0)
        remaining = total - stored

        if remaining <= 0:
            self.console.print("[green]All emails already processed![/green]")
            return

        self.console.print(f"\n[bold]Resuming pipeline[/bold]")
        self.console.print(f"  Already completed: [green]{stored}[/green] emails")
        self.console.print(f"  Remaining: [yellow]{remaining}[/yellow] emails")

        # Show breakdown by stage where work is needed
        for stage_key in STAGE_ORDER[:-1]:  # Exclude stored
            display_name, color = STAGE_DISPLAY.get(stage_key, (stage_key, "white"))
            failed = stages.get(stage_key, {}).get("failed", 0)
            if failed > 0:
                self.console.print(f"  [{color}]{display_name}[/{color}]: [red]{failed} failed[/red] (will retry)")

        self.console.print()
