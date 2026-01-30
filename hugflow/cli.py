"""
CLI interface for Hugflow.
"""

import sys
from pathlib import Path

import typer
from rich import print_json
from rich.console import Console
from rich.table import Table

from hugflow.audit import setup_logging
from hugflow.config import Config, get_dataset_spec_from_yaml, get_remove_spec_from_yaml
from hugflow.sync import SyncManager

app = typer.Typer(
    name="hugflow",
    help="Hugflow - GitOps-based declarative automation for Hugging Face datasets",
    add_completion=False,
)

console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    config_file: str = typer.Option(
        ".env",
        "--config", "-c",
        help="Path to configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
):
    """Hugflow - GitOps-based declarative automation for Hugging Face datasets."""
    # Load configuration
    try:
        config = Config()
        if verbose:
            config.logging.level = "DEBUG"

        # Setup logging
        setup_logging(config)

        # Store config in context
        ctx.ensure_object(dict)
        ctx.obj["config"] = config

    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    ctx: typer.Context,
    yaml_file: Path = typer.Argument(
        ...,
        help="Path to YAML file to validate",
        exists=True,
    ),
):
    """Validate a YAML specification file."""
    config: Config = ctx.obj["config"]

    console.print(f"Validating [cyan]{yaml_file}[/cyan]...")

    try:
        # Determine if it's an add or remove request
        if "remove" in str(yaml_file):
            spec = get_remove_spec_from_yaml(yaml_file)
            console.print(f"✅ Valid [green]remove[/green] specification:")
            console.print(f"   HF ID: {spec.hf_id}")
            console.print(f"   Reason: {spec.reason}")
        else:
            spec = get_dataset_spec_from_yaml(yaml_file)
            console.print(f"✅ Valid [green]add[/green] specification:")
            console.print(f"   HF ID: {spec.hf_id}")
            console.print(f"   Description: {spec.description}")
            console.print(f"   Revision: {spec.revision}")
            if spec.subset:
                console.print(f"   Subset: {spec.subset}")
            console.print(f"   Split: {spec.split}")
            console.print(f"   Download mode: {spec.download_mode}")

        console.print("\n[green]✓ YAML file is valid![/green]")

    except Exception as e:
        console.print(f"\n[red]✗ Validation failed:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def status(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json", "-j",
        help="Output as JSON",
    ),
):
    """Show status of all managed datasets."""
    config: Config = ctx.obj["config"]
    sync_manager = SyncManager(config)

    try:
        # Get active datasets
        active_datasets = sync_manager.storage.get_active_datasets()

        # Get storage usage
        storage_usage = sync_manager.storage.get_storage_usage()

        if json_output:
            import json

            print_json(json.dumps({
                "active_datasets": active_datasets,
                "storage_usage": storage_usage,
            }, indent=2))
        else:
            # Print table
            console.print("\n[bold]Active Datasets[/bold]\n")

            if active_datasets:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("HF ID")
                table.add_column("Description")
                table.add_column("Subset/Split")
                table.add_column("Size")
                table.add_column("Added")

                for ds in active_datasets:
                    subset_split = ""
                    if ds.get("subset"):
                        subset_split = f"{ds['subset']}/{ds['split']}"
                    else:
                        subset_split = ds.get("split", "N/A")

                    size_gb = ds.get("size_bytes", 0) / (1024**3)
                    size_str = f"{size_gb:.2f} GB" if size_gb > 0 else "N/A"

                    table.add_row(
                        ds["hf_id"],
                        ds.get("description", "")[:50],
                        subset_split,
                        size_str,
                        ds.get("added_at", "")[:10],
                    )

                console.print(table)
            else:
                console.print("[yellow]No active datasets[/yellow]")

            console.print(f"\n[bold]Storage Usage:[/bold]")
            console.print(f"  Total datasets: {storage_usage['total_datasets']}")
            console.print(f"  Total size: {storage_usage['total_bytes'] / (1024**3):.2f} GB")

    except Exception as e:
        console.print(f"[red]Error getting status:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def sync(
    ctx: typer.Context,
    ci_mode: bool = typer.Option(
        False,
        "--ci-mode",
        help="Running in CI mode (GitHub Actions)",
    ),
    progress_interval: int = typer.Option(
        10000,
        "--progress-interval",
        help="Progress update interval (number of files)",
    ),
):
    """Sync datasets (called by GitHub Actions)."""
    config: Config = ctx.obj["config"]

    if progress_interval:
        config.download.progress_interval = progress_interval

    sync_manager = SyncManager(config)

    try:
        if ci_mode:
            # In CI mode, detect PR files from environment
            pr_number = config.github.repo.split("/")[-1] if config.github.repo else None

            # Get YAML files from PR
            yaml_files = sync_manager.gitops.get_pr_files()

            if not yaml_files:
                console.print("[yellow]No YAML files found in PR[/yellow]")
                raise typer.Exit(0)

            console.print(f"Found [cyan]{len(yaml_files)}[/cyan] YAML file(s) in PR")

            # Process each YAML file
            for yaml_file in yaml_files:
                console.print(f"\nProcessing [cyan]{yaml_file}[/cyan]...")

                if "remove" in str(yaml_file):
                    result = sync_manager.sync_remove(yaml_file, ci_mode=True)
                else:
                    result = sync_manager.sync_add(yaml_file, ci_mode=True)

                if result["status"] == "success":
                    console.print(f"[green]✓ Success[/green]")
                else:
                    console.print(f"[red]✗ Failed[/red]")
                    raise typer.Exit(1)

        else:
            console.print("[red]Error: --ci-mode is required for sync command[/red]")
            console.print("For local testing, use the 'local-sync' command instead")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error during sync:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def local_sync(
    ctx: typer.Context,
    yaml_file: Path = typer.Argument(
        ...,
        help="Path to YAML file to sync",
        exists=True,
    ),
):
    """Sync a single dataset locally (for testing)."""
    config: Config = ctx.obj["config"]
    sync_manager = SyncManager(config)

    console.print(f"Processing [cyan]{yaml_file}[/cyan]...")

    try:
        if "remove" in str(yaml_file):
            console.print("This is a [yellow]remove[/yellow] request")
            result = sync_manager.sync_remove(yaml_file, ci_mode=False)
        else:
            console.print("This is an [green]add[/green] request")
            result = sync_manager.sync_add(yaml_file, ci_mode=False)

        if result["status"] == "success":
            console.print("\n[green]✓ Sync completed successfully![/green]")

            if "storage_path" in result:
                console.print(f"  Location: {result['storage_path']}")

            if "file_count" in result:
                console.print(f"  Files: {result['file_count']:,}")

            if "size_gb" in result:
                console.print(f"  Size: {result['size_gb']:.2f} GB")

            if result.get("skipped"):
                console.print(f"  Reason: {result['reason']}")
        else:
            console.print(f"\n[red]✗ Sync failed:[/red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error during local sync:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def init(
    ctx: typer.Context,
):
    """Initialize Hugflow directories."""
    config: Config = ctx.obj["config"]
    sync_manager = SyncManager(config)

    try:
        console.print("Initializing Hugflow directories...")

        sync_manager.storage.initialize_directories()

        console.print("[green]✓ Initialization complete![/green]")
        console.print(f"  Storage root: {config.storage.root}")
        console.print(f"  State directory: {config.storage.state_dir}")

    except Exception as e:
        console.print(f"[red]Error during initialization:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
