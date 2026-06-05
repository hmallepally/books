"""`sdsd init` — Scaffold the .agent/ directory structure.

Creates the SDSD infrastructure described in Chapter 7, §7.3:
- .agent/prompts/   — feature, bugfix, and refactor templates
- .agent/workflows/ — starter security rules (Threat Blueprints from Ch.3 §3.5)
- .agent/invariants.yaml — global invariant constraints
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


def init_command(
    directory: str = typer.Argument(
        ".",
        help="Target project directory to initialize. Defaults to current directory.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing .agent/ files if they exist.",
    ),
) -> None:
    """Initialize the SDSD .agent/ directory structure in your project.

    Scaffolds prompt templates, security workflows, and global invariants
    following the Repository-as-Context pattern from Ch.4 and Ch.7.
    """
    project_dir = Path(directory).resolve()
    agent_dir = project_dir / ".agent"

    if agent_dir.exists() and not force:
        console.print(
            "[yellow]WARNING: .agent/ directory already exists.[/yellow] "
            "Use [bold]--force[/bold] to overwrite."
        )
        raise typer.Exit(code=1)

    # Create directory structure
    (agent_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (agent_dir / "workflows").mkdir(parents=True, exist_ok=True)

    # Copy built-in templates
    templates_pkg = "sdsd.templates"
    template_files = {
        "feature_template.md": agent_dir / "prompts" / "feature_template.md",
        "bugfix_template.md": agent_dir / "prompts" / "bugfix_template.md",
        "refactor_template.md": agent_dir / "prompts" / "refactor_template.md",
        "security-rules.yaml": agent_dir / "workflows" / "security-rules.yaml",
        "invariants.yaml": agent_dir / "invariants.yaml",
    }

    for src_name, dest_path in template_files.items():
        try:
            content = resources.files(templates_pkg).joinpath(src_name).read_text(encoding="utf-8")
            dest_path.write_text(content, encoding="utf-8")
            console.print(f"  [green][OK][/green] Created {dest_path.relative_to(project_dir)}")
        except Exception as e:
            console.print(f"  [red][FAIL][/red] Failed to create {src_name}: {e}")

    console.print("")
    console.print(Panel.fit(
        "[bold green]SDSD infrastructure initialized.[/bold green]\n\n"
        "Your .agent/ directory is ready. Edit the templates in\n"
        "[cyan].agent/prompts/[/cyan] and security rules in\n"
        "[cyan].agent/workflows/[/cyan] to match your project.\n\n"
        "Next: Run [bold]sdsd prompt create --type feature --target src/[/bold]",
        title="SDSD Ready",
        border_style="green",
    ))
