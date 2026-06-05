"""`sdsd validate` — Check SDSD readiness of a project.

Runs a programmatic version of the 15-Point Checklist from
Chapter 1, §1.5 and the 5-Pillar check from Chapter 7, §7.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from sdsd.core.context_crawler import crawl_target, detect_language
from sdsd.core.schema_reader import find_schemas
from sdsd.core.threat_linker import find_agent_dir, load_global_invariants, load_threat_rules
from sdsd.models.spec import ValidationResult

console = Console()


def validate_command(
    target: str = typer.Option(
        ".", "--target", "-T",
        help="Target directory or file to validate.",
    ),
) -> None:
    """Validate SDSD readiness of a project or target path.

    Checks the 5 Pillars (Ch.7 §7.1) and reports which SDSD
    infrastructure is in place and what is missing.
    """
    target_path = Path(target).resolve()

    if not target_path.exists():
        console.print(f"[red]ERROR: Target path does not exist:[/red] {target}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]SDSD Readiness Check[/bold] -- {target_path}\n")

    results: list[ValidationResult] = []

    # ----- Check 1: .agent/ directory exists -----
    agent_dir = find_agent_dir(target_path)
    if agent_dir:
        results.append(ValidationResult(
            pillar="Repository-as-Context",
            status="pass",
            message=f".agent/ found at {agent_dir}",
        ))
    else:
        results.append(ValidationResult(
            pillar="Repository-as-Context",
            status="fail",
            message="No .agent/ directory found. Run `sdsd init`.",
        ))

    # ----- Check 2: Prompt templates exist -----
    if agent_dir:
        prompts_dir = agent_dir / "prompts"
        templates = list(prompts_dir.glob("*_template.md")) if prompts_dir.is_dir() else []
        if templates:
            results.append(ValidationResult(
                pillar="Prompt Templates",
                status="pass",
                message=f"{len(templates)} templates found in .agent/prompts/",
            ))
        else:
            results.append(ValidationResult(
                pillar="Prompt Templates",
                status="fail",
                message="No *_template.md files in .agent/prompts/",
            ))
    else:
        results.append(ValidationResult(
            pillar="Prompt Templates",
            status="fail",
            message="Cannot check templates — no .agent/ directory",
        ))

    # ----- Check 3: Threat models exist -----
    if agent_dir:
        rules = load_threat_rules(agent_dir)
        if rules:
            results.append(ValidationResult(
                pillar="Threat Models",
                status="pass",
                message=f"{len(rules)} threat rules defined in .agent/workflows/",
            ))
        else:
            results.append(ValidationResult(
                pillar="Threat Models",
                status="warn",
                message="No threat rules found in .agent/workflows/. Add security-rules.yaml.",
            ))
    else:
        results.append(ValidationResult(
            pillar="Threat Models",
            status="fail",
            message="Cannot check threat models — no .agent/ directory",
        ))

    # ----- Check 4: Global invariants exist -----
    if agent_dir:
        invariants = load_global_invariants(agent_dir)
        if invariants:
            results.append(ValidationResult(
                pillar="Global Invariants",
                status="pass",
                message=f"{len(invariants)} global invariants defined",
            ))
        else:
            results.append(ValidationResult(
                pillar="Global Invariants",
                status="warn",
                message="No global invariants in .agent/invariants.yaml",
            ))
    else:
        results.append(ValidationResult(
            pillar="Global Invariants",
            status="fail",
            message="Cannot check invariants — no .agent/ directory",
        ))

    # ----- Check 5: IDE rule files -----
    scan_root = target_path if target_path.is_dir() else target_path.parent
    # Walk up to project root
    project_root = agent_dir.parent if agent_dir else scan_root
    ide_files = {
        ".github/copilot-instructions.md": "GitHub Copilot",
        ".cursorrules": "Cursor",
        ".cursor/rules": "Cursor MDC",
        ".windsurfrules": "Windsurf",
    }
    found_ide = []
    for path, name in ide_files.items():
        if (project_root / path).exists():
            found_ide.append(name)

    if found_ide:
        results.append(ValidationResult(
            pillar="IDE Invariant Walls",
            status="pass",
            message=f"Found rules for: {', '.join(found_ide)}",
        ))
    else:
        results.append(ValidationResult(
            pillar="IDE Invariant Walls",
            status="warn",
            message="No IDE rule files found (.cursorrules, copilot-instructions.md, etc.)",
        ))

    # ----- Check 6: Test coverage -----
    test_dirs = []
    for name in ("tests", "test", "spec"):
        test_dir = project_root / name
        if test_dir.is_dir():
            test_files = list(test_dir.rglob("test_*.*")) + list(test_dir.rglob("*_test.*"))
            test_dirs.extend(test_files)

    if test_dirs:
        results.append(ValidationResult(
            pillar="Adversarial Tests",
            status="pass",
            message=f"{len(test_dirs)} test files found",
        ))
    else:
        results.append(ValidationResult(
            pillar="Adversarial Tests",
            status="warn",
            message="No test files found in tests/ directory",
        ))

    # ----- Render results -----
    table = Table(title="SDSD Readiness Report", show_header=True, header_style="bold")
    table.add_column("Status", width=6, justify="center")
    table.add_column("Pillar", min_width=24)
    table.add_column("Details")

    pass_count = 0
    for r in results:
        if r.status == "pass":
            icon = "[green][OK][/green]"
            pass_count += 1
        elif r.status == "warn":
            icon = "[yellow][!!][/yellow]"
        else:
            icon = "[red][XX][/red]"

        table.add_row(icon, r.pillar, r.message)

    console.print(table)
    console.print("")

    total = len(results)
    fail_count = sum(1 for r in results if r.status == "fail")
    if fail_count == 0:
        console.print(
            f"[bold green]Result: READY[/bold green] — "
            f"{pass_count}/{total} checks passed"
        )
    else:
        console.print(
            f"[bold red]Result: NOT READY[/bold red] — "
            f"{fail_count}/{total} checks failed"
        )
        raise typer.Exit(code=1)
