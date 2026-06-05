"""`sdsd prompt create` — Assemble a complete SDSD prompt.

This is the core command described in Chapter 7, §7.2. It:
1. Loads the template from .agent/prompts/
2. Crawls the target for function/class signatures (Dependency Crawling)
3. Links threat models from .agent/workflows/ (Threat Model Linking)
4. Reads database schemas from migration directories (Schema Retrieval)
5. Interactively collects the engineer's Goal, State Machine, and custom Invariants
6. Assembles the full prompt and copies it to clipboard
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from sdsd.core.context_crawler import crawl_target, detect_language
from sdsd.core.schema_reader import find_schemas
from sdsd.core.template_engine import load_template, render_prompt
from sdsd.core.threat_linker import (
    find_agent_dir,
    link_threats_to_target,
    load_global_invariants,
    load_threat_rules,
)
from sdsd.models.spec import (
    BlastRadius,
    SDSDSpec,
    StateTransition,
    TemplateType,
)

console = Console()

prompt_app = typer.Typer(help="Prompt assembly commands.")


@prompt_app.command("create")
def create_command(
    template_type: TemplateType = typer.Option(
        ..., "--type", "-t",
        help="Template type: feature, bugfix, or refactor.",
    ),
    target: str = typer.Option(
        ..., "--target", "-T",
        help="Target file or directory to generate a prompt for.",
    ),
    goal: Optional[str] = typer.Option(
        None, "--goal", "-g",
        help="The deterministic goal statement. If omitted, will prompt interactively.",
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d",
        help="Domain hint for threat model matching (e.g., 'payment', 'auth').",
    ),
    clipboard: bool = typer.Option(
        True, "--clipboard/--no-clipboard",
        help="Copy the assembled prompt to clipboard.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Write the assembled prompt to a file.",
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive",
        help="Skip interactive prompts (for CI/scripting).",
    ),
) -> None:
    """Assemble a complete SDSD prompt with dynamic context injection.

    This command implements the full pipeline from Ch.7, §7.2:
    Dependency Crawling → Schema Retrieval → Threat Model Linking → Assembly.
    """
    target_path = Path(target).resolve()

    if not target_path.exists():
        console.print(f"[red]ERROR: Target path does not exist:[/red] {target}")
        raise typer.Exit(code=1)

    console.print(Panel.fit(
        f"[bold]Assembling SDSD Prompt[/bold]\n"
        f"Template: [cyan]{template_type.value}[/cyan]\n"
        f"Target: [cyan]{target}[/cyan]",
        title="sdsd-cli",
        border_style="blue",
    ))

    # Step 1: Find .agent/ directory
    agent_dir = find_agent_dir(target_path)
    if agent_dir:
        console.print(f"  [green][OK][/green] Found .agent/ at {agent_dir}")
    else:
        console.print(
            "  [yellow]WARNING: No .agent/ directory found.[/yellow] "
            "Run [bold]sdsd init[/bold] first. Proceeding without templates."
        )

    # Step 2: Detect language
    language = detect_language(target_path)
    console.print(f"  [green][OK][/green] Detected language: [cyan]{language}[/cyan]")

    # Step 3: Dependency Crawling (Ch.7, §7.2 step 1)
    console.print("  [blue]>>>[/blue] Crawling target for utility signatures...")
    utilities = crawl_target(target_path, language)
    console.print(f"     Found [bold]{len(utilities)}[/bold] utility signatures")

    # Step 4: Schema Retrieval (Ch.7, §7.2 step 2)
    console.print("  [blue]>>>[/blue] Scanning for database schemas...")
    schemas = find_schemas(target_path)
    console.print(f"     Found [bold]{len(schemas)}[/bold] table definitions")

    # Step 5: Threat Model Linking (Ch.7, §7.2 step 3)
    threat_rules = []
    global_invariants = []
    if agent_dir:
        console.print("  [blue]>>>[/blue] Linking threat models...")
        all_rules = load_threat_rules(agent_dir)
        threat_rules = link_threats_to_target(all_rules, target_path, domain)
        console.print(
            f"     Matched [bold]{len(threat_rules)}[/bold] of "
            f"{len(all_rules)} threat rules"
        )

        global_invariants = load_global_invariants(agent_dir)
        if global_invariants:
            console.print(
                f"     Loaded [bold]{len(global_invariants)}[/bold] global invariants"
            )

    # Step 6: Interactive input (or use CLI flags)
    console.print("")

    if goal is None and not non_interactive:
        goal = Prompt.ask(
            "[bold]Engineer's Goal[/bold] (Pillar 1)\n"
            "   Write a deterministic goal statement",
        )
    elif goal is None:
        goal = f"Implement {template_type.value} for {target}"

    # State machine (optional)
    state_transition = None
    if not non_interactive:
        has_state = Prompt.ask(
            "\n[bold]State Machine[/bold] (Pillar 4)\n"
            "   Define state transitions? [y/N]",
            default="n",
        )
        if has_state.lower() in ("y", "yes"):
            initial = Prompt.ask("   Initial state")
            final = Prompt.ask("   Final state")
            state_transition = StateTransition(
                initial_state=initial,
                final_state=final,
            )

    # Custom invariants (optional)
    custom_invariants = list(global_invariants)
    if not non_interactive:
        add_invariants = Prompt.ask(
            "\n[bold]Custom Invariants[/bold] (Pillar 3)\n"
            "   Add custom invariants? [y/N]",
            default="n",
        )
        if add_invariants.lower() in ("y", "yes"):
            console.print("   Enter invariants one per line. Empty line to finish.")
            while True:
                inv = Prompt.ask("   Invariant", default="")
                if not inv:
                    break
                custom_invariants.append(inv)

    # Negative constraints
    negative_constraints = []
    if not non_interactive:
        add_neg = Prompt.ask(
            "\n[bold]Negative Constraints[/bold] (Pillar 5)\n"
            "   Add 'DO NOT' constraints? [y/N]",
            default="n",
        )
        if add_neg.lower() in ("y", "yes"):
            console.print("   Enter constraints one per line. Empty line to finish.")
            while True:
                nc = Prompt.ask("   Constraint", default="")
                if not nc:
                    break
                negative_constraints.append(nc)

    # Build blast radius
    if target_path.is_dir():
        write_files = [str(f.relative_to(target_path.parent))
                       for f in sorted(target_path.rglob("*"))
                       if f.is_file() and not f.name.startswith(".")][:20]  # Cap at 20
    else:
        write_files = [str(target_path.relative_to(target_path.parent.parent))]

    blast_radius = BlastRadius(
        allowed_write=write_files,
        allowed_read=[u.module_path.replace(".", "/") + ".py" for u in utilities
                      if u.module_path not in [str(target_path)]],
        forbidden=["Database migration scripts", "Lock files (requirements.txt, package-lock.json)"],
    )

    # Step 7: Assemble the spec
    spec = SDSDSpec(
        task_title=goal.split(".")[0] if goal else f"{template_type.value} task",
        goal=goal,
        template_type=template_type,
        utilities=utilities,
        schemas=schemas,
        threat_rules=threat_rules,
        blast_radius=blast_radius,
        state_transition=state_transition,
        invariants=custom_invariants,
        negative_constraints=negative_constraints,
        target_path=str(target_path),
        language=language,
    )

    # Step 8: Render the prompt
    prompt_text = render_prompt(spec)

    # Output
    console.print("")
    console.print(Panel(prompt_text, title="Assembled SDSD Prompt", border_style="green"))

    # Copy to clipboard
    if clipboard:
        try:
            import pyperclip
            pyperclip.copy(prompt_text)
            console.print("\n[green][OK] Prompt copied to clipboard![/green]")
        except Exception:
            console.print(
                "\n[yellow]WARNING: Could not copy to clipboard. "
                "Install pyperclip or use --output.[/yellow]"
            )

    # Write to file
    if output:
        output_path = Path(output)
        output_path.write_text(prompt_text, encoding="utf-8")
        console.print(f"\n[green][OK] Prompt written to {output_path}[/green]")
