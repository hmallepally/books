"""Template Engine.

Loads SDSD prompt templates from `.agent/prompts/` and renders
the fully assembled prompt in the exact format described in
Chapter 7, §7.2 (L108-134).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sdsd.models.spec import SDSDSpec, TemplateType


def load_template(agent_dir: Path, template_type: TemplateType) -> Optional[str]:
    """Load a prompt template from the `.agent/prompts/` directory.

    Args:
        agent_dir: Path to the `.agent/` directory.
        template_type: Which template to load (feature, bugfix, refactor).

    Returns:
        The raw template content, or None if not found.
    """
    prompts_dir = agent_dir / "prompts"
    template_file = prompts_dir / f"{template_type.value}_template.md"

    if not template_file.is_file():
        return None

    return template_file.read_text(encoding="utf-8")


def render_prompt(spec: SDSDSpec) -> str:
    """Render a fully assembled SDSD prompt from a spec.

    This generates the exact output format shown in Chapter 7, §7.2,
    lines 108-134 — a dense, highly structured engineering dossier
    ready to be pasted into any AI coding agent.
    """
    sections = []

    # ===== Header =====
    sections.append(f"# TASK: {spec.task_title}")
    sections.append("")

    # ===== Dynamic Context =====
    sections.append("## DYNAMIC CONTEXT [DO NOT MODIFY]")
    sections.append("")

    # Utilities
    if spec.utilities:
        sections.append("**Available Internal Utilities:**")
        for util in spec.utilities:
            if util.kind == "class" and util.methods:
                methods_str = ", ".join(f"`{m}`" for m in util.methods)
                sections.append(f"- `{util.module_path}.{util.name}` (Methods: {methods_str})")
            elif util.kind == "function":
                sections.append(f"- `{util.module_path}.{util.name}` — `{util.signature}`")
            else:
                sections.append(f"- `{util.module_path}.{util.name}`")
        sections.append("")

    # Schemas
    if spec.schemas:
        sections.append("**Database Schema (Target Tables):**")
        sections.append("```sql")
        for schema in spec.schemas:
            sections.append(schema.ddl)
            sections.append("")
        sections.append("```")
        sections.append("")

    # Threat rules
    if spec.threat_rules:
        sections.append("**Global Threat Models Applied:**")
        for rule in spec.threat_rules:
            sections.append(
                f"- {rule.rule_id} ({rule.domain.title()}, {rule.severity}): "
                f"{rule.constraint}"
            )
        sections.append("")

    # ===== Engineer's Goal (Pillar 1) =====
    sections.append("## ENGINEER'S GOAL")
    sections.append(spec.goal)
    sections.append("")

    # ===== Blast Radius (Pillar 2) =====
    sections.append("## BLAST RADIUS")
    if spec.blast_radius.allowed_write:
        sections.append(
            "**Allowed Files (Write):** "
            + ", ".join(f"`{f}`" for f in spec.blast_radius.allowed_write)
        )
    if spec.blast_radius.allowed_read:
        sections.append(
            "**Allowed Files (Read-Only):** "
            + ", ".join(f"`{f}`" for f in spec.blast_radius.allowed_read)
        )
    if spec.blast_radius.forbidden:
        sections.append(
            "**Forbidden:** " + " ".join(spec.blast_radius.forbidden)
        )
    sections.append("")

    # ===== State Machine (Pillar 4) =====
    if spec.state_transition:
        sections.append("## STATE TRANSITION")
        sections.append(f"**Initial State:** `{spec.state_transition.initial_state}`")
        sections.append(f"**Final State:** `{spec.state_transition.final_state}`")
        sections.append("")

    # ===== Invariants (Pillar 3) =====
    if spec.invariants:
        sections.append("## INVARIANTS")
        for i, inv in enumerate(spec.invariants, 1):
            sections.append(f"{i}. {inv}")
        sections.append("")

    # ===== Negative Constraints (Pillar 5) =====
    if spec.negative_constraints:
        sections.append("## NEGATIVE CONSTRAINTS")
        for nc in spec.negative_constraints:
            sections.append(f"- [FORBIDDEN] {nc}")
        sections.append("")

    return "\n".join(sections)
