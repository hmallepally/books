"""Threat Model Linker.

Implements the "Threat Model Linking" step from Chapter 7, §7.2:
searches the global `.agent/workflows/` directory for any security
rules tagged with the target domain and injects them as mandatory
Invariants into the assembled prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from sdsd.models.spec import ThreatRule


def find_agent_dir(start: Path) -> Optional[Path]:
    """Walk up from `start` to find the nearest `.agent/` directory.

    This implements the "Repository-as-Context" lookup described in
    Chapter 4 — the tool searches for the .agent/ directory that
    contains the project's embedded invariants and workflows.
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent

    # Walk up to root, max 20 levels
    for _ in range(20):
        agent_dir = current / ".agent"
        if agent_dir.is_dir():
            return agent_dir
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def load_threat_rules(agent_dir: Path) -> list[ThreatRule]:
    """Load all threat rules from `.agent/workflows/` YAML files.

    Each YAML file in the workflows directory can contain a `rules` key
    with a list of threat model rules, each having:
    - rule_id: str
    - domain: str (tag for matching, e.g. 'payment', 'auth')
    - severity: str (HIGH, MEDIUM, LOW)
    - constraint: str (the actual constraint text)
    """
    workflows_dir = agent_dir / "workflows"
    if not workflows_dir.is_dir():
        return []

    rules = []
    for yaml_file in sorted(workflows_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError):
            continue

        if not isinstance(data, dict):
            continue

        for rule_data in data.get("rules", []):
            if not isinstance(rule_data, dict):
                continue
            try:
                rules.append(ThreatRule(
                    rule_id=str(rule_data.get("rule_id", "UNKNOWN")),
                    domain=str(rule_data.get("domain", "")),
                    severity=str(rule_data.get("severity", "HIGH")),
                    constraint=str(rule_data.get("constraint", "")),
                ))
            except Exception:
                continue

    return rules


def load_global_invariants(agent_dir: Path) -> list[str]:
    """Load global invariants from `.agent/invariants.yaml`.

    Returns a list of invariant constraint strings.
    """
    invariants_file = agent_dir / "invariants.yaml"
    if not invariants_file.is_file():
        return []

    try:
        data = yaml.safe_load(invariants_file.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return []

    if not isinstance(data, dict):
        return []

    return [str(inv) for inv in data.get("invariants", []) if inv]


def link_threats_to_target(
    rules: list[ThreatRule],
    target: Path,
    domain_hint: Optional[str] = None,
) -> list[ThreatRule]:
    """Match threat rules to a target path by domain tagging.

    Uses heuristics to match rules:
    1. If a domain_hint is provided, match rules with that domain.
    2. Otherwise, infer the domain from the target path segments
       (e.g., 'src/payments/' matches domain='payment').
    3. Rules with domain='global' or domain='*' always match.

    This implements the "Threat Model Linking" from Ch.7, §7.2, step 3.
    """
    # Infer domain from target path
    path_parts = set()
    for part in target.parts:
        path_parts.add(part.lower())
        # Also add singular/plural variants
        if part.lower().endswith("s"):
            path_parts.add(part.lower()[:-1])
        else:
            path_parts.add(part.lower() + "s")

    if domain_hint:
        path_parts.add(domain_hint.lower())

    matched = []
    for rule in rules:
        domain = rule.domain.lower().strip()

        # Global rules always match
        if domain in ("global", "*", "all"):
            matched.append(rule)
            continue

        # Check if the rule domain matches any path segment
        if domain in path_parts:
            matched.append(rule)
            continue

        # Partial match: 'payment' matches 'payment_routing'
        for part in path_parts:
            if domain in part or part in domain:
                matched.append(rule)
                break

    return matched
