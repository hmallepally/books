"""Tests for the threat linker module."""

from pathlib import Path

import yaml

from sdsd.core.threat_linker import (
    find_agent_dir,
    link_threats_to_target,
    load_global_invariants,
    load_threat_rules,
)
from sdsd.models.spec import ThreatRule


class TestFindAgentDir:
    """Tests for .agent/ directory discovery."""

    def test_find_agent_dir_exists(self, tmp_path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        result = find_agent_dir(tmp_path)
        assert result == agent_dir

    def test_find_agent_dir_in_parent(self, tmp_path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        child = tmp_path / "src" / "payments"
        child.mkdir(parents=True)
        result = find_agent_dir(child)
        assert result == agent_dir

    def test_find_agent_dir_not_found(self, tmp_path):
        result = find_agent_dir(tmp_path / "nonexistent")
        assert result is None


class TestLoadThreatRules:
    """Tests for loading threat rules from YAML."""

    def test_load_rules(self, tmp_path):
        agent_dir = tmp_path / ".agent"
        workflows = agent_dir / "workflows"
        workflows.mkdir(parents=True)

        rules_data = {
            "rules": [
                {
                    "rule_id": "TR-001",
                    "domain": "global",
                    "severity": "HIGH",
                    "constraint": "Use parameterized queries.",
                },
                {
                    "rule_id": "TR-020",
                    "domain": "payment",
                    "severity": "HIGH",
                    "constraint": "Use Decimal for money.",
                },
            ]
        }
        (workflows / "security-rules.yaml").write_text(
            yaml.dump(rules_data), encoding="utf-8"
        )

        rules = load_threat_rules(agent_dir)
        assert len(rules) == 2
        assert rules[0].rule_id == "TR-001"
        assert rules[1].domain == "payment"

    def test_load_empty_workflows(self, tmp_path):
        agent_dir = tmp_path / ".agent"
        (agent_dir / "workflows").mkdir(parents=True)
        rules = load_threat_rules(agent_dir)
        assert rules == []


class TestLinkThreats:
    """Tests for matching threat rules to target paths."""

    def _make_rules(self) -> list[ThreatRule]:
        return [
            ThreatRule(rule_id="TR-001", domain="global", constraint="Always applies"),
            ThreatRule(rule_id="TR-020", domain="payment", constraint="Use Decimal"),
            ThreatRule(rule_id="TR-010", domain="auth", constraint="Use bcrypt"),
        ]

    def test_global_always_matches(self):
        rules = self._make_rules()
        matched = link_threats_to_target(rules, Path("src/frontend/ui.py"))
        ids = [r.rule_id for r in matched]
        assert "TR-001" in ids

    def test_domain_matches_path(self):
        rules = self._make_rules()
        matched = link_threats_to_target(rules, Path("src/payments/processor.py"))
        ids = [r.rule_id for r in matched]
        assert "TR-001" in ids  # global
        assert "TR-020" in ids  # payment

    def test_domain_hint_overrides(self):
        rules = self._make_rules()
        matched = link_threats_to_target(
            rules, Path("src/utils/helper.py"), domain_hint="auth"
        )
        ids = [r.rule_id for r in matched]
        assert "TR-010" in ids  # auth via hint


class TestLoadGlobalInvariants:
    """Tests for loading global invariants."""

    def test_load_invariants(self, tmp_path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        inv_data = {"invariants": ["Use parameterized queries", "Add timestamps"]}
        (agent_dir / "invariants.yaml").write_text(
            yaml.dump(inv_data), encoding="utf-8"
        )
        result = load_global_invariants(agent_dir)
        assert len(result) == 2

    def test_load_missing_file(self, tmp_path):
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        result = load_global_invariants(agent_dir)
        assert result == []
