"""Tests for the CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from sdsd.cli import app

runner = CliRunner()


class TestVersion:
    """Test --version flag."""

    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "sdsd-cli" in result.stdout


class TestInit:
    """Test `sdsd init` command."""

    def test_init_creates_agent_dir(self, tmp_path):
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / ".agent").is_dir()
        assert (tmp_path / ".agent" / "prompts").is_dir()
        assert (tmp_path / ".agent" / "workflows").is_dir()

    def test_init_creates_templates(self, tmp_path):
        runner.invoke(app, ["init", str(tmp_path)])
        prompts = tmp_path / ".agent" / "prompts"
        assert (prompts / "feature_template.md").is_file()
        assert (prompts / "bugfix_template.md").is_file()
        assert (prompts / "refactor_template.md").is_file()

    def test_init_creates_security_rules(self, tmp_path):
        runner.invoke(app, ["init", str(tmp_path)])
        rules_file = tmp_path / ".agent" / "workflows" / "security-rules.yaml"
        assert rules_file.is_file()
        content = rules_file.read_text()
        assert "TR-001" in content

    def test_init_creates_invariants(self, tmp_path):
        runner.invoke(app, ["init", str(tmp_path)])
        inv_file = tmp_path / ".agent" / "invariants.yaml"
        assert inv_file.is_file()

    def test_init_refuses_overwrite(self, tmp_path):
        (tmp_path / ".agent").mkdir()
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 1

    def test_init_force_overwrites(self, tmp_path):
        (tmp_path / ".agent").mkdir()
        result = runner.invoke(app, ["init", str(tmp_path), "--force"])
        assert result.exit_code == 0


class TestValidate:
    """Test `sdsd validate` command."""

    def test_validate_empty_project(self, tmp_path):
        result = runner.invoke(app, ["validate", "--target", str(tmp_path)])
        # Should fail — no .agent/ directory
        assert result.exit_code == 1
        assert "NOT READY" in result.stdout

    def test_validate_initialized_project(self, tmp_path):
        # First init, then validate
        runner.invoke(app, ["init", str(tmp_path)])
        result = runner.invoke(app, ["validate", "--target", str(tmp_path)])
        assert result.exit_code == 0
        assert "READY" in result.stdout
