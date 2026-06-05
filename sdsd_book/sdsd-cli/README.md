# 🛡️ sdsd-cli — Spec-Driven Secure Development CLI

**Assemble secure, context-rich prompts for AI coding agents.**

`sdsd-cli` is the reference implementation of the Spec-Driven Secure Development (SDSD) methodology described in the book *Spec-Driven Secure Development* by Harinath Mallepally. It transforms the book's concepts — the Five Pillars, Repository-as-Context, Threat Model Linking — into a practical, installable tool.

## The Problem

When engineers prompt AI coding assistants with vague instructions like *"build a login page"*, the AI makes implicit architectural decisions — how to hash passwords, manage sessions, handle rate-limiting — that it is unqualified to make. This creates **hallucination risk**, **Spec Drift**, and **security vulnerabilities at AI velocity**.

## The Solution

`sdsd-cli` forces every AI interaction through a structured, context-rich template. Instead of raw prompts, engineers get **assembled engineering dossiers** that include:

- ✅ **Dependency Crawling** — AST-parsed function/class signatures from your codebase
- ✅ **Schema Retrieval** — Database DDL auto-injected from migration files
- ✅ **Threat Model Linking** — Security rules automatically matched to the target domain
- ✅ **Five-Pillar Templates** — Goal, Blast Radius, Invariants, State Machine, Negative Constraints

---

## Quick Start

### Install

```bash
pip install sdsd-cli
```

Or install from source:

```bash
git clone https://github.com/harinath-mallepally/sdsd-cli.git
cd sdsd-cli
pip install -e .
```

### 1. Initialize Your Project

```bash
cd your-project/
sdsd init
```

This scaffolds the `.agent/` directory:

```
.agent/
├── prompts/
│   ├── feature_template.md
│   ├── bugfix_template.md
│   └── refactor_template.md
├── workflows/
│   └── security-rules.yaml
└── invariants.yaml
```

### 2. Assemble a Prompt

```bash
sdsd prompt create --type feature --target src/payments/
```

The tool will:
1. 🔍 Crawl `src/payments/` for function/class signatures
2. 🔍 Scan for database schemas in nearby migration files
3. 🔍 Link matching threat rules from `.agent/workflows/`
4. 📝 Interactively collect your Goal, State Machine, and Invariants
5. 📋 Assemble and copy the full SDSD prompt to your clipboard

### 3. Validate SDSD Readiness

```bash
sdsd validate --target .
```

Checks your project against the SDSD readiness checklist:

```
SDSD Readiness Report
┌────────┬──────────────────────────┬─────────────────────────────────────┐
│ Status │ Pillar                   │ Details                             │
├────────┼──────────────────────────┼─────────────────────────────────────┤
│   ✅   │ Repository-as-Context    │ .agent/ found                       │
│   ✅   │ Prompt Templates         │ 3 templates found                   │
│   ✅   │ Threat Models            │ 15 threat rules defined             │
│   ✅   │ Global Invariants        │ 5 global invariants defined         │
│   ⚠️   │ IDE Invariant Walls      │ No .cursorrules found               │
│   ✅   │ Adversarial Tests        │ 12 test files found                 │
└────────┴──────────────────────────┴─────────────────────────────────────┘
Result: READY — 5/6 checks passed
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `sdsd init [directory]` | Scaffold `.agent/` directory with templates and threat blueprints |
| `sdsd prompt create --type <template> --target <path>` | Assemble a full SDSD prompt with dynamic context |
| `sdsd validate --target <path>` | Check SDSD readiness of a project |
| `sdsd --version` | Show version |

### `sdsd prompt create` Options

| Flag | Description |
|------|-------------|
| `--type`, `-t` | Template type: `feature`, `bugfix`, `refactor` (required) |
| `--target`, `-T` | Target file or directory (required) |
| `--goal`, `-g` | Goal statement (skips interactive prompt) |
| `--domain`, `-d` | Domain hint for threat model matching |
| `--output`, `-o` | Write prompt to a file instead of clipboard |
| `--no-clipboard` | Don't copy to clipboard |
| `--non-interactive` | Skip all interactive prompts (for CI/scripting) |

---

## Multi-Language Support

`sdsd-cli` automatically detects your project's language and adapts:

| Language | Crawling Method | Supported |
|----------|----------------|-----------|
| Python   | Full AST parsing (`ast` module) | ✅ v0.1 |
| Java     | Regex-based signature extraction | ✅ v0.1 |
| C#       | Regex-based signature extraction | ✅ v0.1 |

---

## Customizing Templates

Edit the templates in `.agent/prompts/` to match your organization's standards. Templates use the **Five Pillars** structure:

1. **Goal** (Pillar 1) — Deterministic, verifiable outcome
2. **Blast Radius** (Pillar 2) — Physical file boundaries
3. **Invariants** (Pillar 3) — Non-negotiable constraints
4. **State Machine** (Pillar 4) — Before/after system states
5. **Negative Constraints** (Pillar 5) — Explicit "DO NOT" rules

### Adding Threat Rules

Edit `.agent/workflows/security-rules.yaml` to add domain-specific threat rules:

```yaml
rules:
  - rule_id: "TR-100"
    domain: "payment"
    severity: "HIGH"
    constraint: >
      All monetary calculations MUST use Decimal types.
      Floating-point arithmetic is STRICTLY FORBIDDEN.
```

Rules are automatically matched to targets by domain tags in the file path.

---

## Development

```bash
git clone https://github.com/harinath-mallepally/sdsd-cli.git
cd sdsd-cli
pip install -e ".[dev]"
pytest
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Author

**Harinath Mallepally** — Author of *Spec-Driven Secure Development*

---

*"Fix the spec, not the code."*
