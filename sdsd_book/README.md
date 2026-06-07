# Spec-Driven Secure Development (SDSD) Companion Repository

Welcome to the official companion repository for the book **Spec-Driven Secure Development** by Hari Mallepally.

This repository provides the open-source tools, code snippets, and IDE configuration templates necessary to implement the SDSD methodology in your own enterprise environments.

## Repository Structure

### 1. `sdsd-cli/`
The official Python-based command-line interface for the SDSD workflow. The CLI assumes a spec-first approach, enforcing that development is driven by defined invariants rather than raw code generation.

*   **Docs & Roadmaps:** Check out `sdsd-cli/docs/design/` to view the architectural roadmap, including [RFC-001](sdsd-cli/docs/design/RFC-001-Spec-Drift-and-Validation.md) which details our upcoming solutions for Spec Drift, Semantic Validation, and Test Framework Integration.

### 2. `code/`
This directory contains every code snippet featured in the book, cleanly organized by chapter and language.
*   `python/` - FastAPI, PyTest, SQLAlchemy implementations.
*   `java/` - Spring Boot, JUnit 5, JPA implementations.
*   `csharp/` - ASP.NET Core, xUnit, Entity Framework Core implementations.

### 3. `templates/`
Ready-to-use IDE Invariant Wall templates from Chapter 14. Drop these into the root of your project to strictly constrain your AI coding assistants.
*   `.cursorrules` - For Cursor IDE.
*   `.windsurfrules` - For Windsurf IDE.

## Getting Started with SDSD-CLI

The `sdsd-cli` tool is designed to parse your specifications, validate them against your architectural boundaries, and safely bootstrap the secure generation process.

To install from source:
```bash
cd sdsd-cli
pip install -e .
```

## Contributing

We welcome contributions! Please review our RFCs in the `docs/design` directory if you are interested in contributing to the V2 roadmap features like AST-hashing drift detection or property-based test scaffolding.
