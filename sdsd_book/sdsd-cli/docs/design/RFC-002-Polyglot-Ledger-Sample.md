# RFC-002: Realistic Polyglot Ledger Sample Ecosystem

**Author:** Hari Mallepally
**Status:** Implemented
**Component:** `sdsd_book/samples`

## Overview
To effectively demonstrate the Spec-Driven Secure Development (SDSD) methodology, toy examples (like simple username/password checks) are insufficient. SDSD is designed to prevent catastrophic semantic failures and AI hallucinations in highly complex, highly regulated environments. 

This RFC proposes replacing the generic `AuthService` samples with a **Polyglot Financial Ledger Ecosystem**. This ecosystem acts as a realistic Monorepo integration test for the `sdsd verify` engine.

## Architecture

The ecosystem consists of three microservices, each demonstrating a different aspect of SDSD enforcement across different language ecosystems:

### 1. Java Ledger Engine (`java-ledger-engine`)
*   **Structure:** Multi-module Gradle project (`core-domain` and `ledger-service`).
*   **Responsibility:** The core transaction processing engine.
*   **SDSD Invariant:** "Account balances can never drop below zero, and transactions cannot be double-spent."
*   **Enforcement:** Strict JUnit 5 invariant testing paired with mandatory 100% PIT Mutation Coverage. SDSD ensures the balance-checking logic cannot be silently stripped or bypassed by an LLM hallucination.

### 2. Python Fraud Analyzer (`python-fraud-analyzer`)
*   **Structure:** Python 3.11 with `pytest`.
*   **Responsibility:** Evaluates transactions asynchronously for fraudulent patterns.
*   **SDSD Invariant:** "Every transaction exceeding $10,000 must be explicitly flagged for manual review, regardless of risk score."
*   **Enforcement:** PyTest property-based boundary tests enforcing the $10,000 hard gate.

### 3. C# API Gateway (`csharp-api-gateway`)
*   **Structure:** .NET 9.0 ASP.NET Core with `xUnit`.
*   **Responsibility:** The entry point routing requests to the ledger.
*   **SDSD Invariant:** "Strict Tenant Isolation (IDOR Prevention). A user in Tenant A cannot execute or view transactions in Tenant B."
*   **Enforcement:** `xUnit` integration tests simulating malicious cross-tenant routing.

## Spec-First Validation Flow
1. A `.yaml` spec is written for the component.
2. The Test Invariants (the "Wall") are written to enforce the spec.
3. The AI agent generates the implementation.
4. `sdsd verify` runs the native test suite and mutation analysis, guaranteeing the code perfectly adheres to the specification before merging.

## Deployment & CI/CD
This ecosystem is embedded directly into the `samples/` directory of the `sdsd_book` repository. A GitHub Actions pipeline (`.github/workflows/sdsd-pipeline.yml`) runs `sdsd verify` against all three projects simultaneously upon every pull request, enforcing the global SDSD Monorepo constraints.
