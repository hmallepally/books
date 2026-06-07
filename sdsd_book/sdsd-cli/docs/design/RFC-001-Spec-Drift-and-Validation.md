# RFC-001: Addressing Spec Drift, Semantic Validation, and Test Integration

**Author:** Hari Mallepally
**Status:** Draft
**Component:** `sdsd-cli` v2.0 Roadmap

## Overview

The Spec-Driven Secure Development (SDSD) methodology requires the specification to be the ultimate source of truth. However, enterprise adoption reveals three critical failure modes when transitioning from syntax-typists to spec-architects:
1.  **Spec Drift:** Engineers update code without updating the corresponding spec under deadline pressure.
2.  **Valid but Wrong (Human Error):** The human authors a syntactically valid YAML spec that describes inherently insecure or incorrect business logic.
3.  **Manual Test Overhead:** Specs are written, but tests verifying those specs are written manually, leading to divergence.

This RFC proposes the architectural roadmap for `sdsd-cli` v2.0 to solve these exact enterprise challenges.

---

## 1. The Drift Problem: Bi-Directional Verification

**The Challenge:** Once the spec and the code are both checked in, a team under a deadline will update the code and forget the spec. Does the tool compare generated behavior against the spec, or just validate format?

**The Solution:** Git-Integrated AST Hashing and `sdsd check-drift`

We will introduce a strict pre-commit verification command: `sdsd check-drift`. 
*   **Mechanism:** When `sdsd generate` creates or updates a source file based on a spec, it will inject a cryptographic hash of the spec's AST (Abstract Syntax Tree) into a hidden metadata file or code comment.
*   **Enforcement:** During a commit hook, `sdsd check-drift` computes the diff of the source code. If the code has changed but the associated spec hash is stale (meaning the spec wasn't modified concurrently), the commit is rejected. 
*   **Behavior vs. Format:** While v1 validates the format and boundaries of the spec before generation, v2 will use AST comparison to ensure that the actual endpoints and data models present in the code strictly mirror the spec definitions.

---

## 2. Garbage In, Garbage Out: Semantic Validation of the Spec

**The Challenge:** When the AI perfectly satisfies the spec, but the spec itself is wrong or unsafe. A clean spec that describes the wrong behavior passes validation but still ships a bug. How do we surface that gap?

**The Solution:** Global Invariant Walls and Semantic Cross-Validation

Humans make mistakes. To save humans from writing bad specs, we must evaluate the spec against a higher-level, immutable policy before code is ever generated.
*   **Mechanism:** Organizations define a `global_invariants.yaml` (e.g., *"Rule 1: All data mutation requires an authenticated tenant_id"*).
*   **Enforcement:** We introduce `sdsd validate-spec --against global_invariants.yaml`. This uses a local semantic parser (or an LLM judge) to evaluate the *intent* of the human's feature spec against the organization's absolute invariants.
*   **Fail-Safe:** If a developer writes a feature spec for a "Fast Transaction API" that accidentally omits the authorization check, `sdsd validate` will block generation, returning: `[FATAL] Spec violates Global Invariant Rule 1: Missing authentication boundary.`

---

## 3. Test Framework Integration: The Spec *is* the Test

**The Challenge:** Is the roadmap to pair this with a test framework, or is the spec itself the test?

**The Solution:** Automated Property-Based Test Scaffolding (`sdsd generate-tests`)

The spec must become executable. Keeping it CLI-only and composable is the right call, which means `sdsd-cli` should integrate elegantly with existing ecosystems.
*   **Mechanism:** We introduce the `sdsd generate-tests` command.
*   **Enforcement:** The CLI will parse the `boundaries`, `data_models`, and `invariants` defined in the YAML spec, and automatically scaffold Property-Based Testing suites tailored to the target language framework.
    *   **Python:** Generates `PyTest` + `Hypothesis` strategies.
    *   **Java:** Generates `JUnit` + `jqwik` strategies.
    *   **C#:** Generates `xUnit` + `FsCheck` strategies.
*   **Result:** The CLI doesn't just run the tests; it creates the rigorous fuzzing tests that *enforce* the spec boundaries, leaving the human to simply execute their standard CI test runner. The spec effectively becomes the test.
