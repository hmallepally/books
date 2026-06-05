# SDSD Bugfix Template
# Based on the Steering Loop (Chapter 5, §5.3)

## BUG DESCRIPTION
<!-- Describe the bug precisely. Include the observed behavior,
     the expected behavior, and how to reproduce it. -->

**Observed Behavior:** [What is happening]
**Expected Behavior:** [What should happen]
**Reproduction Steps:** [How to trigger the bug]

## GOAL (Pillar 1)
<!-- The fix must be deterministic and verifiable. -->
[ENGINEER: Define the fix as a testable assertion]

## BLAST RADIUS (Pillar 2)
**Allowed Files (Write):** [Only the files necessary to fix the bug]
**Allowed Files (Read-Only):** [Related files for context]
**Forbidden:** You may NOT refactor unrelated code. You may NOT modify tests that are currently passing unless the test itself is incorrect.

## ROOT CAUSE ANALYSIS
<!-- CRITICAL: Do NOT let the AI patch the symptom. Identify the root cause
     and fix it at the specification level, not the syntax level. -->
<!-- Reference: Chapter 5 — "If you fix the machine's code manually,
     you have doomed yourself to fix it manually forever." -->

**Suspected Root Cause:** [Engineer's hypothesis]
**Spec Drift Check:** Does this bug indicate the specification is incomplete? If yes, update the invariants below.

## INVARIANTS (Pillar 3)
1. **Regression Guard:** The fix MUST include a new test case that reproduces the original bug and verifies it no longer occurs.
2. [Add any new invariants discovered from this bug]

## NEGATIVE CONSTRAINTS (Pillar 5)
- Do NOT suppress the error with a try/except catch-all.
- Do NOT add workarounds; fix the root cause.
- Do NOT modify the database schema to fix application logic bugs.
