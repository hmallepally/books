# SDSD Refactor Template
# Based on the Bounded Context principles (Chapter 6, §6.2)

## REFACTORING GOAL (Pillar 1)
<!-- A refactor must produce identical external behavior with improved
     internal structure. Define a measurable improvement target. -->
<!-- Example: "Reduce cyclomatic complexity of the payment router from 47 to <15
     while maintaining 100% of existing test coverage." -->

[ENGINEER: Define the measurable refactoring goal]

## BLAST RADIUS (Pillar 2)
<!-- Refactors are the most dangerous AI task. The AI will attempt to
     rewrite the entire codebase if not strictly bounded. -->
**Allowed Files (Write):** [Strictly limit to the files being refactored]
**Allowed Files (Read-Only):** [Files needed for context — interfaces, types, contracts]
**Forbidden:** You may NOT modify any public API signatures. You may NOT change any inter-service communication contracts. You may NOT modify files outside the target module's Bounded Context.

## BEHAVIORAL INVARIANTS (Pillar 3)
<!-- The refactored code MUST pass every existing test without modification.
     If a test fails after refactoring, the refactor is wrong, not the test. -->
1. **Behavioral Equivalence:** All existing tests MUST pass without modification. Zero behavioral regressions allowed.
2. **API Stability:** No public function signatures, return types, or error codes may change.
3. **Performance:** Refactored code must not degrade P95 latency by more than 5%.

## ARCHITECTURAL CONSTRAINTS
<!-- Reference: Chapter 6 — Bounded Context isolation -->
- Maintain existing Bounded Context boundaries. Do not introduce cross-domain imports.
- Preserve existing dependency direction (e.g., if A depends on B, do not make B depend on A).
- Do not introduce circular dependencies.

## NEGATIVE CONSTRAINTS (Pillar 5)
- Do NOT add new external dependencies.
- Do NOT rename public APIs or database columns.
- Do NOT combine this refactor with feature additions. Refactoring is structural only.
