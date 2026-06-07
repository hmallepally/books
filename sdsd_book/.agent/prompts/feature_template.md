# SDSD Feature Template
# Based on the Five Pillars of a Secure Prompt (Chapter 7, §7.1)

## GOAL (Pillar 1 — The 'What')
<!-- Write a deterministic goal statement. Two engineers should be able to
     independently verify whether this goal has been met. -->
<!-- Example: "Refactor the `/api/v1/users` endpoint to return a paginated
     JSON response in under 50ms (P95) when queried with 10,000 records." -->

[ENGINEER: Replace this with your deterministic goal statement]

## BLAST RADIUS (Pillar 2 — The 'Where')
**Allowed Files (Write):** [List the specific files the AI may modify]
**Allowed Files (Read-Only):** [List files the AI may read for context]
**Forbidden:** You may NOT modify database migration scripts. You may NOT import any third-party libraries not already present in the project's dependency lock file.

## INVARIANTS (Pillar 3 — The 'Must')
<!-- These are non-negotiable constraints. Anticipate how the AI will
     implement the feature insecurely, and forbid it. -->
1. **Security:** [Define security invariants, e.g., "All database queries MUST include tenant_id filtering"]
2. **Data Integrity:** [Define data invariants, e.g., "Conservation of Mass: sum(inputs) == sum(outputs)"]
3. **Performance:** [Define performance bounds, e.g., "P95 latency must remain under 100ms"]

## STATE MACHINE (Pillar 4 — The 'How')
**Initial State:** [Describe the system state before this feature]
**Final State:** [Describe the expected system state after this feature]

## NEGATIVE CONSTRAINTS (Pillar 5 — The 'Not')
<!-- Explicitly tell the AI what it must NOT do. -->
- Do NOT use deprecated APIs or libraries.
- Do NOT add new external dependencies without explicit approval.
- Do NOT modify files outside the Blast Radius.
- Do NOT use floating-point arithmetic for financial calculations.
