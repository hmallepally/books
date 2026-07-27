# Skipped & Deferred Findings — All Rounds

> [!NOTE]
> This document tracks all findings from the 5-round critical review process that were **intentionally NOT fixed** in the current edition. Each item includes the rationale for deferral and recommended timing for resolution.

---

## Category 1: Structural Changes (High Risk)

### S1. Chapter Reordering (Round 1: M19, Round 2: M4, Round 3: M4)
**Finding:** Ch 19 (Behavioral Leadership) interrupts the distributed systems flow (Ch 16→17→18→19→20→21). Message Brokers (Ch 21) is separated from System Architecture (Ch 16). Recommended order: 16→21→17→18→20→19.

**Why Skipped:** Reordering chapters is a high-risk structural change that cascades through:
- All 3 study plans in the Prologue (chapter numbers referenced)
- Cross-references in every chapter ("as discussed in Chapter X")
- The build pipeline directory numbering (00-24)
- All 3 compiled editions (Java, Python, C#)
- Amazon KDP page references if already published

**Recommendation:** Address in v3.0 (next major edition). Requires a dedicated reordering pass with full regression testing.

---

### S2. Persona C Dedicated 14-Day Study Plan (Round 2: m1, Round 3: m1)
**Finding:** Persona C (Engineering Manager/Director returning to coding) lacks a dedicated 14-day sprint plan like Personas A and B.

**Why Skipped:** Persona C's needs overlap significantly with both Persona A (algorithmic refresher) and Persona B (system design depth). The 28-Day Comprehensive Plan already labels which days are relevant to each persona. Creating a third 14-day plan risks confusing readers with too many similar options.

**Recommendation:** If reader feedback requests it, add a "14-Day Manager's Balanced Sprint" that alternates between algorithmic (odd days) and system design + behavioral (even days) content.

---

## Category 2: Code Standardization (Low Impact)

### S3. Java `var` and `Deque` Standardization (Round 2: M7, Round 3: M2)
**Finding:** Chapters 10-12 use explicit typing (`Map<Integer, Integer> map = new HashMap<>()`) while Chapter 13 uses modern `var`. Some snippets still use `Stack` instead of `Deque/ArrayDeque`.

**Why Skipped:** This is a stylistic inconsistency across 37+ snippet files per language (100+ files total). It does not affect:
- Code correctness (both styles compile and run identically)
- Algorithm understanding (the logic is the same)
- Interview performance (interviewers accept both styles)

The risk of introducing bugs during mass find-replace across 100+ files outweighs the cosmetic benefit.

**Recommendation:** Address systematically in next edition using an automated linter/formatter pass.

---

### S4. Inconsistent Edge Case Handling (Round 2: M9)
**Finding:** Some solved exemplars guard against `null` and empty inputs, while others do not.

**Why Skipped:** Adding defensive null/empty guards to every code sample changes the pedagogical focus from "teaching the algorithm" to "teaching defensive programming." In a timed assessment context, candidates should focus on the core algorithm first.

**Recommendation:** Add a dedicated "Defensive Coding Checklist" to Chapter 23 (Appendix).

---

## Category 3: Content Depth Enhancements

### S5. Trace Tables for Chapters 10 & 11 (Round 1: m7, Round 2: M8, Round 3: M8)
**Finding:** Explanations in Ch 10-11 are 1-3 sentences per problem, while Ch 13 has detailed trace-through tables.

**Why Skipped:** Adding trace tables to every exemplar in Ch 10-11 would increase each chapter by 40-60%. These cover "Easy/Medium" problems where algorithms are more intuitive. The visual diagrams (read_write_pointer.png, spiral_traversal.png) already serve this purpose. Ch 13's DP/graph problems genuinely require step-by-step traces because state transitions are non-obvious.

**Recommendation:** Add trace tables selectively for the 3-4 most complex problems in each chapter, not all.

---

### S6. Mock Interview Transcripts in Chapters 17-22 (Round 2: M12)
**Finding:** Chapters 17-22 lack interactive mock interview dialogue.

**Why Skipped:** Writing authentic mock transcripts requires realistic interviewer personas with probing follow-ups — this is high-quality creative writing that shouldn't be rushed. Poor mock transcripts (the "AI corporate speak" problem we already fixed in Ch 19) are worse than no transcripts.

**Recommendation:** Write 1 mock transcript per chapter (5 total) in next edition based on real interview experiences.

---

### S7. K8s/eBPF Section Depth Expansion (Round 3: M5)
**Finding:** Modern Infrastructure Patterns in Ch 16 covers K8s, Envoy, eBPF, Serverless in one paragraph each — too surface-level.

**Why Skipped:** Expanding would make Ch 16 disproportionately long. The proper fix is splitting Ch 16 into two chapters, which falls under structural reordering (S1).

**Recommendation:** In v3.0, split Ch 16 into "System Architecture Fundamentals" and "Cloud-Native Infrastructure."

---

### S8. Multimodal AI Capabilities (Round 3: m4)
**Finding:** Ch 22 misses multimodal AI (vision, audio) which is a key 2024-2026 development.

**Why Skipped:** Multimodal AI is rapidly evolving — specific content written today will be outdated within 6 months. The chapter's focus on architectural patterns (RAG, semantic caching, prompt security) is more durable.

**Recommendation:** Add a 1-page "Multimodal Considerations" subsection in next edition update.

---

## Category 4: Minor Polish Items

### S9. Ch 13/14 Ordering (Round 3: m2)
**Finding:** Having Decomposition Capstone (Ch 14) AFTER Optimization/DP (Ch 13) feels backward.

**Why Skipped:** Same as S1 (chapter reordering risk). The Decomposition Primer (Ch 02) already provides the foundational framework BEFORE the algorithmic chapters. Ch 14 serves as a synthesis/capstone.

---

### S10. `int[256]` Notation in Ch 10 Table (Round 3: m3)
**Finding:** A comparison table uses Java-specific `int[256]` notation.

**Why Skipped:** Extremely minor — will be addressed as part of broader language leakage cleanup.

---

## Summary

| Category | Count | Impact | When to Fix |
|----------|-------|--------|-------------|
| Structural Changes | 2 | High risk | v3.0 (next edition) |
| Code Standardization | 2 | Low impact | Next edition (automated) |
| Content Depth | 4 | Medium impact | Next edition (manual) |
| Minor Polish | 2 | Low impact | Next update |
| **Total Skipped** | **10** | | |
