# Round 2 Critical Review Report

> [!WARNING]
> **24 findings — 5 CRITICAL, 12 MAJOR, 7 MINOR**
> Round 1 fixed 19 issues. 8 R1 items were deferred and remain unfixed. 10 new issues discovered.

---

## R1 Fix Verification

| Fix | Status | Notes |
|-----|--------|-------|
| C2: isValidSudoku int encoding | ✅ Fixed | |
| C3: canShip predicate added | ✅ Fixed | |
| C4: Graph exemplars in Ch 13 | ✅ Fixed | |
| C5: Syntax Trap deleted | ⚠️ Partial | Header + paragraph remnant at lines 15-18 |
| C6: Persona study plans | ⚠️ Partial | Missing Persona C plan |
| C8: Decomposition framework | ✅ Fixed | |
| C9: Stream beginner tutorials | ❌ Failed | Practice questions + .map() tutorials still present |
| C12: TokenBucket CAS | ✅ Fixed | Minor race on lastRefillTimestamp |
| M1-M6, M8: Code/depth fixes | ✅ Fixed | All applied correctly |
| M9: Historical trivia | ✅ Fixed | |
| M14: Rainwater derivation | ✅ Fixed | |

---

## 🚨 CRITICAL (5)

| # | Chapter | Finding | Action |
|---|---------|---------|--------|
| C1 | Ch 14/09 | **Pattern ID mismatch.** Ch 14 references wrong PAT-XX IDs (PAT-15=PQ but Ch09 says PAT-15=DFS Flood Fill). | Audit and align ALL pattern cross-references |
| C2 | Ch 09 | **Missing PAT-25 (Heap/PQ).** No dedicated canonical pattern despite heavy usage. | Add PAT-25: Priority Queue / Heap |
| C3 | Ch 06 | **Beginner stream content STILL present.** Practice questions and .map() tutorials remain. | Delete remaining beginner sections |
| C4 | Ch 05 | **No SOLID trade-offs.** Chapter presents SOLID as absolute good without discussing when it hurts. | Add "When SOLID Hurts" section |
| C5 | Ch 19 | **STAR responses still robotic.** "I instructed the team to apply the HikariCP formula..." | Rewrite with human struggle and team credit |

## 🛑 MAJOR (12)

| # | Chapter | Finding | Action |
|---|---------|---------|--------|
| M1 | Ch 03 | ZenithTrade in-memory durability still naive (C7 from R1) | Add Event Sourcing / journaling |
| M2 | Ch 01 | Binary Search invariant still overused (M13 from R1) | Add modern invariant example |
| M3 | Ch 03 | Case studies still framed as "Exercises" (M15 from R1) | Reframe as architectural blueprints |
| M4 | Ch 08 | HikariCP formula still universal (M11 from R1) | Add PostgreSQL/HDD context caveat |
| M5 | Ch 13 | Search in Rotated Array misplaced in Hard chapter | Move to Ch 12 or add prominent Medium note |
| M6 | Ch 15 | Duplicate mock problems (Sets 2/20, 4/8, 3/14 share Q4s) | Replace duplicates with unique problems |
| M7 | Ch 10-12 | Inconsistent Java 21+ style (no `var` vs Ch 13 using it) | Standardize `var` usage across all chapters |
| M8 | Ch 10-11 | Shallow 1-3 sentence explanations (vs Ch 13 trace tables) | Add trace-through tables |
| M9 | Ch 10-12 | Inconsistent edge case handling (some null-check, some don't) | Standardize null/empty guards |
| M10 | Ch 16-22 | No Kubernetes, service mesh, eBPF content (M17 from R1) | Add modern infrastructure section |
| M11 | Ch 24 | Missing modern references (M18 from R1) | Add Transformer, RAG, DynamoDB papers |
| M12 | Ch 16-22 | Missing mock interview transcripts in Ch 17-22 | Add interactive dialogue examples |

## ⚠️ MINOR (7)

| # | Chapter | Finding |
|---|---------|---------|
| m1 | Ch 00 | Missing Persona C dedicated 14-day plan |
| m2 | Ch 00 | 28-Day plan Week 2 skewed to Persona A |
| m3 | Ch 01 | Syntax Trap header remnant (lines 15-18) |
| m4 | Ch 22 | Missing multimodal AI capabilities |
| m5 | Ch 23 | Jeff Dean 2012 latency numbers still outdated |
| m6 | Ch 13/09 | searchRotated code duplicated verbatim |
| m7 | Ch 15 | Mock set hints inconsistently reference pattern IDs |

---

## Round 2 Fix Plan

### Batch A — Critical Content Fixes (C1-C5)
Pattern ID alignment, PAT-25, stream cleanup, SOLID trade-offs, STAR rewrite

### Batch B — Deferred R1 Items (M1-M5)
ZenithTrade durability, Binary Search example, case study framing, HikariCP caveat, Rotated Array placement

### Batch C — Quality & Consistency (M6-M9)
Mock problem dedup, Java 21+ standardization, explanation depth, edge case guards

### Batch D — Strategic (M10-M12 + minors)
Modern tech additions, reference updates, mock transcripts, all minors
