

\part{The Spec-Driven Paradigm}


# Prologue: The Syntax Trap {.unnumbered}

> *"The greatest threat to software craftsmanship is not the speed of the typist, but the direction of their design."*


## The Coding Round Panic

You sit in front of a blank IDE, the timer ticking down. You have seventy minutes to solve four algorithmic challenges on an online assessment platform. Your heart rate rises as you scan the first problem: a convoluted description of array manipulation designed to mimic real-world financial transaction reconciliation. 

Without thinking, you begin typing. You declare variables, nesting loops to handle immediate edge cases. Ten minutes in, you run the initial test suite. Out of twenty test cases, only twelve pass. You patch a conditional check here, mutate a state variable there, and run the tests again. Now, fourteen pass, but two previously passing tests fail. You are caught in the "syntax trap"—the iterative, guessing-based cycle of code modification that degrades design quality in pursuit of green checkboxes.

This is where many experienced software developers, tech leads, and engineering managers fail. They treat coding assessments as a test of speed, syntax recall, and raw typing. They forget that the primary role of a senior engineer is not to type quickly, but to design systems that are secure, reliable, and maintainable.


## The Veteran's Paradox: Returning to the IDE After Decades in Leadership

For professionals who have spent a decade or two in technical leadership, enterprise architecture, or engineering management, returning to live coding assessments represents a unique mental hurdle. You have architected high-throughput financial ledgers, led cross-functional engineering organizations, and managed multi-million-dollar technology budgets. Yet, when faced with a 70-minute timer and a blank editor window, a frustrating cognitive block occurs: your mind goes completely blank. 

You read an algorithmic problem, and conceptually, you understand what it asks. You know it requires a sliding window or a depth-first traversal. But when you place your hands on the keyboard to implement it, the syntax evaporates, the boundary conditions tangle, and the code fails to compile. 

This happens because **algorithmic coding is like mathematics**. You cannot learn calculus or linear algebra by passively reading a textbook or watching someone else solve problems on a whiteboard. Reading a solution creates a deceptive illusion of competence—you nod along, thinking, *"Yes, that makes sense."* But when you pick up the pencil (or open the IDE) to solve a problem from scratch, you realize you have not internalised the mechanics.

Furthermore, attempting to memorize hundreds of specific algorithm solutions is a dangerous trap. Under the stress of a high-stakes assessment, memorized snippets are the first thing to dissolve in your memory. The human brain cannot reliably retrieve hundreds of hyper-specific code blocks under time pressure.

The only effective, sustainable path back to coding mastery is simple:
1. **Understand the core mathematical formulas and invariant patterns** (e.g., the 3-step Sliding Window, the Monotonic Stack sentinel waiting room, the BFS level-by-level queue snapshot).
2. **Analyze the problem structure** to map the requirements to the correct formula rather than guessing.
3. **Practice by doing.** Write out the code independently for two or three exemplar problems of each pattern until the formula becomes pure muscle memory.

When you master the underlying formulas, you no longer need to remember three hundred distinct solutions. You simply recognize the pattern, apply the appropriate code skeleton, and derive the solution cleanly on demand—regardless of how many years you have been away from hands-on programming.


## The Cost of Raw Coding

In professional engineering environments, the cost of the "hack-and-test" mindset is catastrophic. When software is written without defined boundaries and invariants, systems suffer from structural drift, security vulnerabilities, and logic defects. 

Industry data shows that software defects discovered in production cost up to one hundred times more to resolve than those identified during the design phase (National Institute of Standards and Technology [NIST], 2002). In banking-grade environments, a single state-corruption bug in a payment ledger can lead to financial reconciliation failures, regulatory penalties, and reputational damage. Yet, when candidates enter technical interviews, they routinely throw engineering discipline out the window. They write code without pre-conditions, modify state variables without constraints, and build systems that are impossible to reason about. 

This manual is a rejection of that chaos. It is a guide to cracking coding assessments and architecture interviews by applying a rigorous, **spec-driven** approach to software design.


## The Spec-Driven Paradigm

The spec-driven paradigm shifts the focus of the technical interview from coding to design. Instead of jumping directly into implementation, a spec-driven engineer establishes **design invariants** before writing a single line of code.

An invariant is a condition that must always remain true during the execution of a program. By defining these boundaries first, you build an "Invariant Wall" that constrains your implementation, making errors mathematically impossible. When you write code, you are simply translating these formal boundaries into clean, structured prose in your programming language of choice.

![The Spec-Driven Path vs The Syntax Trap](editions/python/chapters/00-prologue/visuals/spec_vs_syntax.png){width=70%}


## What This Book Covers

This manual is organized into four comprehensive parts spanning twenty-five chapters (Chapters 0 through 24), each targeting a specific dimension of modern senior-level technical interviews and enterprise architecture:

### Part I: The Spec-Driven Paradigm for Technical Interviews

- **Chapter 1 — The Invariant-First Strategy:** How to define pre-conditions, post-conditions, and loop invariants before writing any code. Includes a step-by-step mathematical proof of Binary Search correctness.
- **Chapter 2 — The Art of Problem Decomposition:** The 5-Step Decomposition Framework for breaking any novel problem into solvable components mapped to known patterns.
- **Chapter 3 — The Three System-Scale Case Studies:** Introduction to the three enterprise-grade reference architectures (AuraPay, ZenithTrade, ChiramTrust) used throughout the book.

### Part II: Code Design and Craftsmanship

- **Chapter 4 — Principles of Object-Oriented Design:** Self-validating domain entities, rich vs. anemic models, and refactoring walkthroughs.
- **Chapter 5 — SOLID Principles: Enforcing Boundaries:** Interface and dependency boundaries to keep systems modular and decoupled, with a violation detector cheat sheet.
- **Chapter 6 — Modern Functional Programming and Stream APIs:** Clean, declarative data pipelines that minimize side effects, with imperative-vs-stream comparisons and reactive stream explanations.
- **Chapter 7 — Design Patterns in Enterprise Frameworks:** GoF patterns (Builder, Singleton, Observer, State, Strategy), data access patterns (Repository, Unit of Work, DTO, Active Record vs. Data Mapper), and how enterprise frameworks implement them natively.

### Part III: Code Performance & Algorithmic Mastery

- **Chapter 8 — Designing for Performance and Concurrency:** Virtual threads, platform threads, optimistic vs. pessimistic locking, connection pool sizing, caching strategies, and cache invalidation race conditions.
- **Chapter 9 — Core Algorithms & Assessment Tactical Blueprint:** The Assessment Time Allocation Blueprint, pattern recognition decision tree, diagnostic triggers, and canonical code skeletons. Includes an Assessment Format Variants table covering monotonic difficulty, equal-weight peers, single deep problems, take-home projects, and live pair programming.
- **Chapter 10 — Pattern Mastery: Implementation Speed, In-Place Transformations, and String Processing:** Read/Write pointer patterns, character frequency array hashing (`int[26]` / `int[128]`), in-place mutations, and fast string building.
- **Chapter 11 — Pattern Mastery: 2D Matrix Traversal, Grid Simulations, and State Machines:** Matrix coordinate geometry, 90° clockwise rotation formulas, spiral traversals, 2D prefix sums, and flood fill simulation.
- **Chapter 12 — Pattern Mastery: Data Structures, HashMaps, and Sliding Windows:** Complex simulation, HashMap state management, two-pointer sliding window, and frequency tracking.
- **Chapter 13 — Pattern Mastery: Algorithmic Optimization, Monotonic Structures, and Dynamic Programming:** 1-Pass Monotonic Stack (sentinels and width invariants), parametric binary search, 1D/2D DP state compression, and shortest path graph algorithms.
- **Chapter 14 — Mastering Problem Decomposition: The Capstone:** Full deep-dive synthesis chapter with the Problem Analysis Canvas, 15+ decomposition walkthroughs across three tiers, expanded Pattern Recognition Decision Tree, and independent practice exercises.
- **Chapter 15 — 20 Timed Algorithmic Mock Assessment Sets & Survival Guide:** 80 full mock problems across 20 timed sets, complete with hints and the Exam Day 10-Point Speed & Debugging Survival Guide.

### Part IV: System Design, Architecture & Enterprise Leadership

- **Chapter 16 — System Architecture and Design Fundamentals:** DDD bounded contexts, CQRS, CAP theorem trade-offs, consistent hashing, API idempotency, and a full sharded order matching engine mock interview transcript.
- **Chapter 17 — Enterprise Integration and Resiliency:** Transactional Outbox, Saga orchestration vs. choreography, event sourcing, Redis sliding window rate limiting, Circuit Breaker state machines, and OpenTelemetry distributed tracing.
- **Chapter 18 — Database Design, Compliance, and Security:** B-Tree vs. LSM-Tree storage engines, PCI-DSS tokenization vaults, SOC2 cryptographic audit trails, GDPR Crypto-Shredding, and sharding strategies.
- **Chapter 19 — Behavioral and Technical Leadership Interviews:** The Technical STAR Framework, video/Teams call checklists, and three full mock responses for senior leadership scenarios.
- **Chapter 20 — Testing and CI/CD Strategies:** The testing pyramid (unit, integration via Testcontainers, contract via Pact), and automated CI/CD release policies.
- **Chapter 21 — Distributed Event Streaming and Message Brokers:** Apache Kafka internals, partition-key sharding for in-order delivery, consumer group rebalancing, and Exactly-Once Semantics (EOS).
- **Chapter 22 — AI/ML System Design and LLM Integration:** Vector databases (HNSW vs. IVF indexes), Retrieval-Augmented Generation (RAG) pipelines, semantic caching, and prompt injection security filters.
- **Chapter 23 — Appendix and Quick-Reference Cheat Sheets:** Big-O complexity tables, edge-case checklists, system design latency numbers, and day-of-interview preparation guides.
- **Chapter 24 — Works Cited and Academic References:** Primary scholarly and technical citations supporting all architectural principles and benchmarking claims.


## How to Read This Book: Persona Profiles

To maximize the value of this manual, select the path that aligns with your career stage and current interview goals:

### Persona A: The Mid-to-Senior Engineer (Target: Coding Assessments)

- **Goal:** Clear timed coding assessments, optimize runtime performance, and handle live coding screens without panic.
- **Recommended Reading Path:**
  1. Read **Chapter 1 (Invariant-First Strategy)** and **Chapter 2 (Problem Decomposition)** to learn the foundational analysis discipline.
  2. Skip to **Part III (Chapters 8 through 15)**. Master the Assessment Tactical Blueprint in Chapter 9, the Pattern Mastery deep dives in Chapters 10–13, the Capstone synthesis in Chapter 14, and complete the 20 Mock Sets in Chapter 15.
  3. Study **Part II (Chapters 4 & 6)** to learn functional stream optimizations and rich data structures.
  4. Review **Chapter 23 (Appendix)** for the Big-O cheat sheet and edge-case checklist before your assessment.

### Persona B: The Lead / Staff Engineer (Target: System Design & Craftsmanship)

- **Goal:** Design clean microservices, establish domain boundaries, and explain complex distributed system tradeoffs to principal engineers.
- **Recommended Reading Path:**
  1. Read **Part I (Chapters 1–3)** to align on the invariant-first strategy, problem decomposition, and case studies.
  2. Master **Part II (Chapters 4–7)** on rich aggregate boundaries, strict SOLID inversion, and enterprise design patterns.
  3. Deep-dive into **Part IV (Chapters 16–18 & 20–22)**. Study the sharded order matching engine mock script, distributed Saga implementations, Kafka event streaming, and security compliance (PCI-DSS, SOC2, GDPR).

### Persona C: The Engineering Manager / Director (Target: Architectural Strategy & Leadership)

- **Goal:** Evaluate team engineering standards, design resilient systems, and ensure operational compliance under regulatory frameworks.
- **Recommended Reading Path:**
  1. Read **Chapter 3 (Case Studies)** for enterprise system context.
  2. Study **Chapter 5 (SOLID boundaries)** to establish code quality metrics for your team.
  3. Focus on **Part IV (Chapters 16–18)**. Master the CAP theorem tradeoffs, disaster recovery models, rate-limiting patterns, and GDPR Crypto-Shredding architectures.
  4. Read **Chapter 19 (Behavioral & Technical Leadership)** to prepare for the behavioral round with Technical STAR frameworks and full mock responses.


> ⭐ **STAR Moment: The Invariant Principle**
> 
> The best code is code that is correct by design. When you write a method, your first task is not to implement the algorithm, but to define the contract: what must be true *before* the method runs (pre-conditions), and what must be guaranteed *after* it completes (post-conditions). If you enforce these boundaries, the code inside the method almost writes itself.


## How to Use This Book

This manual is designed for a dual audience. For individual engineers preparing for standardized online coding assessments (such as CodeSignal, HackerRank, Codility, or employer-proprietary platforms), it provides a concrete, pattern-based approach to conquer algorithmic challenges under severe time constraints. For engineering leads and managers returning to coding assessments after years of management, it serves as a tactical refresher to translate high-level architectural knowledge back into executable, robust code. Treat this not just as a book, but as a systematic training plan.

## The 14-Day Algorithmic Sprint (Persona A)

For mid-to-senior engineers targeting algorithmic assessments. Follow this intensive schedule to rebuild coding muscle memory.

| Day | Focus Area | Chapters | Practice Target | Time |
|:---:|:------------------------|:----------------------|:-------------------------------------------------------|:-----:|
| 1 | Foundations | Prologue, Ch 1-2 | Read Invariant-First strategy & Decomposition | 3-4 hrs |
| 2 | Core Algorithms | Ch 8-9 | Memorize Big-O table, implement 5 core algorithms | 3-4 hrs |
| 3 | Easy-Tier Patterns | Ch 10 | Solve 15 implementation problems under 8-min timer | 4-5 hrs |
| 4 | Medium-Tier Grid | Ch 11 | Solve 10 matrix/grid problems under 15-min timer | 4-5 hrs |
| 5 | Medium-Tier Window | Ch 12 | Solve 10 sliding window/hashmap problems | 4-5 hrs |
| 6 | REST DAY | Review weak areas | Light review only | 1-2 hrs |
| 7 | Hard-Tier Patterns | Ch 13 | Solve 8 DP/graph problems | 4-5 hrs |
| 8 | Decomposition Capstone | Ch 14 | Capstone walkthroughs | 3-4 hrs |
| 9 | Mock Exam Day 1 | Ch 15 Sets 1-4 | Full timed sessions (4 sets) | 4 hrs |
| 10 | Mock Exam Day 2 | Ch 15 Sets 5-8 | Full timed sessions (4 sets) | 4 hrs |
| 11 | Mock Exam Day 3 | Ch 15 Sets 9-12 | Full timed sessions (4 sets) | 4 hrs |
| 12 | Mock Exam Day 4 | Ch 15 Sets 13-16 | Full timed sessions (4 sets) | 4 hrs |
| 13 | Mock Exam Day 5 | Ch 15 Sets 17-20 | Full timed sessions (4 sets) | 4 hrs |
| 14 | Final Review & Prep | Ch 23 Appendix | Final review and preparation | 4-5 hrs |

## The 14-Day System Design Sprint (Persona B)

For lead and staff engineers focused on system design and architecture.

| Day | Focus Area | Chapters | Practice Target | Time |
|:---:|:------------------------|:----------------------|:-------------------------------------------------------|:-----:|
| 1 | Foundations & Case Studies | Prologue, Ch 1-3 | Internalize case studies and design boundaries | 3-4 hrs |
| 2 | OOP & SOLID | Ch 4-5 | Domain boundaries and strict SOLID inversion | 3-4 hrs |
| 3 | Functional Streams | Ch 6 | Imperative-vs-stream optimizations | 2-3 hrs |
| 4 | Design Patterns | Ch 7 | Enterprise framework pattern recognition | 3-4 hrs |
| 5 | Architecture Fundamentals | Ch 16 | System boundaries and API design | 4-5 hrs |
| 6 | REST DAY | Review weak areas | Light review only | 1-2 hrs |
| 7 | Integration & Resiliency | Ch 17 | Outbox, Saga, rate limiting, distributed tracing | 4-5 hrs |
| 8 | Database Design | Ch 18 | Storage engines, sharding, compliance | 4-5 hrs |
| 9 | Leadership & Testing | Ch 19-20 | STAR frameworks and CI/CD policies | 4 hrs |
| 10 | Event Streaming | Ch 21 | Kafka internals, exactly-once semantics | 4 hrs |
| 11 | AI/ML Design | Ch 22 | Vector DBs and RAG pipelines | 4 hrs |
| 12 | Mock Interview Prep 1 | Ch 16-18 Review | Practice mock design sessions | 4 hrs |
| 13 | Mock Interview Prep 2 | Ch 19-22 Review | Practice mock design sessions | 4 hrs |
| 14 | Final Review | Ch 23 Appendix | Final exam preparation | 4 hrs |

- **Start each day** by reviewing the terminology section of the relevant chapter.
- **Keep a 'mistake log'** to track patterns you consistently get wrong.
- **On rest day**, revisit your mistake log, not new material.

### 14-Day Architectural Strategy Sprint (Persona C: Engineering Manager/Director)

You lead teams but haven't personally coded in assessments recently. Your edge is architectural judgment and leadership — this plan leverages that while rebuilding algorithmic fluency.

| Day | Focus | Chapters | Time |
|:---:|:-------------------------------------------------------|:---------|:----:|
| 1 | Invariant-First Mindset + Decomposition Framework | Ch 1-2 | 2h |
| 2 | Case Study Architectures (AuraPay, ZenithTrade) | Ch 3 | 1.5h |
| 3 | SOLID Trade-offs + Design Patterns (Strategic View) | Ch 5, 7 | 2h |
| 4 | Concurrency & Connection Pool Sizing | Ch 8 | 2h |
| 5 | Pattern Catalog: Top 10 Most-Asked (PAT-01 to PAT-10) | Ch 9 | 2.5h |
| 6 | Implementation Drill: Arrays + HashMaps | Ch 10, 12 | 2h |
| 7 | Mock Assessment Set 1-3 (Timed) | Ch 15 | 2h |
| 8 | System Architecture Deep Dive | Ch 16 | 2.5h |
| 9 | Resiliency + Database Compliance | Ch 17-18 | 2h |
| 10 | Behavioral Leadership: STAR Framework + All 5 Scenarios | Ch 19 | 2h |
| 11 | Message Brokers + AI/ML Architecture | Ch 21-22 | 2h |
| 12 | Mock Assessment Set 4-6 (Timed) + Review Weak Patterns | Ch 15, 9 | 2.5h |
| 13 | System Design Mock: Pick 2 Consumer Archetypes | Ch 16 | 2h |
| 14 | Full Mock Day: 1 Coding Assessment + 1 System Design + 1 Behavioral | Ch 15, 16, 19 | 3h |

**Manager's Edge:** On Days 8-11, practice explaining your architectural decisions aloud. Interviewers evaluate managers on communication clarity as much as technical depth. On Day 14, simulate a full interview loop with time pressure.

## The 28-Day Comprehensive Plan (All Personas)

For candidates targeting roles requiring thorough mastery of both coding and system design.

**Week 1: Foundations & Design Thinking (Personas A, B, C)**

- Day 1-2: Invariants, Decomposition, Case Studies, OOP (Ch 1-4)
- Day 3-4: SOLID, Streams, Design Patterns (Ch 5-7)
- Day 5-6: Concurrency, Core Algorithms Blueprint (Ch 8-9)
- Day 7: Review + implement 10 algorithms from memory

**Week 2: Algorithm Mastery (Persona A Focus)**

- Day 8-9: Easy-Tier patterns (Ch 10) — solve ALL exemplar problems
- Day 10-11: Medium-Tier patterns (Ch 11) — solve ALL exemplar problems
- Day 12-13: Medium-Hard patterns (Ch 12) — solve ALL exemplar problems
- Day 14: Hard-Tier patterns (Ch 13) — start with 15 problems

**Week 3: Advanced Algorithms + System Design (Personas A, B, C)**

- Day 15-16: Finish Hard-Tier patterns (Ch 13) + Capstone Decomposition (Ch 14) (Persona A)
- Day 17-18: Mock assessments (Ch 15 Sets 1-10, two per day) (Persona A)
- Day 19-20: System Architecture (Ch 16), Resiliency (Ch 17), Database Design (Ch 18) (Persona B, C)
- Day 21: Review + identify weakest algorithm pattern

**Week 4: Polish & Exam Readiness (Personas B, C Focus)**

- Day 22-23: Behavioral Leadership (Ch 19) + Testing/CI-CD (Ch 20)
- Day 24-25: Message Brokers (Ch 21), AI/ML (Ch 22) + final mock assessments (Ch 15 Sets 11-20)
- Day 26-27: Full review — re-solve all problems you got wrong
- Day 28: Final full mock assessment under strict conditions + rest

## Pattern Recognition Quick Reference

When you read a problem under pressure, use this decision tree to instantly map the specification to the correct invariant-first design:

1. **Is it asking for a single pass through an array/string?** → Two-pointer or sliding window
2. **Does it involve a sorted array or search space?** → Binary search
3. **Does it need grouping or counting?** → HashMap frequency signature
4. **Is it a 2D grid problem?** → BFS/DFS with direction vectors
5. **Does it ask for 'minimum/maximum' of something?** → DP or greedy
6. **Does it involve intervals?** → Sort by start/end, then merge/sweep
7. **Does 'next greater/smaller' appear?** → Monotonic stack
8. **Does it ask for 'all combinations/permutations'?** → Backtracking
9. **Is it about connected components?** → Union-Find or DFS
10. **Does it have dependencies/ordering?** → Topological sort


# The Invariant-First Interview Strategy

> *"Before you build a system, draw the boundaries. Code written within correct boundaries cannot drift into error."*


## The Panic of the Blank Editor

It is a common scenario in technical interviews: the interviewer presents a coding challenge, and the candidate immediately starts typing. They construct loops, initialize local counters, and write complex nested conditionals. The candidate is trying to solve the problem by writing code, using the editor as a scratching post to find a solution.

This approach is fragile. In the pressure of a live interview or a timed online assessment (like CodeSignal), writing code without a design roadmap leads to cognitive overload. You are trying to manage algorithmic logic, syntax rules, memory allocation, and edge cases simultaneously. When the first test run fails, you start modifying conditions arbitrarily—changing `<` to `<=`, adding random `+1` offsets, or introducing temporary boolean flags. 

This is the "hack-and-test" methodology, and it signals to the interviewer that you lack structural discipline. A senior engineer or manager must demonstrate a systematic, predictable approach to code correctness. The solution is the **Invariant-First Strategy**.





## Defining the Invariant Wall

The Invariant-First Strategy requires you to define the mathematical and logical boundaries of your solution before implementing any code. In computer science, an **invariant** is a property that remains true throughout a specific phase of execution. 

When you apply this to coding assessments, you construct an "Invariant Wall" composed of three layers:

1.  **Pre-conditions:** Constraints on the inputs that must be true before a function or method is executed. If a caller violates a pre-condition, the method should fail immediately (e.g., throwing an `IllegalArgumentException` in Java).
2.  **Post-conditions:** Guarantees that the method promises to satisfy upon successful execution. This defines what "correctness" means for the operation.
3.  **Class/Data Invariants:** State rules that must always hold true for a domain object throughout its entire lifecycle.

![The Invariant Wall](editions/python/chapters/01-invariant-first/visuals/invariant_wall.png){width=70%}

By declaring these boundaries upfront, you decouple *what* the system must do from *how* it will do it. You establish a contract. Once the contract is clear, writing the code is simply a matter of executing that contract.


## The Invariant Interview Framework

When faced with a technical coding challenge in an interview, follow this four-step spec-driven framework:

### Define the Boundary Invariants (Clarification Phase)
Before writing any code, state the inputs, outputs, and their mathematical bounds. For example, if you are asked to process a list of transactions:

- What are the pre-conditions? Can the input list be null or empty? Can transaction amounts be negative?
- What are the post-conditions? Does the output preserve the original order of transactions? How are duplicate entries handled?
Write these down as comments in the IDE or verbalize them to the interviewer.

### Declare the Data Invariants (Type Design Phase)
Design your types to enforce invariants natively. Do not use generic types (like raw integers or strings) where a domain-specific type can prevent invalid states. 
For example, instead of passing a raw `double` representing a monetary amount, define a `Money` record that guarantees the amount cannot be negative and uses the correct currency scale.

### Establish the Loop Invariants (Algorithmic Phase)
If your algorithm requires an iterative process (such as a search or a sliding window), define what remains true during each iteration of the loop.
For instance, in a sliding window algorithm finding the maximum subarray sum:

- *Loop Invariant:* At the start of each iteration `i`, `current_window_sum` represents the sum of elements from index `left` to `i - 1`.
If you can maintain this invariant, your loop is guaranteed to be correct, and off-by-one errors are eliminated.

### Implement and Enforce
Write the code, beginning with explicit checks for your pre-conditions. Use modern language features to keep the code clean and expressive, ensuring that your logic never violates the declared invariants.


## Worked Example: Proving Binary Search

To demonstrate the mathematical power of invariants, let us examine the classic binary search algorithm. Many developers struggle with binary search, often getting trapped in infinite loops or off-by-one errors because they guess the boundary updates (e.g., `right = mid` vs. `right = mid - 1`).

![Loop Invariant States — Boundary Contraction in Binary Search](editions/python/chapters/01-invariant-first/visuals/loop_invariant_states.jpg){width=85%}

### The Challenge
Given a sorted array of integers `nums` and a `target` value, return the index of the `target` if it exists in the array, or `-1` if it does not.

### Define the Boundaries (Step 1)

- **Pre-condition:** `nums` is sorted in ascending order.
- **Post-condition:** The returned index $idx$ satisfies $nums[idx] == target$, or if $idx == -1$, then $target \notin nums$.

### Establish the Loop Invariant (Step 3)
We define two pointers, `left` and `right`, defining our active search range $[left, right]$.

- **The Loop Invariant:** *If target is present in the array, it must reside within the index boundaries:*

```
Invariant P(left, right): target in nums[left...right]
```

### Mathematical Proof of Correctness
To prove the algorithm is correct, we must prove three properties of our loop invariant:

#### A. Initialization
Before the loop starts, the invariant must hold true. We initialize `left = 0` and `right = nums.length - 1`.

- Since the array is sorted, if the target is in the array, it must be within the range $[0, nums.length - 1]$. The invariant holds.

#### B. Maintenance
If the invariant is true before an iteration, we must prove it remains true after updating our pointers.
During the loop, we calculate:

```
mid = left + (right - left) / 2
```

We check three cases:

**Case 1: $nums[mid] == target$**

The target is found, and we return `mid`, satisfying the post-condition.

**Case 2: $nums[mid] < target$**

Since the array is sorted, all elements at or to the left of `mid` are strictly less than the target. Therefore, the target cannot reside in the range $[left, mid]$.

We update `left = mid + 1`. The new range is $[mid + 1, right]$. If the target exists, it must lie within this new range. The invariant is maintained.

**Case 3: $nums[mid] > target$**

All elements at or to the right of `mid` are strictly greater than the target. The target cannot reside in the range $[mid, right]$.

We update `right = mid - 1`. The new range is $[left, mid - 1]$. The invariant is maintained.

#### C. Termination
When the loop terminates, the invariant must help us prove correctness.
The loop terminates when `left > right`.

- If `left > right`, the search range $[left, right]$ has become empty.
- Combining this with our loop invariant (which states that if the target is present, it must lie within $[left, right]$), we prove that the target is **not** present in the array. We return `-1` with mathematical confidence.

### Implementation (Step 4)
Because we have proved our updates mathematically, we do not need to guess the loop conditions:

```python
def binary_search(nums: list[int], target: int) -> int:
    # 1. Enforce Pre-conditions
    if not nums:
        return -1

    left = 0
    right = len(nums) - 1

    # Maintain Invariant: target is in nums[left...right]
    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid  # Post-condition satisfied
        elif nums[mid] < target:
            left = mid + 1  # Invariant maintained
        else:
            right = mid - 1  # Invariant maintained

    return -1  # Search range is empty -> target not in nums
```


By applying this invariant-first approach, we eliminate all cognitive overhead. We do not need to "dry-run" multiple edge cases or guess boundary updates. The math guarantees the correctness of our implementation.

### Invariant Proof #2: The Sliding Window Maximum

Prove the invariant for maintaining a monotonic deque that tracks the maximum element in a sliding window of size K:

**Invariant:** At every step, the deque contains indices in strictly decreasing order of their corresponding values, and all indices are within the current window [i-K+1, i].

**Initialization:** The deque is empty before processing begins. Vacuously true.
**Maintenance:** When processing element A[i]:
1. Remove all indices from the back where A[deque.peekLast()] ≤ A[i] (maintains decreasing order)
2. Remove the front if deque.peekFirst() < i-K+1 (maintains window bounds)
3. Add i to the back

After these operations, deque.peekFirst() always holds the index of the maximum element in the current window.

**Termination:** After processing all N elements, we have extracted N-K+1 window maximums, each in O(1) amortized time.

This proves the Monotonic Deque pattern [PAT-20] achieves O(N) total time for sliding window maximum.


> ⭐ **STAR Moment: The $O(1)$ Failure Principle**
> 
> A robust system fails fast and fails explicitly. The first lines of any method should always be pre-condition validation. If an input is invalid, fail immediately. Do not allow execution to proceed with corrupted or unexpected state, as this leads to hard-to-debug failures deep inside your call stack. In an interview, writing explicit input validations shows that you design for production safety, not just passing test suites.


# The Art of Problem Decomposition

> *"The ability to decompose a novel problem into solvable components is the single most valuable skill a software engineer can demonstrate under assessment conditions."*

## Why Decomposition Matters

In the high-stakes environment of technical assessments, the most common trap engineers fall into is the pursuit of memorization. Memorizing solutions to hundreds of common interview questions might give a false sense of security, but it invariably fails when confronted with novel, unique, or subtly modified problems. The real skill—the one that distinguishes top-tier candidates—is not recall, but the ability to break any complex, unfamiliar problem into a series of recognizable, solvable sub-problems that map directly to known patterns.

![Problem Decomposition Tree — Breaking Complex Problems into Sub-Problems](editions/python/chapters/02-problem-decomposition/visuals/decomposition_tree.jpg){width=85%}

This principle applies universally across all assessment formats. Whether you are facing a monotonically increasing difficulty curve, equal-weight peer questions, a single deep architectural problem, or a live whiteboard interview, decomposition remains your primary analytical tool. When you encounter a question you have never seen before, your memorized catalog of answers is useless. However, your ability to dismantle that question into its atomic components is exactly what the assessment is designed to measure.

Mastering problem decomposition transitions your mindset from "Have I seen this before?" to "What are the underlying structures of this problem?" It transforms an insurmountable challenge into a structured exercise in pattern recognition and application.

## The 5-Step Decomposition Framework

To systematically dismantle any technical problem, you must adhere to a rigorous analytical process. The following 5-step framework is designed for senior-level decomposition, preventing premature coding and ensuring a comprehensive understanding of the problem domain.

### Step 1: Constraint Analysis

Extract time and space bounds directly from the constraints to narrow the algorithm class before you even read the problem narrative. For example, if $N \le 10^5$, an $O(N^2)$ brute-force solution will fail immediately due to time limits. You are mathematically required to find an $O(N \log N)$ or $O(N)$ solution. If $N \le 20$, an $O(2^N)$ backtracking approach is expected. The constraints are not trivia; they are the architectural specifications of your solution.

### Step 2: Data Flow Mapping

Trace the input-to-output transformations to identify the structural nature of the problem. Is this a mapping operation (1:1 transformation)? A reduction operation (N:1 aggregation)? Or a search operation (finding a needle in a haystack)? By mapping the data flow, you constrain the types of data structures that can be used.

### Step 3: Invariant Identification

Define what property must remain mathematically true across iterations. This is the core thesis of the Invariant-First strategy. Whether you are maintaining a sorted boundary in a two-pointer approach, or a monotonic property in a stack, identifying the invariant reduces the algorithm to a simple proof of correctness rather than a guessing game.

### Step 4: Pattern Matching

With constraints, data flow, and invariants defined, map these characteristics to the 24 canonical patterns (Chapter 9). You are no longer inventing an algorithm; you are selecting the appropriate structural blueprint that satisfies the defined bounds.

### Step 5: Edge Case Enumeration

Systematically generate boundary inputs based on the constraints. What happens at $N=0$ or $N=1$? What if the input array contains negative values or duplicates? Enumerating edge cases before implementation guarantees your invariant holds at the boundaries.

## A Quick Decomposition Example

Let us walk through a concrete example using the framework. Consider this problem: 

**"Given an array of non-negative integers representing the heights of adjacent buildings of unit width, compute how much rainwater can be trapped between the buildings after a storm."**


**Step 1: Constraint Analysis**
Assume $N \le 10^5$. This instantly rules out any $O(N^2)$ solution. We must solve this in $O(N)$ or $O(N \log N)$ time.

![Constraint-to-Complexity Flowchart](editions/python/chapters/02-problem-decomposition/visuals/constraint_flowchart.jpg){width=85%}

**Step 2: Data Flow Mapping**
Input: Array of $N$ heights. Output: A single integer (total water). This is a reduction problem. For any building `i`, the water it traps is `min(max_left, max_right) - height[i]`.

**The Failed Naive Approach ($O(N^2)$)**
A junior engineer might immediately code a loop within a loop: for every element `i`, iterate left to find `max_left`, and iterate right to find `max_right`. 
*Why it fails:* Scanning the remaining array for every single element yields $O(N^2)$ time complexity. With $N=10^5$, this requires $10^{10}$ operations, which will time out on any assessment platform.

**Step 3: Invariant Identification**
To achieve $O(N)$, we must eliminate the inner loops. The amount of water trapped depends *only on the shorter of the two maximum boundaries*. 
*Invariant:* If we have two pointers (`left` and `right`), and `height[left] < height[right]`, the trapped water at `left` is strictly bounded by `max_left`, regardless of what happens between `left` and `right`. We can safely process `left` and move inward.

**Step 4: Pattern Matching**
Processing an array from the outsides inward based on boundary conditions maps perfectly to **[PAT-06] Converging Two-Pointers**.

**Step 5: Edge Case Enumeration**
- $N < 3$: Cannot trap water. Return 0.
- All heights equal: Return 0.

**Design Before Coding**
*Approach (Two-Pointer Design):*
- Initialize `left` at 0, `right` at $N-1$.
- Maintain `left_max` and `right_max`.
- While `left < right`:
  - If `heights[left] < heights[right]`, water depends on `left_max`. Update `left_max`, add `left_max - heights[left]` to total, increment `left`.
  - Else, water depends on `right_max`. Update `right_max`, add `right_max - heights[right]` to total, decrement `right`.
- Time Complexity: $O(N)$, Space Complexity: $O(1)$.

By following the framework, a potentially paralyzing problem is reduced to a standard application of the Two-Pointer pattern.

## When Decomposition Saves You

In modern assessment environments, particularly equal-weight assessments where all questions are peers, decomposition is your greatest strategic weapon. Because these formats do not provide difficulty-ordering cues, you cannot rely on the assumption that "Question 1 is easy, Question 4 is hard." You must approach every problem objectively.

When confronted with novel, never-before-seen problems—problems explicitly designed to test engineering limits rather than memorization—decomposition is the *only* reliable strategy. It bridges the gap between the unknown problem domain and your known catalog of patterns, ensuring that you can always make structured, demonstrable progress.

> ⭐ **STAR Moment: The Decomposition Discipline**
>
> Before you write a single line of code, invest 3-5 minutes in decomposition. Write your analysis as comments at the top of your solution file. This serves three purposes: it clarifies your thinking, it provides partial credit if you run out of time, and it creates a roadmap that prevents you from getting lost during implementation.


# The Three System-Scale Case Studies

> *"If you want to evaluate an engineer's design skill, do not ask them about theory. Ask them to design a ledger, an exchange, or a wallet under high-concurrency and security constraints."*


## AuraPay: Core Ledger & Asynchronous Settlement (Canonical)

AuraPay is the primary case study we will implement throughout this book. It is a distributed, banking-grade payment ledger and asynchronous settlement system. 

### Key System Requirements

- **Double-Entry Bookkeeping:** All ledger updates must obey double-entry rules (every debit must have a corresponding credit, and the net balance change of any transaction across the system must be exactly zero).
- **ACID Transaction Isolation:** The ledger must prevent race conditions and double-spending, maintaining strict consistency even under heavy concurrent load on "hot" accounts.
- **Asynchronous Settlement Routing:** Payments are routed to different processing networks (ACH, FedWire, Visa/Mastercard) based on speed, cost, and transaction limits.

### Enforcing the Domain Invariants
To demonstrate the spec-driven approach, we begin by defining the core domain objects of AuraPay: the `TransactionRecord` (an immutable value object representing a transaction intent) and the `LedgerAccount` (a stateful entity enforcing balance and overdraft invariants).

Here is the immutable, self-validating transaction representation:

```python
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from uuid import UUID

@dataclass(frozen=True)
class TransactionRecord:
    """
    Represents an immutable, validated financial transaction record in AuraPay.
    Enforces pre-conditions on initialization.
    """
    transaction_id: UUID
    source_account_id: UUID
    destination_account_id: UUID
    amount: Decimal
    currency: str
    timestamp: datetime

    def __post_init__(self):
        if not self.transaction_id or not self.source_account_id or not self.destination_account_id:
            raise ValueError("Account IDs and Transaction ID cannot be null")
        if not self.amount or not self.currency or not self.timestamp:
            raise ValueError("Amount, currency, and timestamp cannot be null")
        if self.source_account_id == self.destination_account_id:
            raise ValueError("Source and destination accounts must be distinct")
        if self.amount <= 0:
            raise ValueError("Transaction amount must be strictly positive")
        if not self.currency.strip():
            raise ValueError("Currency code cannot be empty")
```


Next, we define the stateful `LedgerAccount` that enforces balance boundaries and thread-safe operations during fund transfers:

```python
from decimal import Decimal
from uuid import UUID
import threading

class LedgerAccount:
    """
    Represents a stateful Ledger Account in AuraPay, enforcing business invariants
    during state transitions.
    """
    def __init__(self, account_id: UUID, currency: str, initial_balance: Decimal, overdraft_limit: Decimal):
        if not account_id or not currency:
            raise ValueError("Account ID and Currency cannot be null")
        if initial_balance is None or overdraft_limit is None:
            raise ValueError("Initial balance and overdraft limit cannot be null")
        if overdraft_limit < 0:
            raise ValueError("Overdraft limit cannot be negative")
        if initial_balance + overdraft_limit < 0:
            raise ValueError("Initial balance violates the overdraft limit")

        self.account_id = account_id
        self.currency = currency
        self._balance = initial_balance
        self.overdraft_limit = overdraft_limit
        self._lock = threading.Lock()

    @property
    def balance(self) -> Decimal:
        with self._lock:
            return self._balance

    def credit(self, amount: Decimal):
        """Credits the account. Enforces positive credit amount."""
        if amount is None or amount <= 0:
            raise ValueError("Credit amount must be positive")
        with self._lock:
            self._balance += amount

    def debit(self, amount: Decimal):
        """Debits the account. Enforces balance invariants and overdraft limits."""
        if amount is None or amount <= 0:
            raise ValueError("Debit amount must be positive")
        
        with self._lock:
            new_balance = self._balance - amount
            # INVARIANT ENFORCEMENT
            if new_balance + self.overdraft_limit < 0:
                raise ValueError(
                    f"Debit of {amount} exceeds account overdraft boundary. "
                    f"Balance: {self._balance}, Limit: -{self.overdraft_limit}"
                )
            self._balance = new_balance
```


![AuraPay System Architecture](editions/python/chapters/03-case-studies/visuals/aurapay_architecture.png){width=80%}

In the following chapters, we will use these domain classes to demonstrate OOP design, SOLID boundary enforcement, Java Streams collection processing, and database concurrency controls.


## ZenithTrade: High-Frequency Matching Engine (Reference Architecture)

ZenithTrade is a high-frequency, low-latency order matching engine. It is designed to process incoming buy and sell limit orders and execute matches in real time.

![ZenithTrade High-Frequency Matching Engine Architecture](editions/python/chapters/03-case-studies/visuals/zenithtrade_architecture.jpg){width=85%}

### Key System Requirements

- **Order Book State:** Maintains separate buy (bid) and sell (ask) order books, sorted by price (highest bid first, lowest ask first) and arrival time (FIFO).
- **Sub-Millisecond Latency:** The engine must execute order matching with minimal latency, avoiding memory allocations and garbage collection pauses.
- **Data Structure Mastery:** Utilizes custom priority queues, heaps, and double-ended queues for low-overhead bookkeeping.

### Reference Architecture Starter Scaffolding
To begin implementing the ZenithTrade engine, use the following `Order` entity as your starting point. It establishes the basic structure of a limit order, enforcing invariants like positive price and quantity:

```python
from enum import Enum

class Side(Enum):
    BUY = 0
    SELL = 1

class Order:
    def __init__(self, id: str, instrument_id: str, side: Side, price: int, quantity: int):
        if price <= 0:
            raise ValueError("Price must be positive")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        self.id = id
        self.instrument_id = instrument_id
        self.side = side
        self.price = price # Fixed-point integer
        self.quantity = quantity
```


These architectures serve as running case studies throughout the book. You will implement components of each system as you learn the patterns in Parts II, III, and IV. Do not attempt to design these systems now — let the patterns guide you.


## ChiramTrust: Decentralized Identity Consent Wallet (Reference Architecture)

ChiramTrust is a decentralized identity wallet that allows users to store credentials locally, negotiate sharing terms with verifiers, and establish consensus-based recovery.

![ChiramTrust Decentralized Identity Wallet Architecture](editions/python/chapters/03-case-studies/visuals/chiramtrust_architecture.jpg){width=85%}

### Key System Requirements

- **W3C DID Compatibility:** Supports W3C Decentralized Identifiers (DIDs) for verifying cryptographic signatures on claims.
- **Granular Consent Engine:** Enforces user-defined access scopes, ensuring verifiers only receive requested claims (e.g., age verification without sharing birth dates).
- **Consensus Recovery:** Shares cryptographic key shards across a network of trusted guardians, using threshold secret sharing (Shamir's) to recover lost keys.

### The Mechanics of Threshold Consensus (Shamir's Secret Sharing)

To implement consensus-based key recovery, the user's private key $S$ is split into $N$ distinct shares. We construct a random polynomial of degree $T - 1$ (where $T$ is the threshold of guardians needed to recover the key):

```
f(x) = a_0 + a_1*x + a_2*x^2 + ... + a_{T-1}*x^{T-1} (mod P)
```

where $a_0 = S$ (the secret key), and the coefficients $a_1, \dots, a_{T-1}$ are randomly generated integers. The prime $P$ defines the finite field $\mathbb{F}_P$. Each guardian $i$ receives a coordinate point $(i, f(i))$. 

By the properties of polynomial interpolation:

1.  **Any $T$ guardians** can pool their shares $(x_i, y_i)$ and reconstruct the polynomial $f(x)$ using Lagrange interpolation, finding $f(0) = a_0 = S$:
   
```
S = Sum_{i=1..T} ( y_i * Product_{j != i} ( -x_j / (x_i - x_j) ) ) (mod P)
```
   
2.  **Any $T - 1$ or fewer guardians** possess a system of equations with infinite solutions, revealing absolutely zero information about the secret key $S$.

### Reference Architecture Starter Scaffolding

To implement the ChiramTrust wallet, use the following `DidConsentRecord` aggregate root as your starting point. It handles W3C identifier validation and thread-safe consent scope modifications:

```python
class DidConsentRecord:
    def __init__(self, did: str, consent_scopes: dict[str, bool]):
        if not did or not did.startswith("did:"):
            raise ValueError("Invalid W3C DID format")
        self.did = did
        self._consent_scopes = dict(consent_scopes)

    def has_consent(self, scope: str) -> bool:
        return self._consent_scopes.get(scope, False)

    def revoke_consent(self, scope: str) -> None:
        self._consent_scopes[scope] = False
```


### Interview Drill: Applying Bounded Context Isolation

Here is a mock interview dialogue showing how to apply the Bounded Context Isolation rule in a real design interview:

**Interviewer:** *"If the AuraPay Ledger database experiences a write lag or becomes temporarily unavailable, how does that affect ZenithTrade's matching engine? How do you prevent ledger issues from cascading and bringing down the trading platform?"*

**Candidate:** "We enforce strict Bounded Context Isolation. The ZenithTrade matching engine runs entirely in-memory and communicates with the AuraPay Ledger asynchronously via a transaction event stream. When an order matches, the matching engine commits the trade to its local state and publishes a `TradeExecuted` event. The Ledger service consumes this event and updates account balances. 

To ensure zero-loss durability, ZenithTrade employs a write-ahead journal (WAJ) inspired by the LMAX Disruptor architecture. Every order and match event is sequentially appended to a persistent ring buffer on NVMe storage BEFORE the in-memory state is updated. On node failure, the engine replays the journal to reconstruct its complete order book state. Additionally, periodic snapshots compress the journal, enabling sub-second recovery times. This design achieves both the microsecond latency of in-memory processing and the durability guarantees required by financial regulators."

If the Ledger database slows down or halts, the matching engine continues to process trades in memory without interruption. The event broker queues the trade events until the ledger recovers. This decoupling guarantees fault isolation and maintains a high-availability trading path."

> ⭐ **STAR Moment: Bounded Context Isolation**
> 
> During system design interviews, explain that microservice division should mirror DDD Bounded Contexts. Say: *"We will isolate the ZenithTrade Matching Engine from the AuraPay Ledger. If the ledger experiences a database write lag, our matching engine can continue to accept and queue orders in memory, preventing system-wide downtime."* This shows you design for fault isolation.


\part{Code Design and Craftsmanship}


# Principles of Object-Oriented Design

> *"Do not expose your state to the world. Encapsulate your data, expose your contracts, and let polymorphism handle the variance."*


## The Anemic Domain Model Anti-Pattern

In many enterprise applications, domain classes are treated as passive data holders—simple collections of fields with auto-generated getters and setters. This is the **Anemic Domain Model** anti-pattern. 

When your domain models are anemic, the business logic shifts into stateless service classes (e.g., `LedgerService`). The service pulls the state out of the domain model, performs validation, modifies the fields, and pushes the data back to the database. The danger of this design is that the domain object itself has no control over its state. Any developer can instantiate a ledger account, set the balance to a negative value without checks, and persist it, violating the core safety boundaries of the system.

![God Object Violation Detector — Single Responsibility Principle](editions/python/chapters/04-oop-principles/visuals/oop_violation_detector.jpg){width=85%}

The following code illustrates this fragile, anemic design:

```python
# Anemic Account Model (Fragile Data Holder)
class Account:
    def __init__(self, id: str, balance: float, currency: str):
        self.id = id
        self.balance = balance
        self.currency = currency

# Stateless Service containing business invariants (Anti-pattern)
class LedgerService:
    def transfer(self, from_acc: Account, to_acc: Account, amount: float) -> None:
        if from_acc.balance < amount:
            raise ValueError("Insufficient funds")
        if from_acc.currency != to_acc.currency:
            raise ValueError("Currency mismatch")
        from_acc.balance -= amount
        to_acc.balance += amount
```


### Why the Anemic Model Fails in Production

1. **Lack of Encapsulation:** Any part of the application can modify the account balance directly: `account.setBalance(new BigDecimal("-1000.00"))`, bypassing the business checks entirely.
2. **Scatter-Shot Validation:** Validation logic is duplicated across multiple services (e.g., `BillingService`, `PayoutService`, `TransferService`). If a validation rule changes, you must locate and modify every instance across the codebase, risking logic drift.
3. **Concurrency Vulnerability:** In high-concurrency systems, separating state from checks leads to **Time-of-Check to Time-of-Use (TOCTOU)** race conditions, resulting in balance corruption.

In a senior coding or architecture interview, presenting an anemic model is a missed opportunity. To demonstrate true software craftsmanship, you must show how to design **rich domain models** that encapsulate state and enforce invariants.

![Anemic vs Rich Domain Model Comparison](editions/python/chapters/04-oop-principles/visuals/anemic_vs_rich.png){width=85%}


## Refactoring Walkthrough: From Anemic to Rich

To refactor a fragile anemic domain into a secure, self-validating rich domain model, follow these three rules:

### Protect Domain Invariants in the Constructor
Ensure that an object can never be created in an invalid state. Validate all inputs during instantiation. If a pre-condition is violated, fail-fast immediately by throwing an exception.

### Remove Setters and Restrict State Access
Eliminate all public setter methods. Fields should be `private` and, where possible, `final`. The only way to modify state is through explicit, domain-specific methods that protect the object's invariants.

### Move Operations Inside the Aggregate Boundary
Instead of letting external service classes manipulate fields, encapsulate the business behavior inside the entity itself. The entity must protect its own state.


## Abstraction & Encapsulation

Encapsulation is not merely the practice of making fields `private` and exposing public getters and setters. True encapsulation means that an object protects its own state, ensuring that its internal data can never enter an invalid state.

In AuraPay, our `LedgerAccount` domain model is rich. It contains its own `debit`, `credit`, and `transferTo` methods, making it impossible to perform a transfer without validating currencies, checking overdraft limits, and preventing concurrency deadlocks.

The following code illustrates this rich encapsulation:

```python
from decimal import Decimal
import threading

class LedgerAccount:
    """
    Demonstrates a rich domain model encapsulating transfer logic and enforcing 
    cross-entity invariants.
    """
    def __init__(self, account_id: str, currency: str, initial_balance: Decimal, overdraft_limit: Decimal):
        self.account_id = account_id
        self.currency = currency
        self._balance = initial_balance
        self.overdraft_limit = overdraft_limit
        self._lock = threading.Lock()

    @property
    def balance(self) -> Decimal:
        with self._lock:
            return self._balance

    def debit(self, amount: Decimal):
        if amount <= 0:
            raise ValueError("Debit amount must be positive")
        with self._lock:
            new_balance = self._balance - amount
            if new_balance + self.overdraft_limit < 0:
                raise ValueError("Overdraft limit exceeded")
            self._balance = new_balance

    def credit(self, amount: Decimal):
        if amount <= 0:
            raise ValueError("Credit amount must be positive")
        with self._lock:
            self._balance += amount

    def transfer_to(self, target: 'LedgerAccount', amount: Decimal):
        """
        Executes a thread-safe transfer to a target account, enforcing business invariants.
        Prevents mismatched currencies and double-debiting.
        """
        if not target or amount is None:
            raise ValueError("Target and amount cannot be null")
        
        # PRE-CONDITION ENFORCEMENT: Currency matching
        if self.currency != target.currency:
            raise ValueError(f"Cannot transfer between mismatched currencies: {self.currency} and {target.currency}")

        # PRE-CONDITION ENFORCEMENT: Self-transfer check
        if self.account_id == target.account_id:
            raise ValueError("Cannot transfer to the same account")

        # To prevent deadlocks, lock accounts in a stable global order
        locks = [self, target]
        locks.sort(key=lambda acc: acc.account_id)

        with locks[0]._lock:
            with locks[1]._lock:
                # Execute atomic debit-credit sequence
                self.debit(amount)
                target.credit(amount)
```


### Deadlock Prevention via Global Ordering
Notice the synchronization logic inside the `transferTo` method. In a high-concurrency payment engine, locking two entities simultaneously (e.g., account $A$ transferring to $B$, while $B$ is transferring to $A$) can lead to a circular wait deadlock. 

To prevent this, the method compares the account identifiers (`this.accountId` and `target.accountId`) and locks them in a consistent, alphabetical global order. This is a classic concurrency pattern that demonstrates your readiness to design banking-grade production code.


## OOP Principles vs. DDD Concepts

Object-Oriented Design and Domain-Driven Design (DDD) are deeply interconnected. When designing enterprise systems, OOD principles map directly to DDD tactical design patterns:

| OOP Principle | DDD Tactical Pattern | Architectural Mapping |
|---|---|---|
| **Encapsulation** | Aggregate Root | The aggregate root acts as a consistency boundary, encapsulating internal entities and protecting invariants from external modification. |
| **Immutability** | Value Object | Objects without distinct identity (like `Money`) are designed as immutable value objects, preventing side effects during sharing. |
| **Polymorphism** | Domain Strategy | Swapping of algorithm strategies (like different fee calculations) is modeled as polymorphic strategy interfaces. |
| **Abstraction** | Repository / Service | Shielding the domain from infrastructure adapters (database, message queues) using clean interface abstractions. |


## Composition over Inheritance

A common mistake in object-oriented design is abusing inheritance. For example, if you are asked to support different settlement networks (ACH, FedWire, Visa), a naive developer might create a base `SettlementService` class and subclass it: `AchSettlementService`, `FedWireSettlementService`, etc.

This creates tight coupling. If you need to change how fees are calculated, or add a new network channel, you risk breaking parent behaviors. The first rule of enterprise OOP design is to **favor composition over inheritance**.

Instead of sub-classing, we compose our routing engine by injecting a collection of independent strategy routes. The core engine is decoupled from the network-specific details.

![Composition over Inheritance](editions/python/chapters/04-oop-principles/visuals/composition_vs_inheritance.png){width=85%}


## Polymorphism over Conditional Logic

One of the easiest ways to spot a junior candidate's code is looking for large `if-else` or `switch` blocks that inspect the type of an object to determine behavior. For example:

```python
# Anti-pattern: Inspecting properties to determine routing
if tx.amount > LIMIT:
    fed_wire_route.process(tx)
else:
    ach_route.process(tx)
```


This violates the Open/Closed Principle. Every time you support a new payment network, you must modify this routing block.

Polymorphism allows you to clean this up. By defining a generic `SettlementRoute` interface, the routing engine can iterate through all available routes, asking each route if it supports the transaction, and executing the process dynamically.

The following code defines this polymorphic settlement design:

```python
from abc import ABC, abstractmethod
from decimal import Decimal
from uuid import UUID

class SettlementRoute(ABC):
    """
    Interface/Abstract Base Class defining the polymorphic contract for payment settlement networks.
    """
    @abstractmethod
    def supports(self, transaction) -> bool:
        pass

    @abstractmethod
    def process(self, transaction):
        pass

    @abstractmethod
    def calculate_fees(self, transaction) -> Decimal:
        pass

class AchRoute(SettlementRoute):
    """
    Concrete implementation for the ACH network (low cost, delayed).
    """
    ACH_FLAT_FEE = Decimal("0.50")

    def supports(self, transaction) -> bool:
        return transaction.amount <= Decimal("100000.00")

    def process(self, transaction):
        print(f"Routing transaction {transaction.transaction_id} via ACH network.")

    def calculate_fees(self, transaction) -> Decimal:
        return self.ACH_FLAT_FEE

class FedWireRoute(SettlementRoute):
    """
    Concrete implementation for the FedWire network (instant, high cost).
    """
    WIRE_FLAT_FEE = Decimal("15.00")

    def supports(self, transaction) -> bool:
        return transaction.amount > Decimal("10000.00")

    def process(self, transaction):
        print(f"Routing transaction {transaction.transaction_id} via FedWire network.")

    def calculate_fees(self, transaction) -> Decimal:
        return self.WIRE_FLAT_FEE
```


By utilizing this interface, the main transaction processor can execute settlements using a clean polymorphic loop, completely decoupled from specific network implementations:

```python
class SettlementProcessor:
    def __init__(self, routes: list[SettlementRoute]):
        self._routes = routes

    def execute(self, transaction: TransactionRecord) -> None:
        active_route = next(
            (route for route in self._routes if route.supports(transaction)), 
            None
        )
        if not active_route:
            raise NoRouteFoundException("No supported route found")
            
        active_route.process(transaction)
```



> ⭐ **STAR Moment: The Encapsulation Test**
> 
> When designing class structures in a technical interview, ask yourself: *Can this class enter an invalid state?* If a client developer can instantiate your object and set its properties to values that violate business rules, your encapsulation has failed. Build your validation boundaries directly into the constructors and state-transition methods of your domain objects.


# SOLID Principles: Enforcing Boundaries

> *"Software architecture is the art of drawing lines between components. SOLID is the rulebook for placing those lines."*


## SOLID in the Senior Interview

In senior and lead engineering interviews, you are almost guaranteed to be asked about the SOLID principles. Too many candidates respond by simply reciting the acronym: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion. 

If you stop there, you fail to show architectural maturity. An interviewer wants to know *why* these principles matter at scale. They want to see how applying these principles prevents structural rot, allows multiple teams to work in parallel without code collisions, and ensures that a change in database technology does not break the core transaction engine.

In this chapter, we will implement the core processing pipeline of AuraPay using a design that strictly conforms to all five SOLID principles.

![The Five SOLID Principles — Quick Reference](editions/python/chapters/05-solid-boundaries/visuals/solid_summary.png){width=70%}

## The SOLID Transaction Pipeline

To illustrate SOLID, we will examine the `TransactionProcessor` in AuraPay. This component is responsible for retrieving ledger accounts, calculating fees, updating account balances, persisting the changes to storage, and notifying external systems.

Here is the decoupled, SOLID-compliant transaction execution flow:

```python
from abc import ABC, abstractmethod
from decimal import Decimal
from uuid import UUID

class LedgerRepository(ABC):
    """
    Abstraction for database operations (Dependency Inversion Principle).
    """
    @abstractmethod
    def find_by_id(self, account_id: UUID):
        pass

    @abstractmethod
    def save(self, account):
        pass

class FeeCalculator(ABC):
    """
    Abstraction for fee calculations (Open/Closed Principle).
    """
    @abstractmethod
    def calculate(self, transaction) -> Decimal:
        pass

class TransactionNotificationSender(ABC):
    """
    Interface Segregation Principle: Focused notification dispatch interface.
    """
    @abstractmethod
    def send_notification(self, transaction, status: str):
        pass

class TransactionProcessor:
    """
    Core transaction processor showing SOLID compliance.
    """
    def __init__(self, repository: LedgerRepository, fee_calculator: FeeCalculator, notification_sender: TransactionNotificationSender):
        self.repository = repository
        self.fee_calculator = fee_calculator
        self.notification_sender = notification_sender

    def process(self, transaction):
        if not transaction:
            raise ValueError("Transaction cannot be null")

        # 1. Retrieve accounts from abstraction (DIP)
        source = self.repository.find_by_id(transaction.source_account_id)
        destination = self.repository.find_by_id(transaction.destination_account_id)

        if not source or not destination:
            raise ValueError("Source or destination account not found")

        # 2. Calculate fee dynamically (OCP)
        fee = self.fee_calculator.calculate(transaction)
        total_debit = transaction.amount + fee

        # 3. Coordinate state transitions on rich domain objects (SRP / LSP)
        source.debit(total_debit)
        destination.credit(transaction.amount)

        # 4. Persist updated states (DIP)
        self.repository.save(source)
        self.repository.save(destination)

        # 5. Notify via segregated interface (ISP)
        self.notification_sender.send_notification(transaction, "SUCCESS")
```


Let us break down how this single class enforces all five design boundaries.


## Single Responsibility Principle (SRP)

The Single Responsibility Principle is often summarized as "a class should do only one thing." A more precise architectural definition is: **"a module should have one, and only one, reason to change."**

In our transaction pipeline, the `TransactionProcessor` has one responsibility: coordinating the business workflow of a transaction. It does not contain database queries, does not know how to format SMS or Email notifications, and does not hardcode fee calculation percentages.

- If the database schema changes, only `LedgerRepository` implementations change.
- If we switch from email notifications to SMS notifications, only `TransactionNotificationSender` implementations change.
- The `TransactionProcessor` remains untouched.


## Open/Closed Principle (OCP)

The Open/Closed Principle states that **software entities should be open for extension, but closed for modification.**

In AuraPay, we must support multiple fee models (e.g., flat fees for retail clients, percentage-based fees for merchants, waived fees for corporate accounts). 
Instead of adding nested `if-else` blocks inside the transaction processor, we inject the `FeeCalculator` interface. If we need to add a new fee model, we simply write a new class implementing `FeeCalculator` and pass it to the processor. The core processor is closed to modifications, yet the fee behavior is infinitely extendable.


## Liskov Substitution Principle (LSP)

The Liskov Substitution Principle states that **subtypes must be substitutable for their base types without altering the correctness of the program.**

In financial systems, this is highly relevant when modeling different account types. For example, a `SavingsAccount` might not allow overdrafts, while a `CheckingAccount` allows up to a certain limit.
If a developer creates a subclass `BlockedAccount` that throws an `UnsupportedOperationException` whenever `debit()` is called, they violate LSP. The `TransactionProcessor` assumes that any `LedgerAccount` returned by the repository can be debited and credited.
LSP ensures that subclass behaviors remain consistent with the contracts defined on their parent classes, preventing runtime crashes.


## Interface Segregation Principle (ISP)

The Interface Segregation Principle states that **clients should not be forced to depend on interfaces they do not use.**

In a large enterprise system, you might have a broad `NotificationService` that handles email, Slack channels, internal logging, and mobile push alerts. 
If the `TransactionProcessor` injected a giant `NotificationService` interface containing twenty unrelated methods, it would be coupled to changes in mobile app push logic. 
Instead, we define a small, segregated interface: `TransactionNotificationSender`, containing only the single `sendNotification` method. The processor only knows about what it needs to execute its task.


## Dependency Inversion Principle (DIP)

The Dependency Inversion Principle states that **high-level modules should not import anything from low-level modules. Both should depend on abstractions.**

This is the most critical principle for decoupling business logic from infrastructure.

- **Low-level modules:** Database APIs, file system writers, network channels, and concrete frameworks.
- **High-level modules:** The core business rules of your application (like transaction routing and double-entry validation).

In our implementation, the `TransactionProcessor` does not import a concrete SQL database connector or Hibernate manager. It depends entirely on the `LedgerRepository` interface. The business logic is at the top of the dependency tree, and database adapters are plugged in at the bottom. This allows you to run unit tests using a mock repository in memory, completely decoupled from a database connection.

![SOLID Dependency Inversion Principle — Before and After](editions/python/chapters/05-solid-boundaries/visuals/solid_dip.png){width=85%}


## SOLID Violation Detector & Remedies

In technical interviews, you must be able to spot structural violations in a code sample and offer clear architectural remedies:

**SRP Violations**

- *Red Flags:* Large source files (500+ lines). Imports both database drivers and UI libraries. Multiple developers editing the same class for unrelated features.
- *Remedy:* **Decomposition** — Split into separate, focused classes coordinated by an Orchestrator or Application Service.

**OCP Violations**

- *Red Flags:* Switch statements or `if-else` blocks inspecting enums/types. Modifying existing service classes to add support for new payment partners.
- *Remedy:* **Abstraction** — Define an interface and implement the Strategy pattern. Inject a collection of these strategies.

**LSP Violations**

- *Red Flags:* Subclass methods returning dummy values or throwing `UnsupportedOperationException`. Typecasting (`instanceof`) inside helper methods.
- *Remedy:* **Hierarchy Flattening** — Replace inheritance with composition, or split the interface into smaller, specialized interfaces.

**ISP Violations**

- *Red Flags:* Concrete classes implementing interfaces with empty or dummy methods. Small client classes coupled to changes in unused interface methods.
- *Remedy:* **Segregation** — Split the fat interface into multiple single-method or small role-based interfaces.

**DIP Violations**

- *Red Flags:* Use of the `new` keyword to instantiate databases/gateways directly inside services. Direct imports of low-level infrastructure modules.
- *Remedy:* **Dependency Injection** — Program to interfaces. Pass dependencies via Constructor Injection, managed by an IoC container.


## Framework Integration: SOLID in Enterprise Containers

Modern web frameworks are designed explicitly around SOLID principles:

### Dependency Injection (IoC) Containers
Frameworks like Spring Boot (Java), ASP.NET Core (C#), and FastAPI/Dependency Injector (Python) serve as Dependency Inversion engines. By registering interfaces and their concrete implementations in the container, the framework automates constructor injection. High-level business modules declare their dependencies as constructor interfaces, completely decoupled from concrete instantiation.

### Aspect-Oriented Programming (AOP)
To adhere to OCP, frameworks use AOP to apply cross-cutting concerns (such as transactions, security, and logging) to service boundaries dynamically using **Proxy decorators**. For instance, adding `@Transactional` in Spring Boot or `[Transaction]` in ASP.NET Core wraps the service class in a proxy container, injecting commit and rollback logic without modifying the service's source code.


### When SOLID Hurts: The Trade-off Analysis

SOLID principles are design heuristics, not commandments. Over-application creates its own category of architectural failures:

**Interface Segregation Overdose:** Splitting every interface into single-method contracts creates an explosion of types. A microservice with 47 single-method interfaces has replaced coupling with cognitive overload. The team spends more time navigating the interface graph than building features.

**Dependency Inversion Overhead:** In small microservices (< 500 lines), injecting every dependency through constructor parameters adds boilerplate without benefit. If a service has exactly one implementation of each dependency and will never be swapped, direct instantiation is simpler and more honest.

**Open-Closed Paralysis:** Designing every class for extension before you have a second use case is speculative generality. YAGNI (You Ain't Gonna Need It) often trumps OCP in early-stage systems. Add extension points when you have evidence of variation, not before.

**Liskov Substitution in Practice:** The classic Rectangle/Square violation is taught in every textbook, but the real-world impact is subtler. When your service contract promises idempotent retries but a subclass implementation has side effects on retry, you've violated LSP in a way that causes production incidents, not just type errors.

> The senior engineer's skill is knowing WHEN to apply SOLID and when the cure is worse than the disease.


> ⭐ **STAR Moment: The Mockability Test**
> 
> The ultimate test of a SOLID design is **mockability**. In a technical interview, explain that a correctly decoupled class can be unit-tested in isolation by mocking all of its interface dependencies. If you cannot test a method without spinning up a real database, an active web server, or a third-party messaging channel, your design violates the Dependency Inversion Principle.


# Modern Functional Programming and Stream APIs

> *"A pipeline of pure functions is a system without side effects. It is a system that can be scaled, tested, and parallelized without fear."*


## The Imperative Loop Trap

A classic interview task is to process a collection of records—filtering out invalid data, transforming the items, and aggregating the result. Historically, developers solved this using imperative structures: `for` loops, nested `if` statements, and mutable local variables.

```python
# Imperative anti-pattern: Hard to read, mutable state, difficult to parallelize
volumes = {}
for tx in transactions:
    if tx.amount >= threshold:
        merchant_id = tx.destination_account_id
        volumes[merchant_id] = volumes.get(merchant_id, 0) + tx.amount
```


While correct, this approach has drawbacks:

- It is highly **imperative**, forcing the reader to track *how* the execution runs rather than *what* is being achieved.
- It relies on **mutable state** (`volumes` map), making it unsafe to parallelize without explicit synchronization locks.
- It lacks clean boundaries, combining filtering, mapping, and aggregation into a single block of code.

Modern software engineering favors the **declarative** approach. Using functional pipelines (Java Streams, C# LINQ, Python Generators), you describe the data transformations as a sequence of side-effect-free operations.

## The AuraPay Batch Pipeline

In AuraPay, we aggregate merchant transaction volumes using functional streams. This allows us to process batches of transactions cleanly.

The following code illustrates this functional pipeline:

```python
from decimal import Decimal
from typing import List, Dict
from uuid import UUID
from collections import defaultdict
from functools import reduce

class TransactionAnalytics:
    """
    Demonstrates high-performance batch transaction analytics in Python.
    """
    def aggregate_merchant_volumes(
        self, 
        transactions: List, 
        min_amount_threshold: Decimal
    ) -> Dict[UUID, Decimal]:
        if transactions is None or min_amount_threshold is None:
            raise ValueError("Transactions and threshold cannot be null")

        # 1. Filter: Retain transactions meeting the value criteria
        filtered_txs = filter(lambda t: t.amount >= min_amount_threshold, transactions)

        # 2. Collect/Reduce: Group by merchant and sum the transaction volume
        merchant_volumes = defaultdict(Decimal)
        for tx in filtered_txs:
            merchant_volumes[tx.destination_account_id] += tx.amount

        return dict(merchant_volumes)

    def get_high_value_transaction_ids(self, transactions: List, limit: Decimal) -> List[UUID]:
        # Declarative list comprehension matching functional map/filter
        return [
            t.transaction_id 
            for t in transactions 
            if t.amount > limit
        ]
```


![Stream Pipeline Visualization](editions/python/chapters/06-functional-streams/visuals/stream_pipeline.png){width=90%}

By declaring the operations as a stream pipeline, the code becomes a readable translation of the business spec:

1.  **Filter** out transaction records below the threshold.
2.  **Collect** the results by grouping by the merchant ID and adding their amounts.



## Debugging Functional Pipelines

Debugging streams can be difficult due to their lazy execution model. To inspect stream internals during test failures, apply these tactics:

![Lazy Evaluation and Short-Circuiting in Streams](editions/python/chapters/06-functional-streams/visuals/lazy_evaluation.jpg){width=85%}

1. **Injecting `peek()` for Logging:**
   Use the `.peek()` intermediate operation to log elements as they flow through specific stages of the pipeline:
```python
def log_and_map(t):
    log.debug(f"Passed Filter: {t.id}")
    return t.merchant_id

merchant_ids = [log_and_map(t) for t in transactions if t.amount > 100]
```


2. **Utilizing IDE Stream Debuggers:**
   Modern IDEs (like IntelliJ IDEA or Visual Studio) contain visual stream debuggers. When you set a breakpoint on a stream statement, the debugger can render a visual representation of how elements are filtered and mapped at each stage.

3. **Splitting the Pipeline for Stack Traces:**
   If a pipeline throws an exception, temporarily break the pipeline into separate intermediate variables to isolate the throwing operation in the stack trace.


> ⭐ **STAR Moment: The Stateless Pipeline Principle**
> 
> A functional stream pipeline must never modify state variables outside the stream. If you write a `.forEach()` or `.map()` that mutates a shared list or updates a local counter, you have violated the functional contract. You lose thread safety, and your code cannot be parallelized. Keep your lambdas pure, stateless, and side-effect-free. In an interview, say: *"I use `collect()` and `reduce()` to accumulate results rather than mutating external variables, because stateless pipelines are safe to parallelize and easy to reason about."*


# Design Patterns in Enterprise Frameworks

> *"Design patterns are not templates to copy; they are vocabulary to describe architectural relationships."*


## Pattern Abuse in Interviews

Many software professionals prepare for design pattern questions by memorizing standard descriptions: "Singleton is a class with one instance," or "Factory creates objects." 

During a senior engineering interview, this is insufficient. A senior candidate must show how patterns solve real architectural problems, such as auditing transaction status, wrapping legacy systems, or handling dynamic business rules. You must also show that you know how these patterns are integrated into the frameworks you use daily (like Spring, Hibernate, or ASP.NET Core).

In this chapter, we will examine how AuraPay utilizes design patterns, focusing on the **Observer Pattern** to audit payment settlement events for financial compliance.


## Creational Patterns

Creational patterns abstract the instantiation process, decoupling your application from how objects are created and composed.

### The Builder Pattern
When constructing complex domain objects like AuraPay's `TransactionRecord`, constructors with ten parameters lead to unreadable code. The **Builder Pattern** solves this, allowing you to build objects step-by-step while maintaining immutability:

```python
# Example of a fluent, type-safe builder for transactions
tx = (TransactionRecordBuilder()
    .with_id(uuid.uuid4())
    .from_account(source_id)
    .to_account(dest_id)
    .with_amount(Decimal("100.00"))
    .in_currency("USD")
    .at_timestamp(datetime.now(timezone.utc))
    .build()) # Immutability and invariants are validated in build()
```


### The Factory Pattern
When the core ledger processor needs to route a payment, it uses a **Factory Pattern** to dynamically instantiate the correct `SettlementRoute` processor based on the transaction metadata (such as routing cards via Visa vs. executing ACH).

### The Singleton Pattern (Creational Deep-Dive)
The Singleton pattern guarantees that a class has only one instance and provides a global point of access to it. In multi-threaded enterprise engines (such as a shared connection pool managed by HikariCP), writing a thread-safe Singleton requires **Double-Checked Locking**:

```python
import threading

class LedgerConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None: # First check (no lock)
            with cls._lock:
                if cls._instance is None: # Second check (with lock)
                    cls._instance = super(LedgerConnectionPool, cls).__new__(cls)
        return cls._instance
```


> **Warning for Senior Candidates:** In cloud-native systems, classical Singletons are often considered an anti-pattern:
> 1. **Testing Complexity:** They introduce global mutable state, making parallel unit tests prone to side effects.
> 2. **Scalability limits:** A Singleton is only single per JVM instance. If your service scales out to ten microservice containers, you have ten connection pool instances, not one.
> 3. **IoC Managed Singletons:** Modern systems delegate singleton lifecycle management to Dependency Injection (IoC) containers rather than hardcoding static `getInstance()` methods.


## Structural Patterns

Structural patterns explain how to assemble objects and classes into larger structures while keeping these structures flexible and efficient.

### The Adapter Pattern
In banking-grade environments, you must frequently integrate with legacy core systems (e.g., COBOL-based mainframes or SOAP APIs). 
The **Adapter Pattern** wraps the legacy API with a clean interface that complies with your domain. For example, a `LegacySoapAdapter` implements the modern `LedgerRepository` interface, converting domain calls into SOAP requests under the hood.

### The Decorator Pattern
If you need to add auditing, metrics, or retry behaviors to transaction execution, do not pollute the core processing code. Use a **Decorator Pattern** to wrap the transaction processor, adding the cross-cutting concerns dynamically:

```python
# Wrapping the core processor with an audit logging decorator
decorated_processor = AuditingTransactionProcessorDecorator(
    CoreTransactionProcessor(repository, calculator, sender)
)
```



## Behavioral Patterns

Behavioral patterns identify common communication patterns between objects and realize these patterns.

### The Strategy Pattern
AuraPay utilizes the **Strategy Pattern** to swap fee calculations dynamically. A `FlatFeeStrategy`, `TieredFeeStrategy`, and `MerchantDiscountRateStrategy` all implement `FeeCalculator`, allowing the routing engine to choose the strategy at runtime based on client profiles.

### The Observer Pattern (Injected)
When a transaction succeeds, external systems—such as the ledger audit index, fraud detection, and SMS notification dispatchers—must be notified. Hardcoding these calls inside the core transaction loop creates tight coupling.

We solve this using the **Observer Pattern**. The `TransactionEventPublisher` manages a list of observers and notifies them of transaction success or failure.

Here is the implementation:

```python
from abc import ABC, abstractmethod

class TransactionObserver(ABC):
    """
    Interface defining the Observer contract for transaction events.
    """
    @abstractmethod
    def on_transaction_success(self, transaction):
        pass

    @abstractmethod
    def on_transaction_failed(self, transaction, error: Exception):
        pass

class AuditTrailObserver(TransactionObserver):
    """
    Concrete Observer that writes a persistent audit trail for security compliance.
    """
    def on_transaction_success(self, transaction):
        print(f"AUDIT SUCCESS: Transaction {transaction.transaction_id} of {transaction.amount} "
              f"{transaction.currency} from {transaction.source_account_id} to {transaction.destination_account_id} "
              f"registered in immutable log.")

    def on_transaction_failed(self, transaction, error: Exception):
        print(f"AUDIT FAILURE: Transaction {transaction.transaction_id} failed. Error: {str(error)}")

class TransactionEventPublisher:
    """
    Subject class managing observers and publishing transaction status updates.
    """
    def __init__(self):
        self._observers = []

    def register_observer(self, observer: TransactionObserver):
        self._observers.append(observer)

    def deregister_observer(self, observer: TransactionObserver):
        self._observers.remove(observer)

    def notify_success(self, transaction):
        for observer in self._observers:
            observer.on_transaction_success(transaction)

    def notify_failure(self, transaction, error: Exception):
        for observer in self._observers:
            observer.on_transaction_failed(transaction, error)
```


![Observer Pattern Class Diagram](editions/python/chapters/07-design-patterns/visuals/observer_pattern.png){width=90%}

### The State Pattern (Behavioral Deep-Dive)
In payment platforms, transactions transition through a strict sequence of states: `CREATED` $\to$ `PENDING` $\to$ `SETTLED` or `FAILED` $\to$ `REFUNDED`.

Instead of writing a massive, hard-to-maintain switch block inside the transaction manager:

- We apply the **State Pattern**.
- We define a `TransactionState` interface representing the allowed operations (e.g., `approve()`, `fail()`, `refund()`).
- Each state is implemented as a concrete class (e.g., `PendingState`, `SettledState`).
- The transition logic is encapsulated inside each state class, preventing invalid state jumps (e.g., you cannot refund a `CREATED` transaction, only a `SETTLED` one), enforcing business invariants at runtime.


## Enterprise Integration & Data Access Patterns

In production-grade enterprise systems, designing clean persistence boundaries is as critical as GoF object coordination:

### Repository and Unit of Work Patterns

- **The Repository Pattern:** Mediates between the domain and data mapping layers using a collection-like interface for accessing domain objects (e.g., `LedgerRepository`). The business layer remains completely ignorant of whether data is stored in Postgres, MongoDB, or an in-memory map.
- **The Unit of Work Pattern:** Tracks all database-modifying operations (inserts, updates, deletes) during a single transaction context. Instead of each repository committing changes independently, the Unit of Work coordinates the commit boundary (e.g., Spring's `@Transactional` boundary or Entity Framework's `DbContext.SaveChanges()`). This guarantees that multiple repository updates succeed or fail together, protecting transactional boundaries.

### Data Transfer Object (DTO) Pattern
Exposing raw database entities directly over public REST/gRPC endpoints is a major security and design vulnerability. Doing so leaks internal database schemas, primary IDs, and sensitive columns (like password hashes).

- **The Solution:** Use **DTOs** (Data Transfer Objects) to define explicit data contracts for request inputs and response outputs. 
- **Mapping:** Utilize mapper libraries to map entities to DTOs before serialization, decoupling internal database schemas from external API consumers.

### Active Record vs. Data Mapper
When designing data access layers, select the persistence mapping style suited for the workload complexity:

- **Active Record (e.g., Ruby on Rails, Django ORM):** An approach where the entity class holds both the data attributes and the database access methods (e.g., `user.save()`, `user.delete()`). Very simple and fast to implement for CRUD applications. However, it violates SRP by coupling the domain model to database connection engines.
- **Data Mapper (e.g., Hibernate, JPA, Entity Framework):** An approach that completely separates data representation (the entity class) from database operations (the mapper/repository layer). The domain object remains database-ignorant, simplifying business unit testing and maintaining clean domain boundaries.


## Framework Integration: Patterns in the Wild

In senior interviews, you must connect patterns to the frameworks you use. Here is how modern enterprise engines implement them natively:

| Pattern | Framework Application | How It Works |
|---|---|---|
| **Factory** | Spring Bean Container | Spring's `BeanFactory` instantiates beans dynamically using reflection and dependency injection maps. |
| **Proxy** | Hibernate Lazy Loading | Hibernate generates proxy wrappers for entity relationships, loading child records from the database only when getter methods are invoked (Lazy Initialization). |
| **Observer** | Spring Application Events | Publishing events via `ApplicationEventPublisher` and consuming them using `@EventListener` decouples services asynchronously. |
| **Adapter** | Spring MVC Handlers | `HandlerAdapter` maps incoming HTTP requests to controller methods, shielding the servlet container from concrete execution signatures. |
| **Template Method** | Spring `JdbcTemplate` | `JdbcTemplate` defines the skeleton of database execution (opening connection, statement preparation, cleanup) while letting subclasses map rows to domain objects. |


> ⭐ **STAR Moment: The Framework Pattern Test**
> 
> During system design interviews, explain design patterns in terms of the framework concepts the interviewer already knows. Instead of drawing a generic observer diagram, say: *"We will implement this like a Spring ApplicationEventPublisher or a Kafka Event Broker, decoupling the transactional write thread from the audit and search indexing consumers."* This shows you understand patterns in modern, production-grade architectures.


# Designing for Performance and Concurrency

> *"High throughput is achieved not by making code run faster, but by eliminating waiting, contention, and coordination."*


## Concurrency in System Design Interviews

When interviewing for a senior staff or engineering manager role, you will inevitably face questions about system bottlenecks. Typical candidates suggest "adding a cache" or "using multi-threading." 

An interviewer wants to hear about the trade-offs of thread management and database contention. You must be able to detail the trade-offs between **Virtual Threads** (Loom) and **Reactive Programming**, explain why a database connection pool that is too large actually degrades system throughput, and articulate exactly when to use **Optimistic** vs. **Pessimistic Concurrency Control** in high-stakes financial operations.

In this chapter, we explore how AuraPay designs its ledger persistence layers to sustain high-volume transaction throughput without risking data drift or transaction races.


## Concurrency Models: Virtual Threads vs. Reactive

In Java 21+, the JVM introduces **Virtual Threads** (Project Loom). In the past, scaling web applications to handle thousands of concurrent connections required reactive frameworks (e.g., Spring WebFlux, Project Reactor). 

### The Thread-per-Request Model
Historically, web servers mapped one platform thread to one HTTP request. Since platform threads map 1-to-1 with operating system threads, they are expensive. Memory footprints (typically 1MB per thread stack) and operating system context-switching overhead capped JVM throughput at a few thousand concurrent threads.

### The Reactive Approach
Reactive programming solved this by decoupling processing execution from threads. Event loops processed chunks of data asynchronously via non-blocking callbacks. 

*   **Advantage:** Extreme scalability with very low resource utilization.
*   **Disadvantage:** Increased code complexity ("callback hell"), difficult stack traces, and complete incompatibility with standard Java threading tools like `ThreadLocal`.

### The Virtual Thread Revolution
Virtual threads are lightweight threads managed by the JVM rather than the OS. They are mounted onto a small carrier pool of platform threads. When a virtual thread blocks on I/O (e.g., executing a SQL query), the JVM unmounts the virtual thread, parking it, and assigns the carrier thread to another task.

![Thread Lifecycle and Context Switching States](editions/python/chapters/08-concurrency-performance/visuals/thread_lifecycle.jpg){width=85%}

*   **Impact:** You can run millions of virtual threads concurrently while writing standard, synchronous, block-on-write code that is easy to read, debug, and trace.

![Virtual Threads vs Platform Threads](editions/python/chapters/08-concurrency-performance/visuals/virtual_threads.png){width=85%}


## Database Locking: Optimistic vs. Pessimistic

When two concurrent transactions attempt to debit the same ledger account, we must prevent double-debiting and race conditions. This requires strict concurrency control.

### Pessimistic Concurrency Control (PCC)
Pessimistic locking assumes that a conflict is highly likely. It blocks concurrent transactions by locking the records at the database level:

```sql
SELECT * FROM accounts WHERE id = ? FOR UPDATE;
```

*   **Pros:** Guaranteed safety; concurrent transactions wait in line until the lock is released.
*   **Cons:** High lock contention, database thread starvation, and high risk of deadlocks under load.
*   **When to use:** When transaction frequency on a single account (e.g., a corporate merchant account) is extremely high, and you cannot afford transaction retries.

### Optimistic Concurrency Control (OCC)
Optimistic locking assumes conflicts are rare. It allows concurrent threads to read and edit records without blocking. When saving the entity, the engine verifies that the record has not been modified by checking a `version` field.

![Optimistic vs Pessimistic Concurrency Control](editions/python/chapters/08-concurrency-performance/visuals/occ_vs_pcc.png){width=70%}

The following code illustrates this version-checking implementation:

```python
from decimal import Decimal
from uuid import UUID

class AccountEntity:
    """
    Represents a database-mapped Ledger Account Entity with versioning for
    Optimistic Concurrency Control (OCC).
    """
    def __init__(self, account_id: UUID, balance: Decimal, currency: str, version: int):
        self.id = account_id
        self.balance = balance
        self.currency = currency
        self.version = version

    def update_balance(self, new_balance: Decimal):
        self.balance = new_balance

    def increment_version(self):
        self.version += 1

class DatabaseLedgerRepository:
    """
    Repository implementation executing the version check update query.
    """
    def save(self, account: AccountEntity):
        # Simulates SQL execution:
        # UPDATE accounts SET balance = ?, version = version + 1 WHERE id = ? AND version = ?;
        sql_query = (
            "UPDATE accounts SET balance = :balance, version = :version + 1 "
            "WHERE id = :id AND version = :version"
        )
        
        rows_updated = self._mock_execute_query(sql_query, account)

        # OCC FAILURE CHECK: No rows updated implies a version conflict
        if rows_updated == 0:
            raise RuntimeError(
                f"Optimistic lock conflict on account {account.id}. "
                f"Outdated version: {account.version}"
            )
            
        account.increment_version()

    def _mock_execute_query(self, query: str, account: AccountEntity) -> int:
        # Simulates the database driver execution
        return 1  # 1 indicates success; 0 indicates a version mismatch conflict
```


*   **Pros:** High throughput; no database locks are held while executing business logic.
*   **Cons:** If a conflict occurs, one of the transactions fails, forcing the application to catch the exception and retry the entire workflow.
*   **When to use:** In low-to-medium contention systems where write conflicts are rare, maximizing parallel performance.


## Concurrency Control & Locking Matrix

When designing financial ledgers, selecting the right locking paradigm is critical. The following matrix contrasts the three primary concurrency control options:

| Criteria | Optimistic Locking (OCC) | Pessimistic Locking (PCC) | Distributed Locking (e.g., Redis Redlock) |
|---|---|---|---|
| **Mechanism** | Application version check (`WHERE version = ?`) | Database row-level locks (`SELECT FOR UPDATE`) | In-memory distributed key lease |
| **Complexity** | Low (handled natively by ORM/SQL) | Medium (requires managing database locks) | High (requires distributed lock manager infrastructure) |
| **Contention Cost** | Low (no blocking, fails fast) | High (blocking threads waiting for lock) | Medium (spins or rejects requests) |
| **Lock Duration** | Nanoseconds (during DB UPDATE commit) | Milliseconds (entire DB transaction block) | Leased duration (typically 5–30 seconds) |
| **Starvation Risk** | High for hot accounts (constant retries) | Low (threads queue in order) | Medium (depends on retry/backoff settings) |
| **Scale Limits** | Scales with DB capacity | Hard limit based on DB connection pool size | Scales horizontally with distributed key store |
| **Deadlock Risk** | Zero | High (requires strict alphabetical locking of aggregates) | Medium (depends on lock lease expiration / release logic) |

![Database Deadlock Cycle — Circular Wait Conditions](editions/python/chapters/08-concurrency-performance/visuals/deadlock_diagram.jpg){width=85%}


## Caching Patterns & Consistency Deep-Dive

In high-throughput platforms, caching is used to offload read traffic from the primary database. However, introducing a cache creates the classic problem of **cache invalidation**.

### Caching Architectures

1. **Cache-Aside (Recommended for Ledgers):**

   - The application queries the cache first.
   - On a *cache hit*, the application returns the cached data.
   - On a *cache miss*, the application queries the database, writes the result to the cache, and returns it.
2. **Write-Through:**

   - The application writes directly to the cache, and the cache synchronizes that write to the database synchronously.
3. **Write-Behind (Write-Back):**

   - The application writes to the cache. The cache buffers these writes and flushes them to the database asynchronously.
   - **WARNING:** Do not use Write-Behind for financial ledgers. A crash of the cache server before the buffer is flushed results in permanent data loss.

### Cache Invalidation & Race Conditions
When updating the database, the application must invalidate the cache key.

- **Naïve Update:** Modifying the database and then updating the cache value. This introduces a race condition: if two concurrent writes occur, they can write to the database and cache in different orders, leading to stale cache states.
- **Correct Pattern:** Always **delete** the cache key after writing to the database. By deleting the key, you force the next read operation to perform a Cache-Aside query from the source database, guaranteeing consistency.
- **Transactional Safety:** Ensure the cache key deletion occurs inside the database transaction's post-commit hook. If the database transaction rolls back, the cache key must not be deleted.


## Memory Architecture: PyMalloc, Reference Counting, Generational Cyclic GC, and GIL

In high-performance Python 3.11+ applications (such as FastAPI microservices and telemetry aggregation pipelines), understanding CPython's internal memory manager is critical for preventing memory leaks, reducing GC overhead, and designing low-latency systems.

### The CPython Layered Memory Architecture

Unlike languages that rely solely on a tracing garbage collector, CPython employs a multi-tiered memory architecture to handle object allocation efficiently.

#### 1. Small Object Allocator (`PyMalloc`)
- **Scope:** Handles all Python object allocations **$\le$ 512 bytes** (e.g., integers, floats, small strings, tuples, dictionaries).
- **Structure:** `PyMalloc` avoids expensive operating system `malloc()` calls by organizing memory into a 3-tier hierarchy:
  - **Arenas (256 KB):** Memory blocks requested directly from the OS page allocator.
  - **Pools (4 KB):** Each Arena is divided into 64 Pools of 4 KB each. Each Pool handles objects of a single fixed size-class (e.g., 16-byte pool, 32-byte pool).
  - **Blocks (8 to 512 bytes):** Subdivisions inside a Pool where actual Python objects reside.
- **Benefit:** Fast $O(1)$ allocation and zero external fragmentation for small objects.

#### 2. System Allocator (`malloc` / `free`)
- **Scope:** Objects **larger than 512 bytes** (e.g., large lists, NumPy arrays, byte buffers) bypass `PyMalloc` and are allocated directly via system `malloc()`.

---

### Dual Garbage Collection Mechanisms

CPython uses a **dual-engine garbage collection architecture**:

#### 1. Primary Engine: Reference Counting ($O(1)$ Instant Reclamation)
Every CPython object structure contains a `ob_refcnt` header field (defined in `PyObject`).

- **Increment:** `ob_refcnt` increases when an object is assigned to a variable, passed to a function, or added to a list/dictionary.
- **Decrement:** `ob_refcnt` decreases when a variable goes out of scope, is reassigned, or is explicitly deleted via `del obj`.
- **Instant Deallocation:** As soon as `ob_refcnt == 0`, the memory is **deallocated instantly** on the current execution thread. No STW pause required!

```python
import sys

x = [1, 2, 3]
print(sys.getrefcount(x))  # Output: 2 (variable 'x' + temporary reference in getrefcount)
y = x
print(sys.getrefcount(x))  # Output: 3
del y
print(sys.getrefcount(x))  # Output: 2
```

#### 2. Secondary Engine: Generational Cyclic Garbage Collector
Reference counting has one fatal flaw: **it cannot detect reference cycles** (e.g., Object A points to Object B, and Object B points to Object A; both variables are deleted, but `ob_refcnt` remains `1` for both).

CPython includes a **Generational Cyclic GC** to detect and break isolated reference cycles.

---

### The CPython Cyclic GC Generations & Cycle Detection

The Cyclic GC only tracks **container objects** (objects capable of holding references to other objects, such as `dict`, `list`, `tuple`, `set`, and custom class instances).

#### 1. The 3 GC Generations
- **Generation 0 (Gen 0):** Every newly created container object is assigned to Gen 0. Checked frequently when allocations exceed `-XX` threshold (`gc.get_threshold()`).
- **Generation 1 (Gen 1):** Containers that survive a Gen 0 collection are promoted to Gen 1.
- **Generation 2 (Gen 2):** Long-lived containers surviving Gen 1 are promoted to Gen 2. Gen 2 collections occur infrequently.

#### 2. Cycle Detection Algorithm
To find cycles, the CPython GC:
1. Creates a candidate list of container objects.
2. Trial-decrements reference counts (`gc_refs`) for all references between tracked containers.
3. Any container whose effective `gc_refs` drops to `0` is part of an isolated reference cycle and is scheduled for destruction.

---

### The Global Interpreter Lock (GIL) & Memory Safety

- **Thread Safety of `ob_refcnt`:** Because reference counts are mutated continuously on every assignment, multi-threaded access without synchronization would cause data races on `ob_refcnt`.
- **The Role of the GIL:** The Global Interpreter Lock ensures that only one native OS thread executes CPython bytecode at a time, protecting `ob_refcnt` mutations from race conditions.
- **Free-Threading in Python 3.13+ (PEP 703):** Modern Python versions introduce experimental build flags (`--disable-gil`) using atomic reference counting (`Py_atomic_int`) to enable true multi-core parallel execution.

---

### Python Memory Optimization Best Practices

- **`__slots__` for Memory Efficiency:** By default, every class instance uses a `__dict__` dictionary to store instance attributes, incurring high `PyMalloc` overhead. Defining `__slots__` eliminates `__dict__`, storing attributes in a fixed flat array and reducing per-instance memory consumption by up to 60%.

```python
class FastTransaction:
    __slots__ = ('id', 'amount', 'timestamp') # Zero __dict__ memory overhead!

    def __init__(self, tx_id, amount, timestamp):
        self.id = tx_id
        self.amount = amount
        self.timestamp = timestamp
```

- **`weakref` Module:** Use `weakref.ref` or `weakref.WeakKeyDictionary` to reference objects without incrementing `ob_refcnt`, preventing reference cycles in caching and observer patterns.



## CPU Cache Locality (L1/L2/L3) in HFT Matching Loops

In ultra-low-latency matching engines (like ZenithTrade), garbage collection pauses and CPU cache misses are the primary bottlenecks. To write code that runs in the microsecond range, you must design for **cache locality**:

- **The Problem with Linked Lists:** A standard `LinkedList` contains nodes linked by memory references. These references can be scattered randomly across heap memory. When the CPU traverses a linked list to match orders, it incurs constant **L1/L2/L3 cache misses**, forcing the CPU to fetch data from physical RAM, which is up to 200 times slower than L1 cache.
- **The Array/Contiguous Layout:** To minimize cache misses, the matching loop must use contiguous memory structures. By storing orders in flat array layouts or utilizing pre-allocated object pools, the CPU can load adjacent elements into L1/L2 cache pre-emptively, accelerating execution speed.


## Connection Pool Sizing: The HikariCP Formula

A common design flaw is over-allocating database connection pool sizes. If you have 500 thread workers, candidates often set the connection pool size to 500.

**The Pitfall:** A database engine is limited by physical resources (CPU cores, disk write I/O speed, memory). When hundreds of threads attempt to execute database operations concurrently, the database server spends more time performing CPU context switches than executing queries.

HikariCP (the industry-standard connection pool manager) uses a formula derived from PostgreSQL benchmark testing to size database pools:

```
Pool Size = (Core Count * 2) + Effective Spindle Count
```

For example, a database server with 8 CPU cores and an SSD array (spindle count of 1) should have a pool size of:

```
(8 * 2) + 1 = 17 Connections
```

Setting the pool size to 17 will yield *higher* overall throughput than setting it to 100, due to the minimization of CPU context switching and disk spindle thrashing.

**Important Context:** This formula was derived empirically by the PostgreSQL community for spinning disk (HDD) workloads where 'Effective Spindle Count' represents physical disk heads. For modern NVMe SSDs and cloud-managed databases (e.g., Aurora, Cloud SQL), this formula is a starting point, not a universal law. Cloud databases often recommend pool sizes of 2-5× CPU cores. Always benchmark with your specific database engine and storage backend.

![HikariCP Connection Pool Sizing](editions/python/chapters/08-concurrency-performance/visuals/hikaricp_formula.png){width=85%}




> ⭐ **STAR Moment: The Cache Invalidation Design**
> 
> When discussing performance during an interview, never say *"We will add a cache."* Say: *"We will implement a Cache-Aside pattern using Redis. To prevent stale reads in our double-entry ledger, we will use a transactional write-through strategy, invalidating cache keys atomically inside the database commit boundary to ensure absolute consistency."* This shows you understand caching boundaries in financial transaction systems.


\part{Algorithmic Mastery}


# Core Algorithms & Assessment Tactical Guide

> *"Algorithms are not trivia; they are the baseline vocabulary of computational efficiency under resource constraints."*

---

## The Veteran's Perspective: Patterns vs. Memorization

For senior engineers, architects, and engineering managers returning to technical assessments after years in leadership, coding assessments present a unique hurdle. You have architected distributed ledgers, managed multi-million-dollar technology budgets, and led high-performing engineering teams. Yet, when faced with a 70-minute timer and a blank editor window, a frustrating mental block occurs: your mind goes blank.

This happens because **algorithmic problem-solving is like mathematics**. You cannot master calculus by passively reading a textbook or watching someone solve equations on a whiteboard. Reading a solution creates a deceptive illusion of competence—you nod along, thinking, *"Yes, that makes sense."* But when you pick up the pencil (or open the IDE) to solve a problem from scratch, you realize you have not internalized the mechanics.

Furthermore, attempting to memorize hundreds of individual algorithm problems is a dangerous trap. Under time pressure, memorized code snippets dissolve. 

The only sustainable path back to coding mastery is **pattern-based problem solving**:
1. **Learn the 24 Canonical Programming Patterns**—the core mathematical invariants and code skeletons that govern all algorithmic problems.
2. **Analyze the problem structure** to map requirements directly to a pattern ID (`[PAT-01]` through `[PAT-24]`).
3. **Practice by doing.** Implement 2–3 problems for each pattern independently until the code skeleton becomes pure muscle memory.

When you master the 24 patterns below, you no longer need to memorize hundreds of solutions. You simply recognize the pattern, apply the appropriate code skeleton, and derive the solution cleanly on demand.

---

## General Coding Assessment (general coding assessment) Tactics

Standardized online coding assessments (e.g., General Coding Assessments, HackerRank, or Codility) evaluate speed, accuracy, and edge-case handling under severe time constraints. The most common format is the **70-Minute, 4-Question Speed Run**.

### The 4-Question Blueprint

| Question | Difficulty | Target Time | Primary Pattern Types | Tactical Rule |
|:-----------------|:------------|:------------|:----------------------|:-------------------------------------------------------------------|
| **Easy-tier** | Easy | 5–8 Min | `[PAT-01]`, `[PAT-02]` | Write clean, brute-force code immediately. Do not over-optimize. |
| **Medium-tier** | Medium | 10–12 Min | `[PAT-03]`, `[PAT-06]`, `[PAT-10]` | Watch for array bounds and off-by-one errors. |
| **Medium-Hard-tier** | Medium-Hard | 15–20 Min | `[PAT-04]`, `[PAT-13]`, `[PAT-14]` | Identify the window state or queue batching early. |
| **Hard-tier** | Hard | 20–25 Min | `[PAT-05]`, `[PAT-09]`, `[PAT-11]`, `[PAT-19]` | If brute force is $O(N^2)$, look for a monotonic property or DP state. |

### The 70-Minute general coding assessment Master Plan

1. **The 3-Minute Limit:** If you get stuck on a compile or logic bug for more than 3 minutes, comment out your changes, revert to your last working baseline, and rethink your boundary conditions.
2. **Never print in a loop:** Printing to standard output inside loops kills execution speed and causes hidden test timeouts.
3. **Submit immediately:** Once your solution passes visible test cases, submit it and move on.
4. **Strategic Order (1 -> 2 -> 4 -> 3):** On platforms like automated testing platforms, Hard-tier is often worth significantly more points than Medium-Hard-tier and is usually more deterministic (e.g., Monotonic Stack or Binary Search) than Medium-Hard-tier, which can involve tedious simulation.

---

### Complexity Foundations: A Quick Reference

Before diving into the 25 canonical patterns, ensure you have instant recall of these complexity classes:

| Complexity | Name | Example | Max N for 1s |
|-----------|------|---------|-------------|
| O(1) | Constant | HashMap lookup | ∞ |
| O(log N) | Logarithmic | Binary search | 10^18 |
| O(N) | Linear | Single pass scan | 10^8 |
| O(N log N) | Linearithmic | Merge sort | 10^6 |
| O(N²) | Quadratic | Nested loops | 10^4 |
| O(2^N) | Exponential | Subset generation | 20-25 |
| O(N!) | Factorial | Permutations | 10-12 |

![Big-O Time Complexity Comparison Graph](editions/python/chapters/09-algorithms-assessment/visuals/big_o_comparison.jpg){width=85%}

**The Constraint-to-Complexity Rule:** Read the problem constraints FIRST. If N ≤ 10^4, O(N²) is acceptable. If N ≤ 10^5, you need O(N log N) or better. If N ≤ 10^6, you need O(N). This single rule eliminates 50% of wrong algorithm choices before you write a line of code.

![Constraint-to-Complexity Flowchart](editions/python/chapters/09-algorithms-assessment/visuals/constraint_flowchart.jpg){width=85%}

---

# The 24 Canonical Programming Patterns

The following catalog defines the 24 fundamental patterns of computational problem-solving. Each pattern represents a proven, invariant structure for solving a specific class of problems.

---

## Module 1: Array & String Mechanics

### [PAT-01] Direct Indexing & Frequency Buckets

- **Invariant:** When the input domain is finite (e.g., ASCII characters, digits $0..9$), a fixed-size array (`int[256]`) provides $O(1)$ direct-indexing lookup without hash overhead.
- **Mental Model:** Use the array index itself as the key.
- **Canonical Code Skeleton:**
```python
def first_unique_char(s: str) -> int:
    counts = [0] * 256
    for c in s:
        counts[ord(c)] += 1
    for i, c in enumerate(s):
        if counts[ord(c)] == 1:
            return i
    return -1
```

- **Diagnostic Triggers:** "First non-repeating character", "Anagram check", "Character frequency".
- **Boundary Conditions:** Ensure array size covers the domain (`256` for ASCII, `26` for lowercase English).
- **Real-World Application:** High-speed network packet inspection, audit log frequency counting.

---

### [PAT-02] In-Place Mutation & Two-Pointer Compaction

- **Invariant:** A `write` pointer tracks the boundary of valid elements while a `read` pointer scans the array, mutating data in-place in $O(1)$ extra space.
- **Mental Model:** Filter or compact elements in a single pass without allocating a new array.
- **Canonical Code Skeleton:**
```python
def remove_duplicates(nums: list[int]) -> int:
    if not nums:
        return 0
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:
            nums[write] = nums[read]
            write += 1
    return write
```

- **Diagnostic Triggers:** "In-place removal", "Compact array", "Move zeroes to end".
- **Boundary Conditions:** Handle empty array or single-element array upfront.
- **Real-World Application:** Memory defragmentation, log stream sanitization.

---

### [PAT-03] Prefix Sums & Range Query Invariants

- **Invariant:** The sum of elements between indices $i$ and $j$ equals `prefix[j + 1] - prefix[i]`, turning range sum queries into $O(1)$ operations.
- **Mental Model:** Precompute cumulative totals so any subarray sum is computed by subtraction.
- **Canonical Code Skeleton:**
```python
from collections import defaultdict

def subarray_sum(nums: list[int], k: int) -> int:
    pref_counts = defaultdict(int)
    pref_counts[0] = 1
    current_sum = 0
    count = 0
    
    for num in nums:
        current_sum += num
        if current_sum - k in pref_counts:
            count += pref_counts[current_sum - k]
        pref_counts[current_sum] += 1
        
    return count
```

- **Diagnostic Triggers:** "Subarray sum equals K", "Range sum queries", "Equal number of 0s and 1s".
- **Boundary Conditions:** Always initialize `prefCounts.put(0, 1)` to account for subarrays starting at index 0.
- **Real-World Application:** Financial ledger balance auditing, telemetry interval aggregation.

---

## Module 2: Windowing & Pointer Navigation

### [PAT-04] Dynamic Sliding Window (Variable Size)

- **Invariant:** Maintain a window `[left...right]`. Expand `right` to include elements. When constraint is violated, shrink from `left` until valid.
- **Mental Model:** An expanding and contracting net scanning an array.
- **Canonical Code Skeleton:**
```python
def longest_subarray(nums: list[int], k: int) -> int:
    left = 0
    result = 0
    zero_count = 0
    
    for right in range(len(nums)):
        if nums[right] == 0:
            zero_count += 1
            
        while zero_count > k:
            if nums[left] == 0:
                zero_count -= 1
            left += 1
            
        result = max(result, right - left + 1)
        
    return result
```

- **Diagnostic Triggers:** "Longest/shortest subarray satisfying condition X", "At most K distinct elements".
- **Boundary Conditions:** Set-based windows must shrink BEFORE expanding; HashMap/Sum-based windows expand FIRST then shrink.
- **Real-World Application:** Sliding-window rate limiters, network throughput monitoring.

---

### [PAT-05] Fixed-Size Monotonic Deque Window

- **Invariant:** Maintain a `Deque` of indices where corresponding values are strictly decreasing from front to back. Front always holds the maximum of the current window.
- **Mental Model:** A sliding window of fixed size $K$ that tracks max/min in $O(1)$ amortized time.
- **Canonical Code Skeleton:**
```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    dq = deque()
    res = []
    
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft() # Expire
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop() # Kill weaker
        dq.append(i)
        if i >= k - 1:
            res.append(nums[dq[0]])
            
    return res
```

- **Diagnostic Triggers:** "Maximum/minimum in every window of size K".
- **Boundary Conditions:** Deque stores INDICES, not values. Window is full when `i >= k - 1`.
- **Real-World Application:** Real-time SLA monitoring, financial tick-data peak detection.

---

### [PAT-06] Converging Two-Pointers

- **Invariant:** Two pointers start at opposite ends (`left = 0`, `right = n - 1`) of a sorted array and move inward based on comparison with target.
- **Mental Model:** Squeezing the search space from both boundaries.
- **Canonical Code Skeleton:**
```python
def two_sum_sorted(nums: list[int], target: int) -> list[int]:
    left, right = 0, len(nums) - 1
    while left < right:
        curr_sum = nums[left] + nums[right]
        if curr_sum == target:
            return [left, right]
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return []
```

- **Diagnostic Triggers:** "Sorted array + find pair", "Container with most water", "Palindrome validation".
- **Boundary Conditions:** Array MUST be sorted. Loop condition is `left < right` (pointers must not overlap for pairs).
- **Real-World Application:** Order matching engines, debit-credit balance pairing.

---

### [PAT-07] Fast & Slow Pointers (Floyd's Cycle Detection)

- **Invariant:** `slow` moves 1 step while `fast` moves 2 steps. If a cycle exists, `fast` will eventually catch `slow`.
- **Mental Model:** Two runners on a circular track.
- **Canonical Code Skeleton:**
```python
def has_cycle(head: 'ListNode') -> bool:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

- **Diagnostic Triggers:** "Detect cycle in linked list", "Find duplicate number", "Happy number".
- **Boundary Conditions:** Check `fast != null && fast.next != null` to avoid `NullPointerException`.
- **Real-World Application:** Circular reference detection in graph engines, deadlock detection.

---

## Module 3: Stacks, Queues & Monotonic Structures

### [PAT-08] LIFO Matching & Expression Parsing

- **Invariant:** Push open symbols onto a stack. When a closing symbol is encountered, pop and verify it matches the expected opening symbol.
- **Mental Model:** Last-in, first-out validation of nested structures.
- **Canonical Code Skeleton:**
```python
def is_valid_parentheses(s: str) -> bool:
    stack = []
    for c in s:
        if c == '(':
            stack.append(')')
        elif c == '{':
            stack.append('}')
        elif c == '[':
            stack.append(']')
        elif not stack or stack.pop() != c:
            return False
    return len(stack) == 0
```

- **Diagnostic Triggers:** "Valid parentheses", "Evaluate expression", "Simplify file path".
- **Boundary Conditions:** Stack must be empty at the end. Check `stack.isEmpty()` before popping.
- **Real-World Application:** JSON/XML syntax parsers, compiler AST validation, undo stacks.

---

### [PAT-09] Monotonic Stack ("The Waiting Room")

- **Invariant:** Stack holds unresolved element indices in decreasing order. When a larger element arrives, it pops colder elements and resolves their answers.
- **Mental Model:** A waiting room where people stay until someone taller arrives to liberate them.
- **Canonical Code Skeleton:**
```python
def daily_temperatures(temps: list[int]) -> list[int]:
    ans = [0] * len(temps)
    stack = [] # Stores INDICES
    
    for i in range(len(temps)):
        while stack and temps[stack[-1]] < temps[i]:
            prev_idx = stack.pop()
            ans[prev_idx] = i - prev_idx
        stack.append(i)
        
    return ans
```

- **Diagnostic Triggers:** "Next greater element", "Daily temperatures", "Largest rectangle in histogram".
- **Boundary Conditions:** Store INDICES on stack, not values. Unresolved items remain `0` or `-1`.
- **Real-World Application:** Stock price drop alerts, automated threshold breach notifications.

---

## Module 4: Search Space & Decision Trees

### [PAT-10] Monotonic Partition Binary Search

- **Invariant:** In a rotated sorted array, at least one half (left or right) is always strictly sorted.
- **Mental Model:** Halving search space by identifying the sorted partition.
- **Canonical Code Skeleton:**
```python
def search_rotated(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
            
        if nums[left] <= nums[mid]: # Left half sorted (MUST use <=)
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else: # Right half sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

- **Diagnostic Triggers:** "Search in rotated sorted array", "Find minimum in rotated sorted array".
- **Boundary Conditions:** Use `nums[left] <= nums[mid]` (with `<=`) to handle single-element partitions.
- **Real-World Application:** Distributed partition log search, sharded database key lookups.

---

### [PAT-11] Binary Search on Solution Range

- **Invariant:** When the answer lies within a known numeric range `[min...max]` and a predicate function `feasible(x)` is monotonic, binary search finds the optimal value.
- **Mental Model:** Guess the answer, test if it works, halve the range.
- **Canonical Code Skeleton:**
```python
def ship_within_days(weights: list[int], days: int) -> int:
    lo, hi = max(weights), sum(weights)
    
    def can_ship(capacity: int) -> bool:
        day_count = 1
        current_load = 0
        for w in weights:
            if current_load + w > capacity:
                day_count += 1
                current_load = 0
            current_load += w
        return day_count <= days

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if can_ship(mid):
            hi = mid # Try smaller capacity
        else:
            lo = mid + 1 # Must increase capacity
            
    return lo
```

- **Diagnostic Triggers:** "Find minimum capacity", "Koko eating bananas", "Split array largest sum".
- **Boundary Conditions:** Define correct range bounds `[lo, hi]` upfront.
- **Real-World Application:** Capacity planning, thread pool sizing, rate limit optimization.

---

### [PAT-12] Backtracking & State-Space Pruning

- **Invariant:** Explore decision paths recursively; when a path violates constraints, backtrack (undo state change) and try the next branch.
- **Mental Model:** Exploring a maze by dropping breadcrumbs and stepping back when hitting a dead end.
- **Canonical Code Skeleton:**
```python
def backtrack(res: list[list[int]], path: list[int], nums: list[int], used: list[bool]) -> None:
    if len(path) == len(nums):
        res.append(list(path))
        return
        
    for i in range(len(nums)):
        if used[i]:
            continue
        used[i] = True
        path.append(nums[i])
        backtrack(res, path, nums, used) # Recurse
        path.pop() # Undo (backtrack)
        used[i] = False
```

- **Diagnostic Triggers:** "Generate all permutations/combinations", "Sudoku solver", "N-Queens".
- **Boundary Conditions:** Always make a deep copy `new ArrayList<>(path)` when adding to results.
- **Real-World Application:** Constraint satisfaction solvers, security permission path traversal.

---

## Module 5: Graph & Grid Traversals

### [PAT-13] Level-by-Level BFS Wavefront

- **Invariant:** Queue processes nodes layer-by-layer (`int size = queue.size()`). First time target is popped = shortest path in unweighted graph/grid.
- **Mental Model:** Water ripples expanding outward in concentric circles.
- **Canonical Code Skeleton:**
```python
from collections import deque

def shortest_path(grid: list[list[str]], start_r: int, start_c: int) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start_r, start_c)])
    visited = [[False] * cols for _ in range(rows)]
    visited[start_r][start_c] = True # Mark visited ON PUSH
    
    steps = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while queue:
        size = len(queue)
        for _ in range(size):
            curr_r, curr_c = queue.popleft()
            if grid[curr_r][curr_c] == 'E':
                return steps
                
            for dr, dc in dirs:
                nr, nc = curr_r + dr, curr_c + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    not visited[nr][nc] and grid[nr][nc] != 'X'):
                    visited[nr][nc] = True # MARK ON PUSH!
                    queue.append((nr, nc))
        steps += 1
        
    return -1
```

- **Diagnostic Triggers:** "Shortest path in grid", "Minimum steps to reach goal", "Word ladder".
- **Boundary Conditions:** ALWAYS mark `visited = true` on `offer()`, NOT on `poll()`.
- **Real-World Application:** Network routing protocols, social network distance calculation.

---

### [PAT-14] Multi-Source BFS Parallel Spreading

- **Invariant:** Push ALL starting origin points into the Queue at time $t=0$. The wavefront expands from all origins simultaneously.
- **Mental Model:** Multiple fires starting at different spots and spreading at equal speed.
- **Canonical Code Skeleton:**
```python
from collections import deque

def oranges_rotting(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c)) # Push ALL sources
            elif grid[r][c] == 1:
                fresh_count += 1
                
    if fresh_count == 0:
        return 0
        
    minutes = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while queue and fresh_count > 0:
        size = len(queue)
        minutes += 1
        for _ in range(size):
            curr_r, curr_c = queue.popleft()
            for dr, dc in dirs:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2 # Mutate grid as visited
                    fresh_count -= 1
                    queue.append((nr, nc))
                    
    return minutes if fresh_count == 0 else -1
```

- **Diagnostic Triggers:** "Rotting oranges", "Walls and gates", "Multi-point fire propagation".
- **Boundary Conditions:** Track remaining fresh target count to avoid extra minute increment.
- **Real-World Application:** Multi-datacenter cache invalidation, rumor/virus propagation modeling.

---

### [PAT-15] DFS Component Sinking & Flood Fill

- **Invariant:** Traverse connected component recursively; mutate cell value (`'1' -> '0'`) to mark visited and eliminate memory overhead.
- **Mental Model:** Sinking an island as you walk over it so you never visit it again.
- **Canonical Code Skeleton:**
```python
def num_islands(grid: list[list[str]]) -> int:
    def dfs_sink(r: int, c: int) -> None:
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] == '0':
            return
        grid[r][c] = '0' # Sink cell
        dfs_sink(r + 1, c)
        dfs_sink(r - 1, c)
        dfs_sink(r, c + 1)
        dfs_sink(r, c - 1)

    count = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '1':
                count += 1
                dfs_sink(r, c)
                
    return count
```

- **Diagnostic Triggers:** "Number of islands", "Surrounded regions", "Flood fill".
- **Boundary Conditions:** Base case must check bounds BEFORE accessing `grid[r][c]`.
- **Real-World Application:** Image segmentation, cluster isolation, GIS landmass detection.

---

### [PAT-16] Topological Sort (Kahn's & DFS)

- **Invariant:** Process nodes with in-degree 0 first. Reduces in-degree of neighbors. If processed count $< N$, a cycle exists.
- **Mental Model:** Resolving build dependencies in order.
- **Canonical Code Skeleton:**
```python
from collections import deque

def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    in_degree = [0] * num_courses
    adj = [[] for _ in range(num_courses)]
    
    for dest, src in prerequisites:
        adj[src].append(dest)
        in_degree[dest] += 1
        
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    
    order = []
    while queue:
        curr = queue.popleft()
        order.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    return order if len(order) == num_courses else []
```

- **Diagnostic Triggers:** "Course schedule", "Task dependency ordering", "Build order".
- **Boundary Conditions:** Return empty array if `idx != numCourses` (cycle detected).
- **Real-World Application:** Maven/Gradle build execution, CI/CD pipeline stage ordering.

---

### [PAT-17] Disjoint Set Union (Union-Find)

- **Invariant:** Maintain connected sets using parent pointers with path compression and rank optimization for near $O(1)$ amortized `find` and `union`.
- **Mental Model:** Merging social groups and checking if two people share the same root leader.
- **Canonical Code Skeleton:**
```python
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        
    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i]) # Path compression
        return self.parent[i]
        
    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
            
        return False # Already connected!
```

- **Diagnostic Triggers:** "Redundant connection", "Number of connected components", "Accounts merge".
- **Boundary Conditions:** Path compression `parent[i] = find(parent[i])` is essential for optimal speed.
- **Real-World Application:** Network topology clustering, distributed consensus membership tracking.

---

### [PAT-18] Weighted Shortest Path (Dijkstra / Min-Heap)

- **Invariant:** Use a `PriorityQueue` ordered by distance. Always expand the unvisited node with the smallest tentative distance.
- **Mental Model:** Exploring shortest path on a map with varying road costs.
- **Canonical Code Skeleton:**
```python
import heapq
from collections import defaultdict

def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    adj = defaultdict(list)
    for u, v, w in times:
        adj[u].append((v, w))
        
    pq = [(0, k)] # [dist, node]
    dist_map = {}
    
    while pq:
        d, node = heapq.heappop(pq)
        
        if node in dist_map:
            continue
        dist_map[node] = d
        
        for neighbor, weight in adj[node]:
            if neighbor not in dist_map:
                heapq.heappush(pq, (d + weight, neighbor))
                
    return max(dist_map.values()) if len(dist_map) == n else -1
```

- **Diagnostic Triggers:** "Network delay time", "Cheapest flight within K stops", "Shortest path with weights".
- **Boundary Conditions:** PriorityQueue stores `[node, total_distance]`. Skip already finalized nodes (`dist.containsKey(node)`).
- **Real-World Application:** Latency-based API gateway routing, Google Maps route optimization.

---

## Module 6: Dynamic Programming & Optimization

### [PAT-19] 1D Choice Optimization (O(1) Space DP)

- **Invariant:** State `dp[i]` depends only on `dp[i - 1]` and `dp[i - 2]`. Space can be optimized from $O(N)$ array to 2 variables (`prev1`, `prev2`).
- **Mental Model:** Making optimal choice between taking current item or skipping it.
- **Canonical Code Skeleton:**
```python
def rob(nums: list[int]) -> int:
    if not nums:
        return 0
    prev2, prev1 = 0, 0
    
    for num in nums:
        curr = max(prev1, prev2 + num) # Skip vs Take
        prev2 = prev1
        prev1 = curr
        
    return prev1
```

- **Diagnostic Triggers:** "House robber", "Climbing stairs", "Min cost climbing stairs".
- **Boundary Conditions:** Handle single-element input upfront.
- **Real-World Application:** Capacity allocation, CPU time-slot scheduling.

---

### [PAT-20] 0/1 & Unbounded Knapsack DP

- **Invariant:** `dp[w]` represents max value for capacity `w`. Iterate items and update capacity backwards for 0/1 (use item once) or forwards for unbounded (use item infinitely).
- **Mental Model:** Packing a backpack with items to maximize value without exceeding weight capacity.
- **Canonical Code Skeleton (Coin Change - Unbounded):**
```python
def coin_change(coins: list[int], amount: int) -> int:
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:
                dp[i] = min(dp[i], dp[i - coin] + 1)
                
    return -1 if dp[amount] > amount else dp[amount]
```

- **Diagnostic Triggers:** "Coin change", "Partition equal subset sum", "Knapsack capacity".
- **Boundary Conditions:** Fill array with sentinel value (`amount + 1`) representing infinity.
- **Real-World Application:** Resource packing in cloud instances, currency change calculators.

---

### [PAT-21] 2D Grid Path Optimization

- **Invariant:** `dp[r][c]` represents min/max value to reach cell `(r, c)`, which depends on `dp[r - 1][c]` (from top) and `dp[r][c - 1]` (from left).
- **Mental Model:** Walking down and right on a grid accumulating values.
- **Canonical Code Skeleton:**
```python
def min_path_sum(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                dp[r][c] = grid[r][c]
            elif r == 0:
                dp[r][c] = dp[r][c - 1] + grid[r][c]
            elif c == 0:
                dp[r][c] = dp[r - 1][c] + grid[r][c]
            else:
                dp[r][c] = min(dp[r - 1][c], dp[r][c - 1]) + grid[r][c]
                
    return dp[rows - 1][cols - 1]
```

- **Diagnostic Triggers:** "Minimum path sum", "Unique paths in grid", "Dungeon game".
- **Boundary Conditions:** Initialize first row and first column carefully.
- **Real-World Application:** Cost-effective data routing across grid-structured networks.

---

### [PAT-22] String Alignment & Sequence DP

- **Invariant:** `dp[i][j]` represents optimal alignment score for prefix `s1[0..i-1]` and `s2[0..j-1]`.
- **Mental Model:** 2D grid matching characters of two strings.
- **Canonical Code Skeleton (Longest Common Subsequence):**
```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    return dp[m][n]
```

- **Diagnostic Triggers:** "Longest common subsequence", "Edit distance", "Wildcard matching".
- **Boundary Conditions:** Matrix dimensions are `(m + 1) x (n + 1)`. Access chars using `i - 1` and `j - 1`.
- **Real-World Application:** Git diff algorithms, DNA sequence alignment, text similarity search.

---

### [PAT-23] Sweep-Line & Interval Scheduling

- **Invariant:** Sort intervals by start time. Use a pointer or heap to process overlapping boundaries.
- **Mental Model:** Sweeping a vertical timeline left-to-right across time intervals.
- **Canonical Code Skeleton:**
```python
import heapq

def min_meeting_rooms(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])
    
    min_heap = [intervals[0][1]] # Stores end times
    
    for i in range(1, len(intervals)):
        if intervals[i][0] >= min_heap[0]:
            heapq.heappop(min_heap) # Room freed up!
        heapq.heappush(min_heap, intervals[i][1]) # Allocate room
        
    return len(min_heap)
```

- **Diagnostic Triggers:** "Meeting rooms II", "Merge intervals", "Non-overlapping intervals".
- **Boundary Conditions:** Always sort intervals by start time `a[0] - b[0]` first.
- **Real-World Application:** Calendar scheduling engines, hotel room allocation, cloud VM provisioning.

---

### [PAT-24] Trie Prefix Search & Retrieval

- **Invariant:** Tree structure where each node represents a character. Root-to-node path forms a string prefix, enabling $O(L)$ word lookup where $L$ is word length.
- **Mental Model:** Dictionary tree branching by character.
- **Canonical Code Skeleton:**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.is_word = True
        
    def search(self, word: str) -> bool:
        node = self._get_node(word)
        return node is not None and node.is_word
        
    def starts_with(self, prefix: str) -> bool:
        return self._get_node(prefix) is not None
        
    def _get_node(self, s: str) -> 'TrieNode':
        curr = self.root
        for c in s:
            if c not in curr.children:
                return None
            curr = curr.children[c]
        return curr
```

- **Diagnostic Triggers:** "Implement Trie", "Word search II (grid + dictionary)", "Replace words / autocomplete".
- **Boundary Conditions:** Use `c - 'a'` for lowercase alphabets. Set `isWord = true` at termination node.
- **Real-World Application:** Autocomplete search suggestions, IP routing prefix tables, spell checkers.

---

### [PAT-25] Priority Queue / Min-Max Heap

**Diagnostic Trigger:** "Find the K-th largest/smallest", "Merge K sorted lists", "Schedule tasks by priority", or any problem requiring efficient access to the minimum or maximum element while dynamically inserting.

**Invariant:** The heap property is maintained: for a min-heap, every parent node is ≤ its children. This guarantees O(1) access to the minimum and O(log N) insertion/extraction.

**Canonical Skeleton:**
```python
import heapq
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    freq_map = Counter(nums)
    
    min_heap = []
    for num, count in freq_map.items():
        heapq.heappush(min_heap, (count, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)
            
    return [num for count, num in min_heap]
```


**Complexity:** O(N log K) time, O(N + K) space.

> **Note on Mathematical and Bit Manipulation Patterns:** Several common interview problems rely on mathematical properties (XOR for finding missing/duplicate numbers, modular arithmetic, Gauss's sum formula) or bitwise operations (bitmask DP, bit counting). These techniques are cross-cutting tools that complement the structural patterns above rather than forming standalone patterns. When you encounter a problem involving XOR properties, power-of-two checks, or bitmask state encoding, recognize these as mathematical invariants that can be combined with the canonical patterns.


# Easy-tier Mastery — Implementation Speed, In-Place Transformations, and String Processing

The first question (Easy-tier) on the automated testing platforms General Coding Assessment (general coding assessment) is designed to evaluate fundamental implementation speed, boundary correctness, and memory hygiene. You have roughly **8 minutes** to solve Easy-tier. While categorized as "Easy," Easy-tier is where candidates most frequently drop valuable points — not because the problem is hard, but because they rush and introduce off-by-one errors, forget null checks, or use inefficient string concatenation. A perfect Easy-tier score is the foundation of a 750+ general coding assessment result.

This chapter teaches you the core vocabulary, the reusable pointer archetypes, 20 fully solved exemplar problems with detailed explanations, and 30 concrete practice problems with strategic hints.

* * *

## Essential Terminology & Vocabulary

Before solving any Easy-tier problem, you must internalize these foundational concepts. Each one maps directly to a class of problems you will encounter on the exam.

### In-Place Mutation
An algorithm is **in-place** if it transforms the input using $\mathcal{O}(1)$ auxiliary space (excluding the input itself). In Java, arrays are mutable references — you can overwrite `arr[i]` directly. Strings, however, are **immutable objects** — every modification creates a new heap allocation.

**Why it matters on Easy-tier:** Many Easy-tier problems explicitly require in-place modification. If you allocate a new array when the spec says "in-place," you lose points even if the output is correct.

### Read/Write Pointer Pattern
A two-pointer technique where:

- The **read pointer** scans every element sequentially (always moves forward).
- The **write pointer** only advances when an element passes a filter condition.

After the loop, `arr[0..write-1]` contains the filtered result. This pattern solves: *Remove Element*, *Move Zeros*, *Remove Duplicates from Sorted Array*, and *String Compression*.

![Read/Write Pointer — In-Place Array Compaction](editions/python/chapters/10-implementation-patterns/visuals/read_write_pointer.png){width=85%}

### Character Frequency Array (Fixed-Size, 256 or 26 Slots)
A fixed-size integer array indexed by character code point. Incrementing the counter at a character's index provides:

- $\mathcal{O}(1)$ per lookup/update (direct array access, no hashing)
- Zero heap allocations (lives on the stack in compiled languages)
- Deterministic performance (no hash collisions)

Use a 26-slot array when input is guaranteed lowercase English letters only (offset by `'a'`). Use a 256-slot array when input may contain any ASCII character.

**Comparison with HashMap/Dictionary:**

| Attribute | Fixed-Size Array (256) | HashMap / Dictionary |
| :--- | :--- | :--- |
| Access Time | $\mathcal{O}(1)$ direct | $\mathcal{O}(1)$ amortized (hash collisions possible) |
| Memory | ~1 KB fixed | Variable heap allocations |
| GC Pressure | Zero | Higher (key/value boxing in some languages) |
| When to Use | ASCII text, known char range | Unicode, arbitrary key types |

### Symmetrical Two-Pointer Convergence
Two pointers start at opposite ends (`left = 0`, `right = len - 1`) and move toward each other. The loop condition is `while (left < right)`. This pattern solves: *Palindrome Check*, *Reverse String*, *Two Sum in Sorted Array*, and *Container With Most Water*.

![Two-Pointer Convergence — Palindrome Verification](editions/python/chapters/10-implementation-patterns/visuals/two_pointer_convergence.png){width=85%}

### Run-Length Encoding (RLE)
Compress consecutive identical elements into `(element, count)` pairs. `"aaabbc"` becomes `"a3b2c1"`. The read pointer tracks the current run; the write pointer emits compressed output. This is a classic Easy-tier problem that combines the Read/Write pattern with counting.

### String Immutability & StringBuilder
In Java, `String` is immutable. The expression `s += char` inside a loop creates a **new String object on every iteration**, copying all previous characters. For a string of length $N$, this produces $\mathcal{O}(N^2)$ total character copies. Always use `StringBuilder` for loop-based string construction — it maintains a resizable `char[]` buffer internally and runs in amortized $\mathcal{O}(N)$.

### XOR Bit Manipulation for Uniqueness
The XOR operator (`^`) has two key properties: `a ^ a = 0` (same values cancel) and `a ^ 0 = a` (zero is identity). XOR-ing all elements in an array where every value appears twice except one produces the unique value. This runs in $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ space with zero branching.

### Prefix Sum / Running Total
A technique where you compute cumulative sums to answer range queries in $\mathcal{O}(1)$. For pivot index problems: `leftSum == totalSum - leftSum - nums[i]` identifies the balance point without nested loops.

### Integer Overflow & Boundary Guarding
This involves handling `Integer.MAX_VALUE` and `Integer.MIN_VALUE` constraints. It requires implementing safe comparisons before executing arithmetic operations to prevent exceeding limits.
Why it matters: Reverse-integer and palindrome-number problems require overflow detection.

### Digit Extraction Loop
This technique involves using the modulo operator `num % 10` to get the last digit of a number. You then use integer division `num / 10` to remove that digit for the next iteration.
Why it matters: This is fundamental for reverse integer, digit sum, and palindrome number checks.

### ASCII Arithmetic
This technique leverages character ASCII values for mathematical operations. Using `c - 'a'` converts a letter to a 0-25 index, `c - '0'` converts a char digit to an integer, and `(char)(i + 'a')` converts it back.
Why it matters: This pattern maps characters to array indices without relying on a HashMap.

### Boolean Flag / Sentinel Pattern
This involves using a single boolean variable to track whether a specific event has occurred across a scan. It monitors state changes cleanly throughout an iteration.
Why it matters: It simplifies complex conditions into clean single-variable tracking.

### Edge Case Taxonomy
This categorizes common input boundaries such as null, empty array/string, single element, all-same values, all-different values, and maximum integer values. Understanding this taxonomy ensures comprehensive test coverage.
Why it matters: You systematically test these BEFORE writing the main loop to catch 80% of bugs.

### Stack-Based Matching
This approach involves pushing opening delimiters onto a stack during traversal. Upon encountering a closing delimiter, you pop from the stack and verify the match.
Why it matters: This is the universal pattern for bracket, parentheses, and tag validation problems.

### Two-Pass Strategy
This algorithm design splits processing into two distinct phases. The first pass collects necessary data like counts, maximums, or positions, and the second pass acts on that collected information.
Why it matters: It avoids complex single-pass logic and significantly reduces bugs.

### Greedy Forward Scan
This strategy involves processing an array from left to right sequentially. At each step, you make the locally optimal choice without looking back.
Why it matters: It is heavily used in array change problems (like bumping each element above the previous) and similar Easy-tier tasks.

### Modular Arithmetic Basics
This encompasses foundational modulo operations for cyclic or remainder logic. Examples include using `n % 2` for parity, `n % k` for divisibility, and `(a + b - 1) / b` for ceiling division.
Why it matters: It avoids floating-point arithmetic entirely and handles circular increments efficiently.

### In-Place Swap
This is the standard programming idiom for swapping two variables using a temporary holder. It uses the `temp = a; a = b; b = temp;` pattern.
Why it matters: It serves as a fundamental building block for partitioning, reversing, and Dutch National Flag problems.

* * *

## Reusable Code Templates

These are the two most important templates to have memorized before the exam.

### Template A: Read/Write In-Place Filter

```python
# Retains elements satisfying a condition, overwrites list in-place
write = 0
for read in range(len(arr)):
    if keep_condition(arr[read]):
        arr[write] = arr[read]
        write += 1
# Result is arr[0..write-1], return write as the new length
```

**Used by:** Remove Element, Move Zeros, Remove Duplicates, Squeeze Spaces.

### Template B: Symmetric Converging Pointers

```python
left, right = 0, len(arr) - 1
while left < right:
    # Process or compare arr[left] and arr[right]
    # Optionally skip invalid elements
    left += 1
    right -= 1
```

**Used by:** Palindrome Check, Reverse Array, Two Sum (sorted), Sort Colors.

* * *

## Solved Exemplar Problems

**1. First Non-Repeating Character**
**Specification:** Given a string `s`, find the first character that appears exactly once. Return its 0-based index. If no unique character exists, return `-1`.

**Example:** `"leetcode"` → `0` (the character `'l'` appears once and is the first such character).

**Pattern:** Two-pass frequency array. First pass counts; second pass finds the first count of 1.
**Why two passes?** A single pass cannot determine uniqueness because later characters might duplicate earlier ones. The frequency array decouples counting from searching.

```python
def first_uniq_char(self, s: str) -> int:
    if not s:
        return -1

    # Pass 1: Count frequency of each character
    counts = [0] * 256
    for char in s:
        counts[ord(char)] += 1

    # Pass 2: Find first character with frequency exactly 1
    for i, char in enumerate(s):
        if counts[ord(char)] == 1:
            return i

    return -1 # All characters repeat
# Time: O(N), Space: O(1) — the counts list is constant size
```

* * *

**2. In-Place String Compression (Run-Length Encoding)**
**Specification:** Given a character array `chars`, compress it in-place using RLE. Consecutive duplicate characters are replaced by the character followed by the count (only if count > 1). Return the new length. You must modify `chars` in-place — no new array allocation.

**Example:** `['a','a','b','b','c','c','c']` → modified to `['a','2','b','2','c','3']`, return `6`.

**Pattern:** Read/Write pointers with a nested counting loop.

**Critical edge case:** When count exceeds 9 (e.g., count = 12), you must write `'1'` then `'2'` as separate characters.

```python
def compress(self, chars: list[str]) -> int:
    if not chars:
        return 0

    write = 0 # Write pointer for compressed output
    read = 0  # Read pointer scanning input

    while read < len(chars):
        current = chars[read]
        count = 0

        # Count consecutive occurrences of current character
        while read < len(chars) and chars[read] == current:
            read += 1
            count += 1

        # Write the character itself
        chars[write] = current
        write += 1

        # Write the count digits (only if count > 1)
        if count > 1:
            # Convert count to individual digit characters
            for digit in str(count):
                chars[write] = digit
                write += 1

    return write
# Time: O(N), Space: O(1) auxiliary
```


**Trace Walkthrough** (input: `['a','a','b','b','c','c','c']`):

| Step | read | write | Action | State |
|:---:|:----:|:-----:|:--------------|:-----------------------------------|
| Init | 0    | 0     | Start  | `['a','a','b','b','c','c','c']` |
| 1    | 2    | 2     | Run 'a' len 2 | `['a','2','b','b','c','c','c']` |
| 2    | 4    | 4     | Run 'b' len 2 | `['a','2','b','2','c','c','c']` |
| 3    | 7    | 6     | Run 'c' len 3 | `['a','2','b','2','c','3','c']` |
* * *

**3. Valid Palindrome with Non-Alphanumeric Skipping**
**Specification:** Given a string `s`, return `true` if it is a palindrome considering only alphanumeric characters and ignoring case. An empty string is a valid palindrome.

**Example:** `"A man, a plan, a canal: Panama"` → `true`.

**Pattern:** Symmetric converging pointers with skip logic for non-alphanumeric characters.

**Common mistake:** Forgetting to check `left < right` inside the skip-while loops, causing `ArrayIndexOutOfBoundsException` on strings like `".,,"`.

```python
def is_palindrome(self, s: str) -> bool:
    if s is None:
        return False

    left, right = 0, len(s) - 1

    while left < right:
        # Skip non-alphanumeric from the left
        while left < right and not s[left].isalnum():
            left += 1
        # Skip non-alphanumeric from the right
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True
# Time: O(N), Space: O(1)
```

* * *

**4. Move Zeros to End**
**Specification:** Given an integer array `nums`, move all `0`s to the end while maintaining the relative order of non-zero elements. Must be done in-place.

**Example:** `[0, 1, 0, 3, 12]` → `[1, 3, 12, 0, 0]`.

**Pattern:** Read/Write pointer. Non-zero elements are copied forward; remaining positions are filled with zeros.

**Why not swap?** Swapping works too, but the two-pass approach (copy then fill) is cleaner and less error-prone under time pressure.

```python
def move_zeroes(self, nums: list[int]) -> None:
    if not nums:
        return

    # Pass 1: Copy all non-zero elements to the front
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write] = nums[read]
            write += 1

    # Pass 2: Fill remaining positions with zeros
    while write < len(nums):
        nums[write] = 0
        write += 1
# Time: O(N), Space: O(1)
```

* * *

**5. Remove Duplicates from Sorted Array**
**Specification:** Given a sorted integer array `nums`, remove duplicates in-place so each element appears only once. Return the number of unique elements. The first `k` elements of `nums` should hold the result.

**Example:** `[1, 1, 2]` → `[1, 2, _]`, return `2`.

**Pattern:** Read/Write pointer. Since the array is sorted, duplicates are always adjacent. The write pointer advances only when `nums[read] != nums[write - 1]`.

```python
def remove_duplicates(self, nums: list[int]) -> int:
    if not nums:
        return 0

    write = 1 # First element is always unique
    for read in range(1, len(nums)):
        if nums[read] != nums[write - 1]:
            nums[write] = nums[read]
            write += 1

    return write
# Time: O(N), Space: O(1)
```

* * *

**6. Single Number (XOR Uniqueness)**
**Specification:** Given a non-empty array where every element appears exactly twice except one, find the single element. Must run in $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ space.

**Example:** `[4, 1, 2, 1, 2]` → `4`.

**Pattern:** XOR accumulation. `a ^ a = 0` cancels pairs; `a ^ 0 = a` preserves the unique element.

```python
def single_number(self, nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num # Pairs cancel, unique value survives
    return result
# Time: O(N), Space: O(1)
```

* * *

**7. Valid Parentheses**
**Specification:** Given a string containing only `(`, `)`, `{`, `}`, `[`, `]`, determine if the input is valid. Every open bracket must be closed by the same type in correct order.

**Example:** `"()[]{}"` → `true`. `"(]"` → `false`.

**Pattern:** Stack-based matching. On open bracket, push the expected closing bracket. On close bracket, pop and compare.
**Optimization:** Use a `char[]` as a manual stack to avoid `java.util.Stack` overhead.

```python
def is_valid(self, s: str) -> bool:
    if not s or len(s) % 2 != 0:
        return False

    stack = []

    for c in s:
        if c == '(': stack.append(')')
        elif c == '{': stack.append('}')
        elif c == '[': stack.append(']')
        else:
            if not stack or stack.pop() != c:
                return False

    return len(stack) == 0 # Stack must be empty
# Time: O(N), Space: O(N) worst case for the stack
```

* * *

**8. Reverse String In-Place**
**Specification:** Reverse a character array in-place using $\mathcal{O}(1)$ extra memory.

**Example:** `['h','e','l','l','o']` → `['o','l','l','e','h']`.

**Pattern:** Symmetric converging pointers with swap.

```python
def reverse_string(self, s: list[str]) -> None:
    if not s or len(s) <= 1:
        return

    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
# Time: O(N), Space: O(1)
```

* * *

**9. Pivot Index (Balance Point)**
**Specification:** Given array `nums`, find the leftmost index where the sum of elements to its left equals the sum of elements to its right. If no such index exists, return `-1`. The element at the pivot is excluded from both sums.

**Example:** `[1, 7, 3, 6, 5, 6]` → `3` (left sum `1+7+3 = 11`, right sum `5+6 = 11`).

**Pattern:** Prefix sum. Compute total sum first, then scan left-to-right maintaining a running left sum. At each index: `rightSum = totalSum - leftSum - nums[i]`.

```python
def pivot_index(self, nums: list[int]) -> int:
    if not nums:
        return -1

    total_sum = sum(nums)
    left_sum = 0
    
    for i, num in enumerate(nums):
        # right_sum = total_sum - left_sum - num
        if left_sum == total_sum - left_sum - num:
            return i
        left_sum += num

    return -1
# Time: O(N), Space: O(1)
```

* * *

**10. Check Array Monotonicity**
**Specification:** Return `true` if the array is entirely non-decreasing or entirely non-increasing.

**Example:** `[1, 2, 2, 3]` → `true`. `[6, 5, 4, 4]` → `true`. `[1, 3, 2]` → `false`.

**Pattern:** Dual boolean flags. Track both `isIncreasing` and `isDecreasing`. If an adjacent pair violates one direction, set its flag to false. Return true if either flag survives.

```python
def is_monotonic(self, nums: list[int]) -> bool:
    if not nums or len(nums) <= 2:
        return True

    increasing = True
    decreasing = True

    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]: increasing = False
        if nums[i] < nums[i + 1]: decreasing = False

    return increasing or decreasing
# Time: O(N), Space: O(1)
```

* * *

**11. Neighbor Sum Transformation**
**Specification:** Given array `A`, return array `B` where `B[i] = A[i-1] + A[i] + A[i+1]`. Treat out-of-bounds indices as `0`.

**Example:** `[4, 0, 1, -2, 3]` → `[4, 5, -1, 2, 1]`.

**Pattern:** Boundary-safe neighbor access with ternary guards.
**Why a new array?** Modifying `A` in-place would corrupt values needed for subsequent index calculations.

```python
def neighbor_sum(self, a: list[int]) -> list[int]:
    if not a:
        return []
    n = len(a)
    b = [0] * n

    for i in range(n):
        left_val = a[i - 1] if i > 0 else 0
        right_val = a[i + 1] if i < n - 1 else 0
        b[i] = left_val + a[i] + right_val

    return b
# Time: O(N), Space: O(N) for output array
```

* * *

**12. Maximum Subarray Sum of Fixed Window K**
**Specification:** Given integer array `nums` and integer `k`, find the maximum sum among all contiguous subarrays of exactly size `k`.

**Example:** `nums = [2, 1, 5, 1, 3, 2], k = 3` → `9` (subarray `[5, 1, 3]`).

**Pattern:** Fixed-size sliding window. Initialize window sum with first `k` elements, then slide by adding the entering element and subtracting the leaving element.

```python
def max_sum_subarray(self, nums: list[int], k: int) -> int:
    if not nums or len(nums) < k or k <= 0:
        return 0

    # Initialize sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide the window: add right element, remove left element
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum
# Time: O(N), Space: O(1)
```

* * *

**13. Find the Added Character**
**Specification:** String `t` is created by shuffling string `s` and inserting one extra character at a random position. Find and return that added character.

**Example:** `s = "abcd"`, `t = "abcde"` → `'e'`.

**Pattern:** XOR accumulation. XOR every character in both strings together. Paired characters cancel to zero; the extra character remains.

```python
def find_the_difference(self, s: str, t: str) -> str:
    result = 0
    for c in s: result ^= ord(c)
    for c in t: result ^= ord(c)
    return chr(result) # Only the unpaired character survives
# Time: O(N), Space: O(1)
```

* * *

**14. Capitalize or Reverse by Word Length Parity**
**Specification:** Given an array of words, transform each word: if the word's length is odd, convert to uppercase; if even, reverse its characters.

**Example:** `["Hello", "Data"]` → `["HELLO", "ataD"]`.

**Pattern:** Per-element transformation with parity branching.

```python
def transform_words(self, words: list[str]) -> list[str]:
    if not words:
        return []
    result = [""] * len(words)

    for i in range(len(words)):
        if len(words[i]) % 2 != 0:
            result[i] = words[i].upper()
        else:
            result[i] = words[i][::-1]

    return result
# Time: O(N * K) where K is average word length, Space: O(N * K) for output
```

* * *

**15. Check Equal Character Frequencies**
**Specification:** Return `true` if every character in string `s` appears the exact same number of times.

**Example:** `"abacbc"` → `true` (each of `a`, `b`, `c` appears 2 times). `"aaabb"` → `false`.

**Pattern:** Frequency array + validation scan. Count all characters (using a size 128 array to handle the full ASCII range), then verify every non-zero count matches.

```python
def are_occurrences_equal(self, s: str) -> bool:
    if not s:
        return True

    from collections import Counter
    counts = Counter(s)
    
    expected = 0
    for count in counts.values():
        if count > 0:
            if expected == 0: expected = count
            elif count != expected: return False

    return True
# Time: O(N), Space: O(1)
```

* * *

**16. Remove Element In-Place**
**Specification:** Given integer array `nums` and integer `val`, remove all occurrences of `val` in-place. Return the count of elements not equal to `val`. The first `k` positions of `nums` should contain the remaining elements.

**Example:** `nums = [3, 2, 2, 3], val = 3` → return `2`, array becomes `[2, 2, ...]`.

**Pattern:** Read/Write pointer — identical structure to Move Zeros.

```python
def remove_element(self, nums: list[int], val: int) -> int:
    if nums is None:
        return 0

    write = 0
    for read in range(len(nums)):
        if nums[read] != val:
            nums[write] = nums[read]
            write += 1

    return write
# Time: O(N), Space: O(1)
```

* * *

**17. Parity Alternation Validation**
**Specification:** Given an integer array, return `true` if every adjacent pair alternates between odd and even (i.e., no two adjacent elements share the same parity).

**Example:** `[1, 2, 3, 4]` → `true`. `[1, 3, 2]` → `false` (1 and 3 are both odd).

**Pattern:** Linear scan comparing `nums[i] % 2` with `nums[i+1] % 2`.
**Edge case with negatives:** `(-3) % 2` in Java returns `-1`, not `1`. Use `Math.abs(nums[i] % 2)` for safe parity checks.

```python
def is_alternating_parity(self, nums: list[int]) -> bool:
    if not nums or len(nums) <= 1:
        return True

    for i in range(len(nums) - 1):
        if (abs(nums[i]) % 2) == (abs(nums[i + 1]) % 2):
            return False

    return True
# Time: O(N), Space: O(1)
```

* * *

**18. Two Sum (Unsorted Array)**
**Specification:** Given an array of integers `nums` and an integer `target`, return the indices of two elements that add up to `target`. Each input has exactly one solution. You may not use the same element twice.

**Example:** `nums = [2, 7, 11, 15], target = 9` → `[0, 1]`.

**Pattern:** HashMap complement lookup. For each element, check if `target - nums[i]` has been seen. If yes, return both indices. If no, store `nums[i] → i` in the map.

```python
def two_sum(self, nums: list[int], target: int) -> list[int]:
    seen = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return [] # Should not reach here per problem guarantee
# Time: O(N), Space: O(N)
```

* * *

**19. Majority Element**
**Specification:** Given an array `nums` of size `n`, return the element that appears more than $\lfloor n/2 \rfloor$ times. The majority element is guaranteed to exist.

**Example:** `[2, 2, 1, 1, 1, 2, 2]` → `2`.

**Pattern:** Boyer–Moore Voting Algorithm. Maintain a candidate and a count. When count drops to zero, switch candidates. The majority element will always survive because it appears more than half the time.

```python
def majority_element(self, nums: list[int]) -> int:
    candidate = nums[0]
    count = 1

    for i in range(1, len(nums)):
        if count == 0:
            candidate = nums[i]
            count = 1
        elif nums[i] == candidate:
            count += 1
        else:
            count -= 1

    return candidate
# Time: O(N), Space: O(1)
```

* * *

**20. Plus One (Large Number as Array)**
**Specification:** Given a large integer represented as an array of digits (most significant digit first), increment the integer by one and return the resulting array.

**Example:** `[1, 2, 3]` → `[1, 2, 4]`. `[9, 9, 9]` → `[1, 0, 0, 0]`.

**Pattern:** Right-to-left carry propagation. Process digits from the least significant end. If a digit becomes 10, set it to 0 and carry. If no carry remains, return immediately.
**Edge case:** All 9s (`[9, 9, 9]`) require a new array of length `n + 1` with a leading 1.

```python
def plus_one(self, digits: list[int]) -> list[int]:
    for i in range(len(digits) - 1, -1, -1):
        digits[i] += 1
        if digits[i] < 10:
            return digits # No further carry needed
        digits[i] = 0 # Carry to next position

    # All digits were 9 — need a new array [1, 0, 0, ..., 0]
    return [1] + [0] * len(digits)
# Time: O(N), Space: O(1) amortized (O(N) only for all-9s edge case)
```

* * *


The following problems are drawn directly from the automated testing platforms Arcade and general coding assessment Easy-tier question bank. They emphasize boundary arithmetic, simple simulations, and filter-sort-reinsert patterns that appear frequently on actual assessments.

* * *

**21. Maximum Adjacent Element Product**
**Specification:** Given an array of integers, find the pair of adjacent elements that has the largest product. Return that product.

**Example:** `[3, 6, -2, -5, 7, 3]` → `21` (from pair `[7, 3]`).

**Invariant:** The maximum adjacent product can only occur between `arr[i]` and `arr[i+1]` for some valid `i`. A single linear scan tracking the running max is sufficient.

**Common mistake:** Forgetting that two large negative numbers produce a large positive product (e.g., `[-5, -4]` → `20`).

```python
def adjacent_elements_product(self, input_array: list[int]) -> int:
    if not input_array or len(input_array) < 2:
        return 0

    max_prod = input_array[0] * input_array[1]

    for i in range(1, len(input_array) - 1):
        prod = input_array[i] * input_array[i + 1]
        if prod > max_prod:
            max_prod = prod

    return max_prod
# Time: O(N), Space: O(1)
```

* * *

**22. Century From Year**
**Specification:** Given a year, return the century it belongs to. The first century spans year 1 through 100 inclusive, the second spans 101 through 200, etc.

**Example:** `1905` → `20`. `1700` → `17`. `2000` → `20`. `2001` → `21`.

**Pattern:** Integer ceiling division. The formula `(year + 99) / 100` computes the ceiling of `year / 100` using only integer arithmetic, avoiding floating-point rounding errors.

```python
def century_from_year(self, year: int) -> int:
    return (year + 99) // 100
# Time: O(1), Space: O(1)
```

* * *

**23. All Longest Strings**
**Specification:** Given an array of strings, return a new array containing all strings that share the maximum length.

**Example:** `["aba", "aa", "ad", "vcd", "aba"]` → `["aba", "vcd", "aba"]`.

**Pattern:** Two-pass filter. Pass 1 finds the maximum string length. Pass 2 collects all strings matching that length.
**Why two passes?** A single pass would require backtracking to remove shorter strings discovered before the true maximum is known.

```python
def all_longest_strings(self, input_array: list[str]) -> list[str]:
    # Pass 1: Find the maximum length
    max_length = 0
    for s in input_array:
        if len(s) > max_length:
            max_length = len(s)

    # Pass 2: Collect strings matching the max length
    result = []
    for s in input_array:
        if len(s) == max_length:
            result.append(s)

    return result
# Time: O(N), Space: O(N) for output
```

* * *

**24. Common Character Count**
**Specification:** Given two strings `s1` and `s2`, find the number of common characters between them. Each character match consumes one occurrence from each string.

**Example:** `s1 = "aabcc"`, `s2 = "adcaa"` → `3` (common: `'a'`, `'a'`, `'c'`).

**Pattern:** Dual frequency arrays with element-wise minimum. Build `int[26]` for each string. The number of shared instances of character `c` is `Math.min(count1[c], count2[c])`.

```python
def common_character_count(self, s1: str, s2: str) -> int:
    count1 = [0] * 26
    count2 = [0] * 26

    for c in s1: count1[ord(c) - ord('a')] += 1
    for c in s2: count2[ord(c) - ord('a')] += 1

    common = 0
    for i in range(26):
        common += min(count1[i], count2[i])

    return common
# Time: O(N + M), Space: O(1) — fixed 26-element lists
```

* * *

**25. Lucky Ticket (Digit Sum Halves)**
**Specification:** A ticket number (even number of digits) is "lucky" if the sum of its first-half digits equals the sum of its second-half digits. Determine if a given number is lucky.

**Example:** `1230` → `true` (`1 + 2 = 3`, `3 + 0 = 3`). `239017` → `false` (`2+3+9 = 14`, `0+1+7 = 8`).

**Pattern:** Convert to string for digit access. Split at midpoint. Sum each half independently.

```python
def is_lucky(self, n: int) -> bool:
    s = str(n)
    mid = len(s) // 2
    sum1 = 0
    sum2 = 0

    for i in range(mid):
        sum1 += int(s[i])       # First half digit
        sum2 += int(s[i + mid]) # Second half digit

    return sum1 == sum2
# Time: O(D) where D is digit count, Space: O(D) for string conversion
```

* * *

**26. Sort By Height (Obstacles in Place)**
**Specification:** People are standing in a row with immovable trees (represented by `-1`) between them. Sort the people by height in non-descending order without moving the trees.

**Example:** `[-1, 150, 190, 170, -1, -1, 160, 180]` → `[-1, 150, 160, 170, -1, -1, 180, 190]`.

**Pattern:** Filter-Sort-Reinsert. Extract non-tree values into a separate list, sort that list, then write the sorted values back into the original array at non-tree positions only.

**Invariant:** Tree positions (`-1`) are never touched. Only human positions are modified.

```python
def sort_by_height(self, a: list[int]) -> list[int]:
    # Step 1: Extract all non-tree heights
    heights = [h for h in a if h != -1]

    # Step 2: Sort the extracted heights
    heights.sort()

    # Step 3: Reinsert sorted heights at non-tree positions
    index = 0
    for i in range(len(a)):
        if a[i] != -1:
            a[i] = heights[index]
            index += 1

    return a
# Time: O(N log N) for sorting, Space: O(N) for extracted list
```

* * *

**27. Alternating Team Sums**
**Specification:** People in a row are divided into two teams by alternating index: person 0 → Team 1, person 1 → Team 2, person 2 → Team 1, etc. Return the total weight of each team as `[team1Sum, team2Sum]`.

**Example:** `[50, 60, 60, 45, 70]` → `[180, 105]`.

**Pattern:** Index parity accumulation. `i % 2 == 0` accumulates into Team 1, `i % 2 == 1` into Team 2.

```python
def alternating_sums(self, a: list[int]) -> list[int]:
    team1 = 0
    team2 = 0

    for i in range(len(a)):
        if i % 2 == 0:
            team1 += a[i]
        else:
            team2 += a[i]

    return [team1, team2]
# Time: O(N), Space: O(1)
```

* * *

**28. Add Border to Character Matrix**
**Specification:** Given a rectangular array of strings (representing rows of a character matrix), add a border of asterisks (`*`) around it. Return the new bordered matrix.

**Example:** `["abc", "ded"]` → `["*****", "*abc*", "*ded*", "*****"]`.

**Pattern:** String construction with dimensional arithmetic. New width = original width + 2. New height = original height + 2. First and last rows are full asterisk strings. Middle rows are wrapped with `*` on each side.

```python
def add_border(self, picture: list[str]) -> list[str]:
    new_width = len(picture[0]) + 2
    result = [""] * (len(picture) + 2)

    # Build the border row
    border = '*' * new_width

    # Top border
    result[0] = border

    # Wrap each interior row with side asterisks
    for i in range(len(picture)):
        result[i + 1] = f"*{picture[i]}*"

    # Bottom border
    result[-1] = border

    return result
# Time: O(rows * cols), Space: O(rows * cols) for output
```

* * *

**29. Array Change (Minimum Moves for Strict Increase)**
**Specification:** Given an integer array, find the minimum number of single-increment moves needed to make the sequence strictly increasing (every element must be greater than the previous one).

**Example:** `[1, 1, 1]` → `3` (sequence becomes `[1, 2, 3]`). `[3, 2]` → `2` (sequence becomes `[3, 4]`).

**Pattern:** Greedy forward scan. At each position `i`, if `arr[i] <= arr[i-1]`, compute the deficit `arr[i-1] - arr[i] + 1`, increment `arr[i]` by that amount, and accumulate the moves.

**Invariant:** After processing index `i`, the constraint `arr[i] > arr[i-1]` is guaranteed. The greedy minimum at each step is globally optimal because increasing `arr[i]` to `arr[i-1] + 1` (the smallest valid value) minimizes cascading costs downstream.

```python
def array_change(self, input_array: list[int]) -> int:
    moves = 0

    for i in range(1, len(input_array)):
        if input_array[i] <= input_array[i - 1]:
            # Calculate the minimum increment needed
            deficit = input_array[i - 1] - input_array[i] + 1
            input_array[i] += deficit
            moves += deficit

    return moves
# Time: O(N), Space: O(1)
```

* * *

**30. Matrix Elements Sum (Haunted Rooms)**
**Specification:** A building is represented as a 2D matrix where each element is the rent price of a room. Rooms directly below a free room (value `0`) on any floor are also considered "haunted" and should be excluded from the total. Calculate the sum of all non-haunted rooms.

**Example:** `[[0, 1, 1, 2], [0, 5, 0, 0], [2, 0, 3, 3]]` → `9` (rooms below any `0` in the column above are excluded).

**Pattern:** Column-wise top-down scan with a boolean "poisoned" flag per column. Once a `0` is encountered in a column, all values below it in that column are skipped.

```python
def matrix_elements_sum(self, matrix: list[list[int]]) -> int:
    rows = len(matrix)
    cols = len(matrix[0])
    total = 0

    for c in range(cols):
        for r in range(rows):
            if matrix[r][c] == 0:
                break # All rooms below are haunted — skip rest of column
            total += matrix[r][c]

    return total
# Time: O(rows * cols), Space: O(1)
```

* * *

**31. Almost Increasing Sequence**
**Specification:** Given a sequence of integers, determine whether it is possible to obtain a strictly increasing sequence by removing no more than one element.

**Example:** `[1, 3, 2, 1]` → `false`. `[1, 3, 2]` → `true` (remove `3` → `[1, 2]`).

**Pattern:** Count violations (positions where `arr[i] >= arr[i+1]`). If zero violations, it is already increasing. If exactly one violation at position `i`, check two removal candidates: removing `arr[i]` or removing `arr[i+1]`. If either removal produces a valid increasing sequence around the gap, return `true`. If more than one violation, return `false`.
**This is one of the trickiest Easy-tier problems.** The naive approach of "just remove one element and re-check" is $\mathcal{O}(N^2)$. The optimal approach is $\mathcal{O}(N)$.

```python
def almost_increasing_sequence(self, sequence: list[int]) -> bool:
    count = 0   # Number of violations
    bad_idx = -1  # Index of first violation

    for i in range(len(sequence) - 1):
        if sequence[i] >= sequence[i + 1]:
            count += 1
            bad_idx = i
            if count > 1: return False # More than one violation

    if count == 0: return True # Already strictly increasing

    # Try removing element at bad_idx
    if bad_idx == 0 or sequence[bad_idx - 1] < sequence[bad_idx + 1]:
        return True

    # Try removing element at bad_idx + 1
    if bad_idx + 2 >= len(sequence) or sequence[bad_idx] < sequence[bad_idx + 2]:
        return True

    return False
# Time: O(N), Space: O(1)
```


**Trace Walkthrough** (input: `[1, 3, 2, 1]`):

| Step | i | nums[i] | nums[i+1] | Violation? | Action | State (Violations) |
|:---:|:---:|:-------:|:---------:|:----------:|:-------|:-------------------|
| 1    | 0 | 1       | 3         | No         | Continue | 0 |
| 2    | 1 | 3       | 2         | Yes        | Check removals | 1 |
| 3    | 2 | 2       | 1         | Yes        | Return false   | >1 |
* * *

**32. Reverse Parentheses (Nested String Reversal)**
**Specification:** Given a string `s` with lowercase letters and parentheses, reverse the strings in each pair of matching parentheses, starting from the innermost pair. Remove the parentheses from the result.

**Example:** `"(abcd)"` → `"dcba"`. `"(u(love)i)"` → `"iloveu"`. `"(ed(et(oc))el)"` → `"leetcode"`.

**Pattern:** Stack-based simulation. Use a stack of `StringBuilder`s. When `(` is encountered, push a new builder. When `)` is encountered, pop the top builder, reverse it, and append its contents to the new top of the stack.

```python
def reverse_in_parentheses(self, s: str) -> str:
    stack = [[]]

    for c in s:
        if c == '(':
            stack.append([]) # Start new nested context
        elif c == ')':
            inner = stack.pop()  # Pop innermost context
            inner.reverse()       # Reverse it
            stack[-1].extend(inner) # Append to enclosing context
        else:
            stack[-1].append(c)  # Accumulate character

    return "".join(stack[0])
# Time: O(N^2) worst case for nested reversals, Space: O(N)
```


**Trace Walkthrough** (input: `"(u(love)i)"`):

| Step | char | Action | Stack | Current String |
|:---:|:----:|:-------|:------|:---------------|
| 1    | '('  | Push new | `[""]` | `""` |
| 2    | 'u'  | Append   | `[""]` | `"u"` |
| 3    | '('  | Push new | `["", "u"]` | `""` |
| 4    | 'l'..'e' | Append | `["", "u"]` | `"love"` |
| 5    | ')'  | Pop & Reverse | `[""]` | `"uevol"` |
| 6    | 'i'  | Append   | `[""]` | `"uevoli"` |
| 7    | ')'  | Pop & Reverse | `[]` | `"iloveu"` |
* * *

## Practice Problem Bank

The following 30 problems cover every Easy-tier pattern you may encounter on the General Coding Assessments. Each includes a full specification, concrete examples, input constraints, and a strategic hint pointing you toward the correct pattern.

* * *

**1. Reverse Words in a Sentence**
**Specification:** Given a string `s` containing words separated by single spaces, reverse the order of the words. Leading/trailing spaces should be removed, and multiple spaces between words should be reduced to a single space.

**Example:** `"  the sky is blue  "` → `"blue is sky the"`.

*Constraints:* $1 \le |s| \le 10^4$. `s` contains English letters, digits, and spaces.

**Strategic Hint:** Split on whitespace, filter empty strings, reverse the resulting array, and join with single spaces. Alternatively, reverse the entire string, then reverse each word individually for an in-place solution.

* * *

**2. Rotate Array by K Steps**
**Specification:** Given an integer array `nums`, rotate the array to the right by `k` steps in-place.

**Example:** `nums = [1,2,3,4,5,6,7], k = 3` → `[5,6,7,1,2,3,4]`.

*Constraints:* $1 \le n \le 10^5$, $0 \le k \le 10^5$. Handle `k > n` by taking `k % n`.

**Strategic Hint:** Three-reverse trick: reverse entire array, reverse first `k` elements, reverse remaining `n - k` elements. All in $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ space.

* * *

**3. Contains Duplicate**
**Specification:** Given an integer array `nums`, return `true` if any value appears at least twice.

**Example:** `[1, 2, 3, 1]` → `true`. `[1, 2, 3, 4]` → `false`.

*Constraints:* $1 \le n \le 10^5$, $-10^9 \le nums[i] \le 10^9$.

**Strategic Hint:** Use a `HashSet`. Add each element — if `add()` returns `false`, a duplicate exists. $\mathcal{O}(N)$ time.

* * *

**4. Length of Last Word**
**Specification:** Given a string `s` of words and spaces, return the length of the last word. A word is a maximal substring of non-space characters.

**Example:** `"Hello World"` → `5`. `"   fly me   to   the moon  "` → `4`.

*Constraints:* $1 \le |s| \le 10^4$. `s` contains only English letters and spaces.

**Strategic Hint:** Scan backwards from the end. Skip trailing spaces, then count consecutive non-space characters. No need to split the entire string.

* * *

**5. Roman Numeral to Integer**
**Specification:** Convert a Roman numeral string to its integer value. Roman numerals: I=1, V=5, X=10, L=50, C=100, D=500, M=1000. Subtractive cases: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900.

**Example:** `"MCMXCIV"` → `1994`.

*Constraints:* $1 \le |s| \le 15$. Input is guaranteed valid.

**Strategic Hint:** Scan left-to-right. If the current symbol's value is less than the next symbol's value, subtract it (subtractive case). Otherwise, add it. Use a `Map<Character, Integer>` or switch statement for value lookup.

* * *

**6. Missing Number in Range**
**Specification:** Given an array `nums` containing `n` distinct numbers in the range `[0, n]`, return the one number in the range that is missing.

**Example:** `[3, 0, 1]` → `2`. `[0, 1]` → `2`.

*Constraints:* $n = nums.length$, $0 \le nums[i] \le n$. All numbers are unique.

**Strategic Hint:** Use Gauss's formula: expected sum = $n \times (n + 1) / 2$. Subtract the actual sum. The difference is the missing number. $\mathcal{O}(N)$ time, $\mathcal{O}(1)$ space. Alternatively, XOR all indices and values.

* * *

**7. Merge Two Sorted Arrays into One**
**Specification:** Given two sorted integer arrays `nums1` (length `m + n` with trailing zeros as placeholders) and `nums2` (length `n`), merge `nums2` into `nums1` in-place so the result is sorted.

**Example:** `nums1 = [1,2,3,0,0,0], m = 3`, `nums2 = [2,5,6], n = 3` → `nums1 = [1,2,2,3,5,6]`.

*Constraints:* $0 \le m, n \le 200$.

**Strategic Hint:** Merge from the back (right-to-left). Compare `nums1[m-1]` with `nums2[n-1]` and place the larger one at `nums1[m+n-1]`. This avoids overwriting unprocessed elements.

* * *

**8. Find Numbers with Even Number of Digits**
**Specification:** Given an array of integers, return the count of elements that have an even number of digits.

**Example:** `[12, 345, 2, 6, 7896]` → `2` (only `12` and `7896` have an even number of digits).

*Constraints:* $1 \le n \le 500$, $1 \le nums[i] \le 10^5$.

**Strategic Hint:** For each number, count digits using `Integer.toString(num).length()` or repeated division by 10. Check if the digit count is even.

* * *

**9. Implement strStr() — Find First Occurrence**
**Specification:** Given two strings `haystack` and `needle`, return the index of the first occurrence of `needle` in `haystack`, or `-1` if not found.

**Example:** `haystack = "sadbutsad", needle = "sad"` → `0`.

*Constraints:* $1 \le |haystack|, |needle| \le 10^4$.

**Strategic Hint:** Iterate from index `0` to `haystack.length() - needle.length()`. At each position, check if the substring matches using `haystack.substring(i, i + needle.length()).equals(needle)` or a character-by-character comparison loop.

* * *

**10. Intersection of Two Arrays (Unique Elements)**
**Specification:** Given two integer arrays `nums1` and `nums2`, return an array of their unique intersection. Each element must appear at most once in the result.

**Example:** `nums1 = [1,2,2,1], nums2 = [2,2]` → `[2]`.

*Constraints:* $1 \le n \le 1000$.

**Strategic Hint:** Add all elements of `nums1` into a `HashSet`. Iterate `nums2` and check membership. Use a second `HashSet` for the result to avoid duplicates.

* * *

**11. Valid Anagram**
**Specification:** Given two strings `s` and `t`, return `true` if `t` is an anagram of `s` (same characters, same frequencies, different order).

**Example:** `s = "anagram", t = "nagaram"` → `true`.

*Constraints:* $1 \le |s|, |t| \le 5 \times 10^4$.

**Strategic Hint:** Use an `int[26]` frequency array. Increment for characters in `s`, decrement for characters in `t`. If all counts are zero at the end, it is an anagram.

* * *

**12. Remove All Adjacent Duplicates in String**
**Specification:** Repeatedly remove adjacent pairs of equal characters until no more removals are possible. Return the final string.

**Example:** `"abbaca"` → remove `"bb"` → `"aaca"` → remove `"aa"` → `"ca"`.

*Constraints:* $1 \le |s| \le 10^5$.

**Strategic Hint:** Use a `StringBuilder` as a stack. For each character, if it matches the last character in the builder, pop (delete last). Otherwise, append. Single pass, $\mathcal{O}(N)$.

* * *

**13. Best Time to Buy and Sell Stock (Single Transaction)**
**Specification:** Given array `prices` where `prices[i]` is the price on day `i`, find the maximum profit from buying on one day and selling on a later day. If no profit is possible, return `0`.

**Example:** `[7, 1, 5, 3, 6, 4]` → `5` (buy at `1`, sell at `6`).

*Constraints:* $1 \le n \le 10^5$.

**Strategic Hint:** Track `minPriceSoFar` as you scan left-to-right. At each day, compute `profit = prices[i] - minPriceSoFar` and update `maxProfit`. Single pass, $\mathcal{O}(N)$.

* * *

**14. Jewels and Stones**
**Specification:** Given string `jewels` (types of jewel stones, each unique) and string `stones` (stones you have), count how many of your stones are jewels.

**Example:** `jewels = "aA", stones = "aAAbbbb"` → `3`.

*Constraints:* $1 \le |jewels|, |stones| \le 50$.

**Strategic Hint:** Put all jewel characters in a `HashSet`. Iterate through stones, count membership matches. $\mathcal{O}(J + S)$ time.

* * *

**15. Squeeze Multiple Spaces to Single Space**
**Specification:** Given a string with multiple consecutive spaces between words, replace each sequence of spaces with a single space. Trim leading and trailing spaces.

**Example:** `"  hello    world  "` → `"hello world"`.

*Constraints:* $1 \le |s| \le 10^4$.

**Strategic Hint:** Use the Read/Write pattern on a `char[]`. Write a space only if the previous written character is not already a space. Skip leading spaces by initializing a flag or checking `write == 0`.

* * *

**16. Check if Array is Sorted and Rotated**
**Specification:** Given an array `nums`, return `true` if the array was originally sorted in non-decreasing order and then rotated some number of positions (including zero).

**Example:** `[3, 4, 5, 1, 2]` → `true`. `[2, 1, 3, 4]` → `false`.

*Constraints:* $1 \le n \le 100$.

**Strategic Hint:** Count the number of "descents" (positions where `nums[i] > nums[i+1]`, wrapping around to compare `nums[n-1]` with `nums[0]`). A valid sorted-and-rotated array has at most one descent.

* * *

**17. Running Sum of 1D Array**
**Specification:** Given array `nums`, return an array where `result[i] = sum(nums[0]..nums[i])`.

**Example:** `[1, 2, 3, 4]` → `[1, 3, 6, 10]`.

*Constraints:* $1 \le n \le 1000$.

**Strategic Hint:** Modify in-place: `nums[i] += nums[i-1]` for `i >= 1`. Single pass, $\mathcal{O}(N)$, $\mathcal{O}(1)$ extra space.

* * *

**18. Count Common Characters in Array of Strings**
**Specification:** Given an array of lowercase strings, find all characters that appear in every string (including duplicates). Return them as a list.

**Example:** `["bella", "label", "roller"]` → `["e", "l", "l"]`.

*Constraints:* $1 \le n \le 100$, $1 \le |s_i| \le 100$.

**Strategic Hint:** Use an `int[26]` initialized to `Integer.MAX_VALUE`. For each string, compute its own `int[26]` frequency, then take the element-wise minimum with the global array. The final array represents the minimum frequency of each character across all strings.

* * *

**19. Maximum Consecutive Ones**
**Specification:** Given a binary array `nums`, return the maximum number of consecutive `1`s.

**Example:** `[1, 1, 0, 1, 1, 1]` → `3`.

*Constraints:* $1 \le n \le 10^5$.

**Strategic Hint:** Track `currentStreak` and `maxStreak`. When `nums[i] == 1`, increment `currentStreak`. When `nums[i] == 0`, reset `currentStreak` to 0. Update `maxStreak` at each step.

* * *

**20. Determine if String Halves Are Alike**
**Specification:** A string `s` of even length is split into two halves. Return `true` if both halves contain the same number of vowels (`a, e, i, o, u` — case-insensitive).

**Example:** `"book"` → `true` (`"bo"` has 1 vowel, `"ok"` has 1 vowel).

*Constraints:* $2 \le |s| \le 1000$, `|s|` is even.

**Strategic Hint:** Count vowels in `s[0..n/2-1]` and `s[n/2..n-1]`. Compare counts. $\mathcal{O}(N)$.

* * *

**21. Sign of the Product of an Array**
**Specification:** Return `1` if the product of all elements is positive, `-1` if negative, `0` if zero. Do not compute the actual product (it may overflow).

**Example:** `[-1, -2, -3, -4, 3, 2, 1]` → `1` (product is positive).

*Constraints:* $1 \le n \le 1000$.

**Strategic Hint:** If any element is 0, return 0. Otherwise, count negative numbers. If the count is even, the product is positive; if odd, negative.

* * *

**22. Replace Elements with Greatest on Right Side**
**Specification:** Given array `arr`, replace each element with the greatest element to its right. The last element should be replaced with `-1`.

**Example:** `[17, 18, 5, 4, 6, 1]` → `[18, 6, 6, 6, 1, -1]`.

*Constraints:* $1 \le n \le 10^4$.

**Strategic Hint:** Scan right-to-left tracking `maxSoFar`. At each position, the answer is `maxSoFar`, then update `maxSoFar = Math.max(maxSoFar, originalValue)`.

* * *

**23. Sort Array by Parity**
**Specification:** Rearrange array so all even elements come before all odd elements. Relative order within even/odd groups does not matter.

**Example:** `[3, 1, 2, 4]` → `[2, 4, 3, 1]` (any valid ordering).

*Constraints:* $1 \le n \le 5000$.

**Strategic Hint:** Read/Write pointer: write pointer tracks the next "even slot." Swap `nums[write]` with `nums[read]` whenever `nums[read]` is even.

* * *

**24. Convert Sorted Array to Binary Search Tree**
**Specification:** Given a sorted integer array, create a height-balanced Binary Search Tree (BST).

**Example:** `[-10, -3, 0, 5, 9]` → BST with root `0`, left subtree `[-10, -3]`, right subtree `[5, 9]`.

*Constraints:* $1 \le n \le 10^4$.

**Strategic Hint:** Recursive binary split. The middle element becomes the root. Left half becomes the left subtree; right half becomes the right subtree. Base case: empty range returns null.

* * *

**25. Richest Customer Wealth**
**Specification:** Given 2D array `accounts` where `accounts[i][j]` is the amount of money customer `i` has in bank `j`, return the wealth of the richest customer (sum of all bank balances).

**Example:** `[[1,2,3],[3,2,1]]` → `6`.

*Constraints:* $1 \le m, n \le 50$.

**Strategic Hint:** For each customer, compute row sum. Track the maximum row sum. Two nested loops, $\mathcal{O}(m \times n)$.

* * *

**26. Number of Good Pairs**
**Specification:** Given array `nums`, count the number of pairs `(i, j)` where `i < j` and `nums[i] == nums[j]`.

**Example:** `[1, 2, 3, 1, 1, 3]` → `4`.

*Constraints:* $1 \le n \le 100$, $1 \le nums[i] \le 100$.

**Strategic Hint:** Use a frequency array. For each number, if it has been seen `c` times before, it forms `c` new pairs. Increment count by `c`, then increment frequency.

* * *

**27. Check if N and Its Double Exist**
**Specification:** Given array `arr`, check if there exist two indices `i` and `j` such that `i != j` and `arr[i] == 2 * arr[j]`.

**Example:** `[10, 2, 5, 3]` → `true` (10 = 2 * 5).

*Constraints:* $2 \le n \le 500$.

**Strategic Hint:** Use a `HashSet`. For each element, check if `2 * num` or `num / 2` (when `num` is even) is already in the set. Then add `num` to the set.

* * *

**28. Truncate Sentence**
**Specification:** Given a sentence `s` and integer `k`, truncate `s` to contain only the first `k` words.

**Example:** `s = "Hello how are you Contestant", k = 4` → `"Hello how are you"`.

*Constraints:* $1 \le |s| \le 500$, $1 \le k \le$ word count.

**Strategic Hint:** Scan character-by-character counting spaces. When the space count reaches `k`, return `s.substring(0, i)`. Or split and rejoin the first `k` tokens.

* * *

**29. Cells in Range on an Excel Sheet**
**Specification:** Given a string `s` representing a cell range like `"K1:L2"`, return all cells in the range in row-major order.

**Example:** `"K1:L2"` → `["K1", "K2", "L1", "L2"]`.

*Constraints:* Column is a single uppercase letter, row is a single digit.

**Strategic Hint:** Parse start column, start row, end column, end row. Nested loop: outer on columns (`col = s.charAt(0)` to `s.charAt(3)`), inner on rows. Build each cell string.

* * *

**30. Sum of Digits Until Single Digit**
**Specification:** Given a non-negative integer `num`, repeatedly add its digits until the result is a single digit. Return that digit.

**Example:** `38` → `3 + 8 = 11` → `1 + 1 = 2`. Return `2`.

*Constraints:* $0 \le num \le 2^{31} - 1$.

**Strategic Hint:** Iterative approach: extract digits with `num % 10`, accumulate sum, reduce with `num = sum`. Repeat until `num < 10`. Mathematical shortcut: Digital Root formula `1 + (num - 1) % 9` for $\mathcal{O}(1)$.


# Medium-tier Mastery — 2D Matrix Traversal, Grid Simulations, and State Machine Processing

This chapter covers Medium-tier of the General Coding Assessments (Medium difficulty, ~15 minutes target time). Medium-tier tests multidimensional array processing, grid boundary control, BFS/DFS flood fill, and step-by-step state machine simulation.

## Essential Terminology & Vocabulary

*   **Row-Major vs Column-Major layout**: Row-major layout stores 2D arrays row by row in memory (used in Java, C/C++), while column-major stores them column by column (Fortran, MATLAB). In Java, `matrix[r][c]` means row `r`, column `c`. Traversing row-major arrays by row is cache-friendly and faster.
*   **In-Place Matrix Transposition**: The process of flipping a matrix over its main diagonal without allocating a new matrix. Mathematical formula: $A^T[i][j] = A[j][i]$. For an $N \times N$ matrix, iterate `i` from 0 to N-1 and `j` from `i+1` to N-1, swapping `matrix[i][j]` and `matrix[j][i]`.
*   **90-Degree Clockwise/Counter-Clockwise Rotation Theorem**: Rotating a grid 90° can be done with two simpler operations. Clockwise: Transpose the matrix, then reverse each row. Counter-Clockwise: Transpose the matrix, then reverse each column.
*   **Spiral Matrix Boundary Contraction**: A traversal technique using four pointer boundaries (`top`, `bottom`, `left`, `right`). We traverse the perimeter, then shrink the boundaries (e.g., `top++`, `right--`) and repeat until the boundaries overlap.
*   **Coordinate Direction Vectors**: Pre-defined arrays to cleanly iterate through grid neighbors. Standard 4-directional setup: `int[] dr = {-1, 1, 0, 0}; int[] dc = {0, 0, -1, 1};`. This prevents writing four repetitive `if` statements for North, South, West, East.
*   **Flood Fill / BFS vs DFS on grids**: Techniques to traverse connected components in a matrix. DFS uses recursion (call stack) to go deep, which is easier to write but can cause stack overflow on massive grids. BFS uses a `Queue` to process level-by-level, ideal for shortest path calculations.
*   **2D Prefix Sum**: A precomputation technique where `prefix[i][j]` stores the sum of all elements in the submatrix from `(0,0)` to `(i-1,j-1)`. Allows answering arbitrary submatrix sum queries in $\mathcal{O}(1)$ time using inclusion-exclusion.

*   **State Machine Simulation**: Problems where you process a sequence of commands or instructions step-by-step. Often requires maintaining a "current state" (e.g., direction, coordinate, phase) and applying transition logic based on the input stream.
*   **Toeplitz Matrix**: A matrix in which every diagonal descending from left to right has constant values. Property to check: `matrix[i][j] == matrix[i-1][j-1]` for all valid $i>0, j>0$.
*   **In-Place State Encoding**: A trick to update states simultaneously without a copy grid. We use bits or sentinel values to encode both "old state" and "new state" (e.g., 0=dead, 1=live, 2=was dead now live, 3=was live now dead) and later decode it with modulo/division.

### Row-Major Index Linearization
This technique converts 2D coordinates into a 1D index using `index = r * cols + c`. It can also reverse the process using `r = index / cols` and `c = index % cols`.
Why it matters: It is needed for matrix reshape operations and binary searching in a sorted matrix.

### BFS Level-by-Level Tracking
This approach uses an inner loop based on `int size = queue.size()` inside the standard BFS `while` loop. This ensures the algorithm processes one full level of nodes before advancing depth.
Why it matters: It is crucial for calculating minimum steps, rotting oranges, and shortest path problems.

### Visited Set vs In-Place Marking
This evaluates the trade-off between allocating a separate `boolean[][] visited` array and modifying grid cells directly, such as setting `grid[r][c] = '#'`. In-place marking avoids extra memory allocation but destroys the original matrix.
Why it matters: In-place marking saves memory but mutates the input, which is a key discussion point in interviews.

### Diagonal Traversal Pattern
This property states that elements sharing the same `r + c` sum belong to the same anti-diagonal. Conversely, elements sharing the same `r - c` difference are on the same main diagonal.
Why it matters: This pattern is key for zigzag traversals and Toeplitz matrix verification.

### Boundary Validation Helper
This is the practice of extracting boundary logic into a separate `boolean inBounds(r, c, rows, cols)` utility method. It centralizes coordinate checks during grid traversal.
Why it matters: It eliminates repetitive boundary checks and significantly reduces bugs in grid BFS/DFS.

### Multi-Source BFS
Instead of running BFS individually from each source, this technique seeds the initial queue with ALL starting positions simultaneously. The search then expands outwards concurrently from multiple origins.
Why it matters: It solves rotting oranges and walls-and-gates problems in a single, highly efficient BFS pass.

![Multi-Source BFS — Rotting Oranges Wavefront](editions/python/chapters/11-matrix-grid-patterns/visuals/bfs_grid_levels.png){width=85%}

## Reusable Code Templates

### Template A: Spiral Boundary Traversal
```python
top, bottom = 0, len(matrix) - 1
left, right = 0, len(matrix[0]) - 1
while top <= bottom and left <= right:
    for j in range(left, right + 1): pass # process matrix[top][j]
    top += 1
    for i in range(top, bottom + 1): pass # process matrix[i][right]
    right -= 1
    if top <= bottom:
        for j in range(right, left - 1, -1): pass # process matrix[bottom][j]
        bottom -= 1
    if left <= right:
        for i in range(bottom, top - 1, -1): pass # process matrix[i][left]
        left += 1
```

![Spiral Boundary Traversal — Layer-by-Layer Contraction](editions/python/chapters/11-matrix-grid-patterns/visuals/spiral_traversal.png){width=85%}

### Template B: 4-Directional BFS/DFS Grid Walk
```python
dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]

def dfs(grid: list[list[int]], r: int, c: int) -> None:
    if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] == -1: return
    grid[r][c] = -1 # mark visited
    for i in range(4):
        dfs(grid, r + dr[i], c + dc[i])
```

### Template C: 2D Prefix Sum Construction + Query
```python
# Construction
sum_grid = [[0] * (C + 1) for _ in range(R + 1)]
for r in range(1, R + 1):
    for c in range(1, C + 1):
        sum_grid[r][c] = matrix[r-1][c-1] + sum_grid[r-1][c] + sum_grid[r][c-1] - sum_grid[r-1][c-1]

# Query from (r1, c1) to (r2, c2)
def query(r1: int, c1: int, r2: int, c2: int) -> int:
    return sum_grid[r2+1][c2+1] - sum_grid[r1][c2+1] - sum_grid[r2+1][c1] + sum_grid[r1][c1]
```

**Understanding the Construction — Worked Example.** Given a 3×3 matrix, we build a 4×4 prefix sum array `S` padded with a zero row and zero column. Each cell `S[r][c]` stores the sum of all original elements from `(0,0)` to `(r-1, c-1)`.

Original Matrix A:

|     | c0  | c1  | c2  |
|-----|-----|-----|-----|
| r0  |  1  |  2  |  3  |
| r1  |  4  |  5  |  6  |
| r2  |  7  |  8  |  9  |

Prefix Sum Array S (row 0 and column 0 are all zeros):

|     | c0  | c1  | c2  | c3  |
|-----|-----|-----|-----|-----|
| r0  |  0  |  0  |  0  |  0  |
| r1  |  0  |  1  |  3  |  6  |
| r2  |  0  |  5  | 12  | 21  |
| r3  |  0  | 12  | 27  | 45  |

**Cell-by-cell trace for S[2][2] = 12:**

```
S[r][c] = A[r-1][c-1] + S[r-1][c] + S[r][c-1] - S[r-1][c-1]
```

```
S[2][2] = A[1][1] (5) + S[1][2] (3) + S[2][1] (5) - S[1][1] (1) = 12
```

The two 5s come from different sources: `A[1][1] = 5` is the center cell of the original matrix, while `S[2][1] = 5` is the prefix sum of the first column (`1 + 4 = 5`). Verify: `S[2][2]` should equal `1 + 2 + 4 + 5 = 12` — the sum of all elements from `(0,0)` to `(1,1)`. ✓

**Sanity check**: `S[3][3] = 45` equals `1+2+3+4+5+6+7+8+9 = 45`. ✓

![2D Prefix Sum — Construction via Inclusion-Exclusion (Trace)](editions/python/chapters/11-matrix-grid-patterns/visuals/prefix_sum_construction.png){width=85%}

**Understanding the Query — Inclusion-Exclusion.** To find the sum of a sub-rectangle from `(r1, c1)` to `(r2, c2)`, we carve it out of the full prefix sum using four overlapping rectangles:

```
query(r1, c1, r2, c2) = S[r2+1][c2+1] - S[r1][c2+1] - S[r2+1][c1] + S[r1][c1]
```

**The `+1` rule**: `+1` means "include this boundary." The middle two terms are *crossed* — each keeps one dimension full and chops the other:

| Term | Row | Col | Covers |
|------|-----|-----|--------|
| `S[r2+1][c2+1]` | +1 (include r2) | +1 (include c2) | Full rectangle |
| `- S[r1][c2+1]` | raw (cut before r1) | +1 (include c2) | Rows above target |
| `- S[r2+1][c1]` | +1 (include r2) | raw (cut before c1) | Cols left of target |
| `+ S[r1][c1]` | raw | raw | Top-left overlap (subtracted twice, add back) |

**Worked query**: Sum of sub-rectangle `(1,1)` to `(2,2)` — cells `{5, 6, 8, 9}` = 28:

```
S[3][3] - S[1][3] - S[3][1] + S[1][1] = 45 - 6 - 12 + 1 = 28
```

![2D Prefix Sum — Query via Inclusion-Exclusion](editions/python/chapters/11-matrix-grid-patterns/visuals/prefix_sum_2d_query.png){width=85%}

## Solved Exemplar Problems

* * *
**1. Rotate Matrix 90° Clockwise**
**Specification:** You are given an $n \times n$ 2D matrix representing an image. Rotate the image by 90 degrees (clockwise) in-place.

**Example:** Input: `[[1,2],[3,4]]` -> Output: `[[3,1],[4,2]]`

**Pattern:** Transpose + Reverse Rows.

**Explanation:** Rotating 90 degrees clockwise is mathematically equivalent to transposing the matrix (swapping $i,j$ with $j,i$) and then reversing the elements of each row. This avoids needing complex 4-way coordinate swaps.

```python
def rotate(self, matrix: list[list[int]]) -> None:
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse each row
    for i in range(n):
        for j in range(n // 2):
            matrix[i][j], matrix[i][n - 1 - j] = matrix[i][n - 1 - j], matrix[i][j]
```
Time: $\mathcal{O}(N^2)$ | Space: $\mathcal{O}(1)$

* * *
**2. Spiral Matrix Traversal**
**Specification:** Given an $m \times n$ matrix, return all elements of the matrix in spiral order.

**Example:** Input: `[[1,2,3],[4,5,6],[7,8,9]]` -> Output: `[1,2,3,6,9,8,7,4,5]`

**Pattern:** Boundary Contraction.

**Explanation:** Maintain `top`, `bottom`, `left`, `right` pointers. Traverse the top row, increment `top`. Traverse right col, decrement `right`. Traverse bottom row (if `top <= bottom`), decrement `bottom`. Traverse left col (if `left <= right`), increment `left`.

```python
def spiral_order(self, matrix: list[list[int]]) -> list[int]:
    res = []
    t, b, l, r = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    while t <= b and l <= r:
        for j in range(l, r + 1): res.append(matrix[t][j]) # Top
        t += 1
        for i in range(t, b + 1): res.append(matrix[i][r]) # Right
        r -= 1
        if t <= b:
            for j in range(r, l - 1, -1): res.append(matrix[b][j]) # Bottom
            b -= 1
        if l <= r:
            for i in range(b, t - 1, -1): res.append(matrix[i][l]) # Left
            l += 1
    return res
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

**Trace Walkthrough** (input: `3x3 matrix`):

| Step | Row | Col | Direction | Value | Action |
|:---:|:---:|:---:|:----------|:-----:|:-------|
| 1    | 0   | 0   | Right     | 1     | Add to result |
| 2    | 0   | 1   | Right     | 2     | Add to result |
| 3    | 0   | 2   | Right     | 3     | Add, contract top bound |
| 4    | 1   | 2   | Down      | 6     | Add to result |
| 5    | 2   | 2   | Down      | 9     | Add, contract right bound |
| 6    | 2   | 1   | Left      | 8     | Add to result |
| 7    | 2   | 0   | Left      | 7     | Add, contract bottom bound |
| 8    | 1   | 0   | Up        | 4     | Add, contract left bound |
| 9    | 1   | 1   | Right     | 5     | Add, contract top bound |

* * *
**3. Set Matrix Zeros**
**Specification:** Given an $m \times n$ integer matrix, if an element is 0, set its entire row and column to 0's in-place.

**Example:** Input: `[[1,1,1],[1,0,1],[1,1,1]]` -> Output: `[[1,1,1],[0,0,0],[1,1,1]]`

**Pattern:** In-Place State Encoding (using first row/col as markers).

**Explanation:** We use the first row and first column to store information about whether that row or column should be zeroed out. We need a separate variable for the first column to avoid overlapping state.

```python
def set_zeroes(self, matrix: list[list[int]]) -> None:
    m, n = len(matrix), len(matrix[0])
    first_col_zero = False
    
    # Mark zeros on first row/col
    for i in range(m):
        if matrix[i][0] == 0: first_col_zero = True
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
                
    # Zero out based on marks
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
                
    # Handle first row/col specifically
    if matrix[0][0] == 0:
        for j in range(n): matrix[0][j] = 0
    if first_col_zero:
        for i in range(m): matrix[i][0] = 0
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**4. Diagonal Matrix Traversal**
**Specification:** Given an $m \times n$ matrix, return an array of all its elements arranged in a diagonal zigzag order.

**Example:** Input: `[[1,2,3],[4,5,6],[7,8,9]]` -> Output: `[1,2,4,7,5,3,6,8,9]`

**Pattern:** Zigzag Direction Switching.

**Explanation:** In a diagonal traversal, the sum of indices `(i+j)` is constant for each diagonal. For even sums, we move Up-Right. For odd sums, we move Down-Left. Boundary conditions handle when we hit the edges.

```python
def find_diagonal_order(self, mat: list[list[int]]) -> list[int]:
    m, n = len(mat), len(mat[0])
    res = [0] * (m * n)
    r, c = 0, 0
    for i in range(m * n):
        res[i] = mat[r][c]
        if (r + c) % 2 == 0: # Moving Up-Right
            if c == n - 1: r += 1
            elif r == 0: c += 1
            else: r -= 1; c += 1
        else: # Moving Down-Left
            if r == m - 1: c += 1
            elif c == 0: r += 1
            else: r += 1; c -= 1
    return res
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**5. Matrix Reshape Validation**
**Specification:** In MATLAB, `reshape` changes an $m \times n$ matrix into an $r \times c$ matrix. If impossible, return original. Otherwise, fill row by row.

**Example:** Input: `mat = [[1,2],[3,4]], r = 1, c = 4` -> Output: `[[1,2,3,4]]`

**Pattern:** Row-Major Index Mapping.

**Explanation:** A 2D matrix can be flattened logically. The 1D index `k` maps to 2D coordinates `(k / cols, k % cols)`. We map the original matrix into the new shape using a single counter `k`.

```python
def matrix_reshape(self, mat: list[list[int]], r: int, c: int) -> list[list[int]]:
    m, n = len(mat), len(mat[0])
    if m * n != r * c: return mat # Invalid shape
    
    res = [[0] * c for _ in range(r)]
    for i in range(m * n):
        res[i // c][i % c] = mat[i // n][i % n]
    return res
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(R \times C)$

* * *
**6. Rotate Matrix 90° Counter-Clockwise**
**Specification:** Rotate an $N \times N$ matrix by 90 degrees counter-clockwise in place.

**Example:** Input: `[[1,2],[3,4]]` -> Output: `[[2,4],[1,3]]`

**Pattern:** Transpose + Reverse Columns.

**Explanation:** Counter-clockwise rotation is similar to clockwise. We transpose first, then reverse the columns (top to bottom swap) instead of rows.

```python
def rotate_counter(self, matrix: list[list[int]]) -> None:
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse each column
    for j in range(n):
        for i in range(n // 2):
            matrix[i][j], matrix[n - 1 - i][j] = matrix[n - 1 - i][j], matrix[i][j]
```
Time: $\mathcal{O}(N^2)$ | Space: $\mathcal{O}(1)$

* * *
**7. Search in Row-Column Sorted Matrix**
**Specification:** Write an efficient algorithm that searches for a value in an $m \times n$ matrix where each row and column is sorted in ascending order.

**Example:** Input: `mat = [[1,4],[2,5]], target = 2` -> Output: `true`

**Pattern:** Staircase Search from Top-Right.

**Explanation:** Start at the top-right corner. If target is smaller than the current value, it can't be in this column (move left). If target is larger, it can't be in this row (move down).

```python
def search_matrix(self, matrix: list[list[int]], target: int) -> bool:
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        if matrix[r][c] == target: return True
        elif matrix[r][c] > target: c -= 1
        else: r += 1
    return False
```
Time: $\mathcal{O}(M + N)$ | Space: $\mathcal{O}(1)$

* * *
**8. Game of Life**
**Specification:** Given a board of 0s (dead) and 1s (live), compute the next state based on Conway's Game of Life rules simultaneously.

**Example:** Rules: <2 neighbors dies, 2-3 lives, >3 dies. Dead with 3 lives.

**Pattern:** In-Place State Encoding.

**Explanation:** To update in-place without a copy, encode transitions. Let 2 mean "was dead, now live", and -1 mean "was live, now dead". When counting neighbors, check if `abs(val) == 1`. After updating all, decode the states.

```python
def game_of_life(self, board: list[list[int]]) -> None:
    m, n = len(board), len(board[0])
    for r in range(m):
        for c in range(n):
            live = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0: continue
                    nr, nc = r + i, c + j
                    if 0 <= nr < m and 0 <= nc < n and abs(board[nr][nc]) == 1: live += 1
            if board[r][c] == 1 and (live < 2 or live > 3): board[r][c] = -1
            if board[r][c] == 0 and live == 3: board[r][c] = 2
            
    for r in range(m):
        for c in range(n):
            if board[r][c] > 0: board[r][c] = 1
            else: board[r][c] = 0
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**9. Toeplitz Matrix Verification**
**Specification:** Given an $m \times n$ matrix, return true if the matrix is Toeplitz. A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same elements.

**Example:** Input: `[[1,2],[3,1]]` -> Output: `true`

**Pattern:** Matrix Traversal Property.

**Explanation:** Simply check every cell `matrix[i][j]` against its top-left neighbor `matrix[i-1][j-1]`. If they mismatch, return false.

```python
def is_toeplitz_matrix(self, matrix: list[list[int]]) -> bool:
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[i][j] != matrix[i-1][j-1]:
                return False
    return True
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**10. Spiral Matrix Construction**
**Specification:** Given a positive integer $n$, generate an $n \times n$ matrix filled with elements from 1 to $n^2$ in spiral order.

**Example:** Input: `n = 3` -> Output: `[[1,2,3],[8,9,4],[7,6,5]]`

**Pattern:** Boundary Contraction (Write mode).

**Explanation:** Similar to spiral traversal, but instead of reading, we write an incrementing counter `val++` into the boundaries, contracting inwards until we fill $n^2$ elements.

```python
def generate_matrix(self, n: int) -> list[list[int]]:
    mat = [[0] * n for _ in range(n)]
    t, b, l, r = 0, n - 1, 0, n - 1
    val = 1
    while t <= b and l <= r:
        for j in range(l, r + 1):
            mat[t][j] = val
            val += 1
        t += 1
        for i in range(t, b + 1):
            mat[i][r] = val
            val += 1
        r -= 1
        if t <= b:
            for j in range(r, l - 1, -1):
                mat[b][j] = val
                val += 1
            b -= 1
        if l <= r:
            for i in range(b, t - 1, -1):
                mat[i][l] = val
                val += 1
            l += 1
    return mat
```
Time: $\mathcal{O}(N^2)$ | Space: $\mathcal{O}(N^2)$

* * *
**11. Flood Fill**
**Specification:** An image is an $m \times n$ grid. Perform a flood fill starting from `(sr, sc)` replacing the connected old color with a `color`.

**Example:** Input: `img=[[1,1,1],[1,1,0],[1,0,1]], sr=1,sc=1, color=2` -> Output: `[[2,2,2],[2,2,0],[2,0,1]]`

**Pattern:** DFS Recursive 4-Directional.

**Explanation:** We check if the starting pixel is already the target color. If not, we recursively replace all adjacent cells of the original color with the new color using DFS.

```python
def flood_fill(self, image: list[list[int]], sr: int, sc: int, color: int) -> list[list[int]]:
    if image[sr][sc] != color:
        self._dfs(image, sr, sc, image[sr][sc], color)
    return image

def _dfs(self, img: list[list[int]], r: int, c: int, old_c: int, new_c: int) -> None:
    if r < 0 or r >= len(img) or c < 0 or c >= len(img[0]) or img[r][c] != old_c: return
    img[r][c] = new_c # mark and fill
    self._dfs(img, r-1, c, old_c, new_c)
    self._dfs(img, r+1, c, old_c, new_c)
    self._dfs(img, r, c-1, old_c, new_c)
    self._dfs(img, r, c+1, old_c, new_c)
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**12. Transpose Rectangular Matrix**
**Specification:** Given a 2D integer array matrix, return the transpose of matrix. Matrix may not be square.

**Example:** Input: `[[1,2,3],[4,5,6]]` -> Output: `[[1,4],[2,5],[3,6]]`

**Pattern:** Allocation + Row-Major Mapping.

**Explanation:** Since the matrix isn't square, we cannot transpose in place. We allocate a new matrix of size $C \times R$, and assign `ans[j][i] = matrix[i][j]`.

```python
def transpose(self, matrix: list[list[int]]) -> list[list[int]]:
    r = len(matrix)
    c = len(matrix[0])
    ans = [[0] * r for _ in range(c)]
    for i in range(r):
        for j in range(c):
            ans[j][i] = matrix[i][j]
    return ans
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**13. Valid Sudoku**
**Specification:** Determine if a $9 \times 9$ Sudoku board is valid. Only the filled cells need to be validated according to standard rules.

**Example:** Input: Standard Sudoku grid with duplicates in row 1 -> Output: `false`

**Pattern:** HashSet Encoding Trick.

**Explanation:** We iterate through the grid. For each cell, we encode its presence in its row, column, and block as unique integers to avoid slow string concatenations. If `HashSet.add()` returns false, a duplicate exists.

```python
def is_valid_sudoku(self, board: list[list[str]]) -> bool:
    seen = set()
    for i in range(9):
        for j in range(9):
            number = board[i][j]
            if number != '.':
                box_idx = (i // 3) * 3 + j // 3
                row_key = f"{number} in row {i}"
                col_key = f"{number} in col {j}"
                box_key = f"{number} in box {box_idx}"
                if row_key in seen or col_key in seen or box_key in seen:
                    return False
                seen.add(row_key)
                seen.add(col_key)
                seen.add(box_key)
    return True
```
Time: $\mathcal{O}(1)$ (fixed 9×9) | Space: $\mathcal{O}(1)$

**Trace Walkthrough** (input: `Sudoku with duplicate 5s in row 0`):

| Step | Row | Col | Value | Encoded Strings | Action |
|:---:|:---:|:---:|:-----:|:----------------|:-------|
| 1    | 0   | 0   | 5     | "5 in row 0", "5 in col 0", "5 in block 0-0" | Add to HashSet (Success) |
| 2    | 0   | 1   | 3     | "3 in row 0", "3 in col 1", "3 in block 0-0" | Add to HashSet (Success) |
| 3    | 0   | 4   | 5     | "5 in row 0", "5 in col 4", "5 in block 0-1" | Add to HashSet (Collision on "5 in row 0") -> Return false |

* * *
**14. Island Perimeter**
**Specification:** You are given row x col grid representing a map where 1 is land and 0 is water. Calculate the perimeter of the island.

**Example:** Input: `[[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]` -> Output: `16`

**Pattern:** Neighbor Subtraction.

**Explanation:** Each land cell adds 4 to the perimeter. For each land cell, we check its left and top neighbors. If they are also land, they share an edge, meaning we subtract 2 from the total perimeter (1 for each cell).

```python
def island_perimeter(self, grid: list[list[int]]) -> int:
    perimeter = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                perimeter += 4
                if i > 0 and grid[i - 1][j] == 1: perimeter -= 2
                if j > 0 and grid[i][j - 1] == 1: perimeter -= 2
    return perimeter
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**15. Maximum K×K Submatrix Sum**
**Specification:** Given an $M \times N$ matrix and integer $K$, find the max sum of a contiguous $K \times K$ submatrix.

**Example:** Input: `mat=[[1,2],[3,4]], K=1` -> Output: `4`

**Pattern:** 2D Prefix Sum.

**Explanation:** Construct a 2D prefix sum array. Then iterate through all possible bottom-right corners `(i,j)` of size $K \times K$, extracting the sum in $\mathcal{O}(1)$ time.

```python
def max_sum(self, mat: list[list[int]], k: int) -> int:
    m, n = len(mat), len(mat[0])
    pre = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            pre[i][j] = mat[i-1][j-1] + pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1]
            
    max_val = float('-inf')
    for i in range(k, m + 1):
        for j in range(k, n + 1):
            s = pre[i][j] - pre[i-k][j] - pre[i][j-k] + pre[i-k][j-k]
            max_val = max(max_val, s)
    return max_val
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

**Trace Walkthrough** (input: `mat=[[1,2,3],[4,5,6],[7,8,9]], K=2`):

| Step | Row | Col | Value | Action |
|:---:|:---:|:---:|:-----:|:-------|
| 1    | 2   | 2   | 12    | Query (2,2) with K=2: 12 - 0 - 0 + 0 = 12 |
| 2    | 2   | 3   | 16    | Query (2,3) with K=2: 18 - 0 - 2 + 0 = 16 |
| 3    | 3   | 2   | 24    | Query (3,2) with K=2: 27 - 3 - 0 + 0 = 24 |
| 4    | 3   | 3   | 28    | Query (3,3) with K=2: 45 - 6 - 12 + 1 = 28 (Max) |

* * *
**16. Number of Islands**
**Specification:** Given an $m \times n$ grid of '1's (land) and '0's (water), return the number of islands (connected components).

**Example:** Input: `[["1","1","0"],["0","0","1"]]` -> Output: `2`

**Pattern:** BFS/DFS Connected Components.

**Explanation:** Iterate over every cell. When a '1' is found, increment the island count, and launch a DFS/BFS to mark all connected '1's as '0' to avoid recounting.

```python
def num_islands(self, grid: list[list[str]]) -> int:
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                self._dfs(grid, i, j)
    return count

def _dfs(self, grid: list[list[str]], r: int, c: int) -> None:
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] == '0': return
    grid[r][c] = '0'
    self._dfs(grid, r+1, c); self._dfs(grid, r-1, c)
    self._dfs(grid, r, c+1); self._dfs(grid, r, c-1)
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**17. Flip and Invert Image**
**Specification:** Given an $n \times n$ binary matrix, flip the image horizontally, then invert it. Flipping means reversing the row. Inverting means changing 0 to 1 and 1 to 0.

**Example:** Input: `[[1,1,0]]` -> Output: `[[1,0,0]]`

**Pattern:** Two-Pointer XOR + Reverse.

**Explanation:** In a single pass per row, we can use two pointers `i` and `j`. We assign `row[i] = row[j] ^ 1` and `row[j] = temp ^ 1`. Note the middle element when length is odd.

```python
def flip_and_invert_image(self, image: list[list[int]]) -> list[list[int]]:
    for row in image:
        left, right = 0, len(row) - 1
        while left <= right:
            row[left], row[right] = row[right] ^ 1, row[left] ^ 1
            left += 1; right -= 1
    return image
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**18. Shift 2D Grid**
**Specification:** Given a 2D `grid` of size $m \times n$ and an integer `k`, shift the grid `k` times. Shifting means element at `(i,j)` moves to `(i, j+1)`, last column moves to next row, bottom-right moves to `(0,0)`.

**Example:** Input: `[[1,2],[3,4]], k=1` -> Output: `[[4,1],[2,3]]`

**Pattern:** Modular Index Arithmetic (1D Flattening).

**Explanation:** Map the grid to a 1D array conceptually of size $M \times N$. The new position of an element at index `i` is `(i + k) % (M * N)`. We can construct a new result grid based on this mapping.

```python
def shift_grid(self, grid: list[list[int]], k: int) -> list[list[int]]:
    m, n = len(grid), len(grid[0])
    total = m * n
    k %= total
    res = [[0] * n for _ in range(m)]
    
    for r in range(m):
        for c in range(n):
            new_1d = (r * n + c + k) % total
            res[new_1d // n][new_1d % n] = grid[r][c]
    return res
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**19. Word Search in Grid**
**Specification:** Given an $m \times n$ grid of characters and a `word`, return true if the word exists. The word can be constructed from letters of sequentially adjacent cells (horizontally or vertically).

**Example:** Input: `[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]`, word="ABCCED" -> Output: `true`

**Pattern:** DFS Backtracking.

**Explanation:** Iterate over all cells. If the first character matches, launch DFS. Temporarily mark cells (e.g., `#`) during recursion to prevent reuse, and restore them after the recursive call returns.

```python
def exist(self, board: list[list[str]], word: str) -> bool:
    for i in range(len(board)):
        for j in range(len(board[0])):
            if self._dfs(board, i, j, word, 0): return True
    return False

def _dfs(self, b: list[list[str]], r: int, c: int, word: str, idx: int) -> bool:
    if idx == len(word): return True
    if r < 0 or c < 0 or r >= len(b) or c >= len(b[0]) or b[r][c] != word[idx]: return False
    
    temp = b[r][c]
    b[r][c] = '#'
    found = (self._dfs(b, r+1, c, word, idx+1) or self._dfs(b, r-1, c, word, idx+1) or
             self._dfs(b, r, c+1, word, idx+1) or self._dfs(b, r, c-1, word, idx+1))
    b[r][c] = temp
    return found
```
Time: $\mathcal{O}(M \times N \times 4^L)$ | Space: $\mathcal{O}(L)$

* * *
**20. Determine If Matrix Can Be Obtained By Rotation**
**Specification:** Given two $n \times n$ binary matrices `mat` and `target`, return `true` if it is possible to make `mat` equal to `target` by rotating `mat` in 90-degree increments.

**Example:** Input: `mat = [[0,1],[1,0]], target = [[1,0],[0,1]]` -> Output: `true`

**Pattern:** Multiple Rotation Validation.

**Explanation:** A matrix can be rotated at most 3 times (90, 180, 270 degrees). We compare `mat` to `target` up to 4 times, rotating `mat` by 90 degrees each time.

```python
def find_rotation(self, mat: list[list[int]], target: list[list[int]]) -> bool:
    for k in range(4):
        if mat == target: return True
        self.rotate(mat)
    return False

def rotate(self, mat: list[list[int]]) -> None:
    n = len(mat)
    for i in range(n):
        for j in range(i + 1, n):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
    for i in range(n):
        for j in range(n // 2):
            mat[i][j], mat[i][n-1-j] = mat[i][n-1-j], mat[i][j]
```
Time: $\mathcal{O}(N^2)$ | Space: $\mathcal{O}(1)$

* * *
**21. Chess Board Cell Color**
**Specification:** Given two cell strings on a standard chessboard (e.g. `"A1"`, `"C3"`), determine if they are the same color.

**Example:** Input: `cell1 = "A1", cell2 = "C3"` -> Output: `true`

**Pattern:** Parity Check.

**Explanation:** Convert the column letter and row number to integers. The color of a cell `(x, y)` is uniquely determined by `(x + y) % 2`. Compare the parity.

```python
def solution(self, cell1: str, cell2: str) -> bool:
    sum1 = (ord(cell1[0]) - ord('A')) + (ord(cell1[1]) - ord('1'))
    sum2 = (ord(cell2[0]) - ord('A')) + (ord(cell2[1]) - ord('1'))
    return (sum1 % 2) == (sum2 % 2)
```
Time: $\mathcal{O}(1)$ | Space: $\mathcal{O}(1)$

* * *
**22. Minesweeper Click Reveal**
**Specification:** Given a Minesweeper board and a click coordinate, if it's a mine 'M', turn to 'X'. If empty 'E' with no adjacent mines, turn to 'B' and recursively reveal neighbors. If empty with mines, turn to digit.

**Example:** Input: `board=[['E','E'],['E','M']], click=[0,0]` -> Output: `[['1','1'],['1','M']]`

**Pattern:** BFS/DFS Simulation with 8 Directions.

**Explanation:** Count adjacent mines (8 directions). If > 0, set to digit. If == 0, set to 'B' and DFS to 8 adjacent 'E' neighbors.

```python
def update_board(self, board: list[list[str]], click: list[int]) -> list[list[str]]:
    r, c = click[0], click[1]
    if board[r][c] == 'M':
        board[r][c] = 'X'
        return board
    self._dfs(board, r, c)
    return board

def _dfs(self, b: list[list[str]], r: int, c: int) -> None:
    if r < 0 or c < 0 or r >= len(b) or c >= len(b[0]) or b[r][c] != 'E': return
    mines = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            nr, nc = r + i, c + j
            if 0 <= nr < len(b) and 0 <= nc < len(b[0]) and b[nr][nc] == 'M':
                mines += 1
                
    if mines > 0:
        b[r][c] = str(mines)
    else:
        b[r][c] = 'B'
        for i in range(-1, 2):
            for j in range(-1, 2):
                self._dfs(b, r+i, c+j)
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**23. Battleship Placement Validation**
**Specification:** Given an $m \times n$ matrix where 'X' are ships and '.' are water. Count valid battleships. They can only be placed horizontally or vertically. Ships are separated by at least one cell.

**Example:** Input: `[["X",".",".","X"],[".",".",".","X"]]` -> Output: `2`

**Pattern:** Top-Left Identifier Traversal.

**Explanation:** Instead of a full DFS, just count the "top-left" cell of every battleship. A cell is a top-left if it is 'X' and has no 'X' above or to the left of it.

```python
def count_battleships(self, board: list[list[str]]) -> int:
    count = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'X':
                if i > 0 and board[i-1][j] == 'X': continue
                if j > 0 and board[i][j-1] == 'X': continue
                count += 1
    return count
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**24. Box Blur**
**Specification:** Apply a box blur algorithm to an image. Each pixel in the blurred image is the average of a $3 \times 3$ block centered at that pixel (rounded down).

**Example:** Input: $3 \times 3$ matrix. Output: $1 \times 1$ matrix with average.

**Pattern:** Sliding Window Matrix Accumulation.

**Explanation:** The output matrix size is $(M-2) \times (N-2)$. We iterate over these valid centers and compute the sum of the $3 \times 3$ area.

```python
def box_blur(self, image: list[list[int]]) -> list[list[int]]:
    m, n = len(image), len(image[0])
    res = [[0] * (n - 2) for _ in range(m - 2)]
    
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            s = sum(image[i + di][j + dj] for di in range(-1, 2) for dj in range(-1, 2))
            res[i-1][j-1] = s // 9
    return res
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**25. Zigzag String Conversion**
**Specification:** The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows. Read line by line to return the result.

**Example:** Input: `s = "PAYPALISHIRING", numRows = 3` -> Output: `"PAHNAPLSIIGYIR"`

**Pattern:** Simulation with Direction Vector.

**Explanation:** Maintain a `row` index and a `direction`. Add characters to `StringBuilder[]` corresponding to each row. When hitting top or bottom row, reverse direction.

```python
def convert(self, s: str, num_rows: int) -> str:
    if num_rows == 1: return s
    rows = ["" for _ in range(min(num_rows, len(s)))]
    
    cur_row = 0
    going_down = False
    
    for c in s:
        rows[cur_row] += c
        if cur_row == 0 or cur_row == num_rows - 1:
            going_down = not going_down
        cur_row += 1 if going_down else -1
        
    return "".join(rows)
```
Time: $\mathcal{O}(N)$ | Space: $\mathcal{O}(N)$

* * *
**26. Simulate Robot Commands on Grid**
**Specification:** A robot is on a $(0,0)$ facing North. It receives commands: -2 (turn left), -1 (turn right), 1..9 (move forward). There are obstacles. Find max distance squared from origin.

**Example:** Input: `commands = [4,-1,3], obstacles = []` -> Output: `25`

**Pattern:** State Machine Simulation (Direction Matrix).

**Explanation:** Encode North, East, South, West using `dx` and `dy`. Turn right is `dir = (dir + 1) % 4`. Move step by step checking against an obstacle `HashSet`.

```python
def robot_sim(self, commands: list[int], obstacles: list[list[int]]) -> int:
    dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
    obs = set((o[0], o[1]) for o in obstacles)
    
    x = y = dir_idx = max_dist = 0
    for cmd in commands:
        if cmd == -2: dir_idx = (dir_idx + 3) % 4
        elif cmd == -1: dir_idx = (dir_idx + 1) % 4
        else:
            for k in range(cmd):
                nx, ny = x + dx[dir_idx], y + dy[dir_idx]
                if (nx, ny) in obs: break
                x, y = nx, ny
                max_dist = max(max_dist, x*x + y*y)
    return max_dist
```
Time: $\mathcal{O}(C + O)$ | Space: $\mathcal{O}(O)$

* * *
**27. Matrix Water Flow (Pacific Atlantic)**
**Specification:** Grid representing island heights. Pacific touches left/top, Atlantic touches right/bottom. Find coordinates where water can flow to BOTH oceans (must go to equal or lower height).

**Example:** Input: `[[1,2],[3,1]]` -> Output: `[[0,1],[1,0]]`

**Pattern:** Reverse Multi-Source DFS.

**Explanation:** Instead of going downhill from every cell, go UPHILL from the ocean borders to mark reachable cells. Intersection of Pacific-reachable and Atlantic-reachable is the answer.

```python
def pacific_atlantic(self, heights: list[list[int]]) -> list[list[int]]:
    m, n = len(heights), len(heights[0])
    pac, atl = [[False] * n for _ in range(m)], [[False] * n for _ in range(m)]
    
    for i in range(m):
        self._dfs_pa(heights, pac, i, 0)
        self._dfs_pa(heights, atl, i, n-1)
    for j in range(n):
        self._dfs_pa(heights, pac, 0, j)
        self._dfs_pa(heights, atl, m-1, j)
        
    res = []
    for i in range(m):
        for j in range(n):
            if pac[i][j] and atl[i][j]:
                res.append([i, j])
    return res

def _dfs_pa(self, h, v, r, c):
    v[r][c] = True
    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(h) and 0 <= nc < len(h[0]) and not v[nr][nc] and h[nr][nc] >= h[r][c]:
            self._dfs_pa(h, v, nr, nc)
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**28. Rotting Oranges**
**Specification:** 0=empty, 1=fresh orange, 2=rotten. Every minute, fresh oranges adjacent to rotten ones become rotten. Return min minutes to rot all, or -1.

**Example:** Input: `[[2,1,1],[1,1,0],[0,1,1]]` -> Output: `4`

**Pattern:** Multi-Source BFS.

**Explanation:** Add all initially rotten oranges to a queue. Use BFS level-by-level to rot adjacent oranges. Track minutes. Finally, check if any fresh oranges remain.

```python
def oranges_rotting(self, grid: list[list[int]]) -> int:
    from collections import deque
    q = deque()
    fresh = 0
    m, n = len(grid), len(grid[0])
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2: q.append((i, j))
            elif grid[i][j] == 1: fresh += 1
            
    if fresh == 0: return 0
    mins = 0
    
    while q:
        rotted = False
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    q.append((nr, nc))
                    rotted = True
        if rotted: mins += 1
        
    return mins if fresh == 0 else -1
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

**Trace Walkthrough** (input: `[[2,1,1],[1,1,0],[0,1,1]]`):

| Step | Row | Col | Minute | Value | Action |
|:---:|:---:|:---:|:------:|:-----:|:-------|
| 1    | 0   | 0   | 0      | 2     | Initial rotten, enqueue |
| 2    | 0   | 1   | 1      | 1->2  | Rot right neighbor, enqueue |
| 3    | 1   | 0   | 1      | 1->2  | Rot bottom neighbor, enqueue |
| 4    | 0   | 2   | 2      | 1->2  | Rot right neighbor, enqueue |
| 5    | 1   | 1   | 2      | 1->2  | Rot bottom neighbor, enqueue |
| 6    | 2   | 1   | 3      | 1->2  | Rot bottom neighbor, enqueue |
| 7    | 2   | 2   | 4      | 1->2  | Rot right neighbor, enqueue |

* * *
**29. Surrounded Regions**
**Specification:** Given a grid of 'X' and 'O', capture all regions surrounded by 'X' by flipping 'O' to 'X'. A region is surrounded if no 'O' is on the border.

**Example:** Input: `[['X','X','X'],['X','O','X'],['X','X','X']]` -> Output: all 'X'.

**Pattern:** Border DFS.

**Explanation:** Any 'O' connected to a border 'O' cannot be captured. DFS from all border 'O's and mark them as safe ('#'). Flip all remaining 'O' to 'X', then revert '#' to 'O'.

```python
def solve(self, board: list[list[str]]) -> None:
    m, n = len(board), len(board[0])
    for i in range(m):
        self._dfs_s(board, i, 0)
        self._dfs_s(board, i, n-1)
    for j in range(n):
        self._dfs_s(board, 0, j)
        self._dfs_s(board, m-1, j)
        
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O': board[i][j] = 'X'
            elif board[i][j] == '#': board[i][j] = 'O'

def _dfs_s(self, b: list[list[str]], r: int, c: int) -> None:
    if r < 0 or r >= len(b) or c < 0 or c >= len(b[0]) or b[r][c] != 'O': return
    b[r][c] = '#'
    self._dfs_s(b, r+1, c); self._dfs_s(b, r-1, c)
    self._dfs_s(b, r, c+1); self._dfs_s(b, r, c-1)
```
Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**30. Path with Minimum Effort**
**Specification:** You are a hiker traversing an $m \times n$ matrix of heights. Effort is the maximum absolute difference in heights between two consecutive cells. Return min effort to go $(0,0)$ to $(m-1,n-1)$.

**Example:** Input: `[[1,2,2],[3,8,2],[5,3,5]]` -> Output: `2`

**Pattern:** Binary Search + BFS.

**Explanation:** We can binary search the answer range [0, 10^6]. For a chosen effort limit `K`, use BFS. If BFS reaches the end using only edges $\le K$, then `K` is possible, so search lower. Else, search higher.

```python
def minimum_effort_path(self, heights: list[list[int]]) -> int:
    left, right, ans = 0, 1000000, 1000000
    while left <= right:
        mid = (left + right) // 2
        if self._can_reach(heights, mid):
            ans = mid
            right = mid - 1
        else:
            left = mid + 1
    return ans

def _can_reach(self, h: list[list[int]], limit: int) -> bool:
    from collections import deque
    m, n = len(h), len(h[0])
    vis = [[False] * n for _ in range(m)]
    q = deque([(0, 0)])
    vis[0][0] = True
    
    while q:
        r, c = q.popleft()
        if r == m - 1 and c == n - 1: return True
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and not vis[nr][nc]:
                if abs(h[nr][nc] - h[r][c]) <= limit:
                    vis[nr][nc] = True
                    q.append((nr, nc))
    return False
```
Time: $\mathcal{O}(M \times N \times \log(\text{MaxH}))$ | Space: $\mathcal{O}(M \times N)$

## Practice Problem Bank

**1. Snake Traversal Verification**
   **Specification:** Given an $N \times N$ matrix and a 1D array representing a path, verify if the array follows a strict snake-like (zigzag) traversal of the matrix row by row.

**Example:** Input: `[[1,2],[4,3]]`, Path: `[1,2,3,4]`. Output: `true`.
   *Constraints*: $N \le 100$. Path length equals $N^2$.
   **Strategic Hint:** Use zigzag string conversion pattern. Flip the column iteration direction based on `row % 2`.

**2. Diagonal Submatrix Sums**
   **Specification:** Given a square matrix, compute the sum of the primary diagonal and secondary diagonal. If they intersect at a center element, do not double-count the center.

**Example:** Input: `[[1,2,3],[4,5,6],[7,8,9]]`. Output: `25`.
   *Constraints*: $N \le 500$.
   **Strategic Hint:** Only one loop `i` from $0$ to $N-1$ is needed. Primary is `(i, i)`, secondary is `(i, N-1-i)`.

**3. Check Matrix Symmetries**
   **Specification:** Return true if a binary matrix is symmetrically identical horizontally, vertically, and diagonally (both diagonals).

**Example:** Input: `[[1,0,1],[0,1,0],[1,0,1]]`. Output: `true`.
   *Constraints*: $N \le 100$.
   **Strategic Hint:** Check `mat[i][j]` against `mat[N-1-i][j]`, `mat[i][N-1-j]`, `mat[j][i]`.

**4. K-Rotations of Matrix**
   **Specification:** Given an $N \times N$ matrix and integer $K$, return the matrix rotated clockwise $K$ times by 90 degrees.

**Example:** Input: `[[1,2],[3,4]], K = 5`. Output: `[[3,1],[4,2]]`.
   *Constraints*: $0 \le K \le 10^9$.
   **Strategic Hint:** Rotate in-place. $K \% 4$ gives the true number of rotations needed.

**5. Local Minima Grid Search**
   **Specification:** A local minimum in a matrix is strictly less than its up to 4 neighbors. Find any local minimum's coordinates and return it.

**Example:** Input: `[[9,8,7],[6,1,2]]`. Output: `[1,1]`.
   *Constraints*: $M, N \le 1000$. All elements unique.
   **Strategic Hint:** Use DFS or a greedy walk. Always move to a strictly smaller neighbor until trapped.

**6. Matrix Block Sum (K-Radius)**
   **Specification:** Return a matrix `answer` where `answer[i][j]` is the sum of all elements `mat[r][c]` for $i - K \le r \le i + K, j - K \le c \le j + K$.

**Example:** Input: `[[1,2,3],[4,5,6],[7,8,9]], K=1`. Output: `[[12,21,16],...]`.
   *Constraints*: Matrix size up to $100 \times 100$.
   **Strategic Hint:** Use Template C (2D Prefix Sum) to answer each cell's block sum in $\mathcal{O}(1)$.

**7. Count Submatrices with All Ones**
   **Specification:** Given a binary matrix, count how many rectangular submatrices consist entirely of 1s.

**Example:** Input: `[[1,1],[1,1]]`. Output: `9` (four 1x1, two 1x2, two 2x1, one 2x2).
   *Constraints*: $M, N \le 150$.
   **Strategic Hint:** For each cell, count contiguous 1s on the left, then scan upwards to form rectangles.

**8. Sparse Matrix Multiplication**
   **Specification:** Multiply two sparse matrices $A$ and $B$. Return the result matrix.

**Example:** Input: $A = [[1,0],[0,1]]$, $B = [[2,0],[0,2]]$. Output: `[[2,0],[0,2]]`.
   *Constraints*: Matrices up to $100 \times 100$.
   **Strategic Hint:** Only multiply and accumulate `A[i][k] * B[k][j]` if `A[i][k]` is non-zero.

**9. Maximum Path Sum in Grid**
   **Specification:** Given an $M \times N$ grid, find the path from top-left to bottom-right that minimizes the sum of its values. You can only move right or down.

**Example:** Input: `[[1,3,1],[1,5,1],[4,2,1]]`. Output: `7`.
   *Constraints*: Contains positive integers.
   **Strategic Hint:** This is DP but simulates grid walks. State transition: `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`.

**10. Robot Bounded in Circle**
    **Specification:** A robot follows a string of instructions ("G", "L", "R"). After executing the instructions infinitely, does it stay in a bounded circle?

**Example:** Input: `"GGLLGG"`. Output: `true`.
    *Constraints*: String length $\le 100$.
    **Strategic Hint:** State Machine Simulation. If after one cycle the robot is at $(0,0)$ OR not facing North, it is bounded.

**11. Grid Game (Two Robots)**
    **Specification:** A $2 \times N$ grid of points. Robot 1 goes $(0,0) \to (1,N-1)$ setting visited cells to 0. Robot 2 does the same, trying to maximize its points. Robot 1 plays optimally to MINIMIZE Robot 2's points. Return Robot 2's score.

**Example:** Input: `[[2,5,4],[1,5,1]]`. Output: `4`.
    *Constraints*: $N \le 5 \times 10^4$.
    **Strategic Hint:** Robot 1 only has 1 turn to drop down. Use Prefix and Suffix arrays to simulate the remaining paths for Robot 2.

**12. As Far from Land as Possible**
    **Specification:** Grid of 0s (water) and 1s (land). Find a water cell such that its distance to the nearest land is maximized. Return this distance.

**Example:** Input: `[[1,0,1],[0,0,0],[1,0,1]]`. Output: `2`.
    *Constraints*: $M, N \le 100$.
    **Strategic Hint:** Multi-Source BFS. Add all 1s to queue, then BFS outwards. The last layer reached is the answer.

**13. Spiral Matrix III**
    **Specification:** Start at `(rStart, cStart)` in an $R \times C$ grid facing East. Walk in a spiral shape. Return coordinates of all cells visited in the grid.

**Example:** Input: `R=1, C=4, rStart=0, cStart=0`. Output: `[[0,0],[0,1],[0,2],[0,3]]`.
    *Constraints*: $1 \le R, C \le 100$.
    **Strategic Hint:** Step sequence is 1, 1, 2, 2, 3, 3... Simulate the walk, only adding valid in-bound coordinates to the result.

**14. Enclaves (Number of Closed Islands)**
    **Specification:** Binary matrix (0=land, 1=water). A closed island is completely surrounded by 1s (no land touches borders). Count them.

**Example:** Input: `[[1,1,1],[1,0,1],[1,1,1]]`. Output: `1`.
    *Constraints*: $N \le 100$.
    **Strategic Hint:** Border DFS. Eliminate all 0s connected to the grid borders. Then count remaining components of 0s.

**15. Ant on a Grid (Langton's Ant)**
    **Specification:** Simulate $K$ steps of an ant on an infinite white grid. White square -> turn right, flip to black, move forward. Black square -> turn left, flip to white, move.

**Example:** Input: `K = 10`. Output: Return bounds of modified grid.
    *Constraints*: $K \le 10^5$.
    **Strategic Hint:** State Machine Simulation using a `HashSet` to store coordinates of black squares. Track max/min X and Y.

**16. Shortest Bridge**
    **Specification:** An $N \times N$ matrix contains exactly two islands (1s). Find the shortest water bridge (0s to flip) to connect them.

**Example:** Input: `[[0,1],[1,0]]`. Output: `1`.
    *Constraints*: $N \le 100$.
    **Strategic Hint:** Find the first island with DFS and push all its cells to a queue. Then BFS from that queue to find the second island.

**17. Count Negative Numbers in Sorted Matrix**
    **Specification:** Matrix is sorted in decreasing order row-wise and column-wise. Count the negative numbers.

**Example:** Input: `[[4,3,-1],[2,1,-2]]`. Output: `2`.
    *Constraints*: $M, N \le 100$.
    **Strategic Hint:** Staircase search. Start at bottom-left or top-right and eliminate rows/columns.

**18. Build Matrix with Conditions**
    **Specification:** Build a $K \times K$ matrix with numbers 1 to $K$. Given row condition array and column condition array representing relative ordering (like $u$ must appear before $v$).

**Example:** Input: `K=3, rowConditions=[[1,2]], colConditions=[[2,1]]`. Output: valid placement grid.
    *Constraints*: $K \le 400$.
    **Strategic Hint:** Topological Sort. Apply Kahn's Algorithm independently for rows and columns to find the exact coordinate for each number.

**19. Determine Matrix is Magic Square**
    **Specification:** Given a $3 \times 3$ grid of integers, determine if it is a magic square (distinct numbers 1-9, rows/cols/diagonals sum to 15).

**Example:** Input: `[[4,3,8],[9,5,1],[2,7,6]]`. Output: `true`.
    *Constraints*: Grid is always $3 \times 3$.
    **Strategic Hint:** HashSet to check uniqueness (1-9), and 8 sums (3 rows, 3 cols, 2 diagonals) must equal 15.

**20. Coloring a Border**
    **Specification:** Given a grid, `(r, c)`, and `color`. Color the border of the connected component at `(r, c)`. A border cell touches a cell outside the component or the grid edge.

**Example:** Input: `[[1,1],[1,2]], (0,0), 3`. Output: `[[3,3],[3,2]]`.
    *Constraints*: $M, N \le 50$.
    **Strategic Hint:** DFS. If a cell has a neighbor of a different original color or is on the boundary, it's a border cell.

**21. Rotate Grid by K Steps**
    **Specification:** Rotate the layers of an $M \times N$ grid counter-clockwise $K$ times independently.

**Example:** Input: `[[1,2],[3,4]], K=1`. Output: `[[2,4],[1,3]]`.
    *Constraints*: Layers are concentric rectangles.
    **Strategic Hint:** Extract each spiral layer into a 1D array, perform 1D cyclic shift, and write it back using Boundary Contraction.

**22. Max Area of Island**
    **Specification:** Grid of 0s and 1s. Find the maximum area of a single connected component of 1s.

**Example:** Input: `[[1,1,0],[1,0,0]]`. Output: `3`.
    *Constraints*: $M, N \le 50$.
    **Strategic Hint:** DFS returning integer size. `return 1 + dfs(up) + dfs(down) + dfs(left) + dfs(right)`.

**23. Reshape the Matrix to 1D**
    **Specification:** Convert a $2D$ jagged array (rows of different lengths) into a strict $1D$ array in row-major order.

**Example:** Input: `[[1,2],[3],[4,5,6]]`. Output: `[1,2,3,4,5,6]`.
    *Constraints*: Total elements $\le 10^5$.
    **Strategic Hint:** Sequential iteration `for (int[] row : matrix) for (int val : row)`.

**24. Find the Winner of Tic-Tac-Toe**
    **Specification:** Given an array of moves (coordinates), determine the winner ("A", "B", "Draw", or "Pending").

**Example:** Input: `[[0,0],[1,1],[0,1],[0,2],[1,0],[2,0]]`. Output: `"B"`.
    *Constraints*: Standard $3 \times 3$ grid.
    **Strategic Hint:** Maintain arrays `rows[3]`, `cols[3]`, `diag`, `anti_diag`. Player A adds 1, B adds -1. Check for sum == 3 or -3.

**25. Maximal Square**
    **Specification:** Find the largest square submatrix containing only 1s and return its area.

**Example:** Input: `[[1,1],[1,1]]`. Output: `4`.
    *Constraints*: $M, N \le 300$.
    **Strategic Hint:** DP. `dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1` if cell is '1'.

**26. Bomb Enemy**
    **Specification:** Grid with '0' (empty), 'E' (enemy), 'W' (wall). Place a bomb at an empty cell to kill max enemies in its row/col until a wall is hit.

**Example:** Input: `[["0","E","0","0"],["E","0","W","E"]]`. Output: `3`.
    *Constraints*: $M, N \le 500$.
    **Strategic Hint:** State Encoding. Cache the row kill count and column kill count. Recalculate row hits only when crossing a wall.

**27. Check if Move is Legal (Othello/Reversi)**
    **Specification:** Given an $8 \times 8$ board, an `(r, c)` position, and `color`, return true if placing the stone forms a valid Reversi line.

**Example:** Input: Board state. Output: `true`.
    *Constraints*: Exactly $8 \times 8$.
    **Strategic Hint:** Raycasting simulation. Cast a ray in all 8 directions. It must pass through $\ge 1$ opponent stones before hitting a friendly stone.

**28. Minimum Knight Moves**
    **Specification:** Infinite chessboard. Starting at $(0,0)$, find min moves for a Knight to reach $(x,y)$.

**Example:** Input: `x = 2, y = 1`. Output: `1`.
    *Constraints*: $|x|, |y| \le 300$.
    **Strategic Hint:** BFS with 8 knight direction vectors. Use a `Set<String>` for visited coordinates. Leverage symmetry (absolute values of $x,y$) to bound search.

**29. Matrix Diagonal Sort**
    **Specification:** Sort each `i - j` diagonal of an $M \times N$ matrix in ascending order.

**Example:** Input: `[[3,3,1],[2,2,1],[1,1,1]]`. Output: `[[1,1,1],[1,2,2],[2,3,3]]`.
    *Constraints*: $M, N \le 100$.
    **Strategic Hint:** Use a `HashMap<Integer, PriorityQueue<Integer>>` where key is `i - j`. Add all elements, then write them back out.

**30. Game of Life 3D (Infinite Space)**
    **Specification:** Similar to Game of Life but in 3D. Find active cells after 6 cycles. Start with $2D$ plane in $3D$ space.

**Example:** Input: `[[0,1],[1,1]]`. Output: count of active cells.
    *Constraints*: Fixed 6 cycles.
    **Strategic Hint:** State Machine Simulation using a `HashSet` of string coordinates `"x,y,z"`. Only simulate neighbors of currently active cells.


# Medium-Hard-tier Mastery — Dynamic Sliding Windows, HashMap Frequency Signatures, and Prefix Sum Analytics

## Essential Terminology & Vocabulary

**Dynamic Sliding Window**
A technique where a window expands to the right to include elements and contracts from the left when a specific invariant or constraint is violated. It matters because it optimizes $\mathcal{O}(N^2)$ brute-force subarray checks into $\mathcal{O}(N)$ operations by avoiding redundant recalculations. Use when searching for the longest/shortest contiguous subarray satisfying a condition.

![Dynamic Sliding Window — Longest Substring Without Repeating Characters](editions/python/chapters/12-hashmaps-sliding-windows/visuals/sliding_window.png){width=85%}

**Fixed-Size Sliding Window vs Dynamic Sliding Window**

| Feature | Fixed-Size Window | Dynamic Sliding Window |
|:-----------------|:--------------------------------------------|:--------------------------------------------|
| **Window Size** | Constant (e.g., length K). | Variable (expands and contracts). |
| **Movement** | Move both left and right pointers together. | Move right continuously, move left only to fix invariants. |
| **Use Case** | Anagrams in a fixed window, max sum of K elements. | Longest substring with K distinct chars, minimum subarray sum. |

**HashMap Frequency Signature**
Creating a unique key for a group of items (like anagrams) based on their character frequencies rather than sorting. Usually represented as a mapped string of an `int[26]` array. This avoids the $\mathcal{O}(N \log N)$ sorting cost, providing an $\mathcal{O}(N)$ way to group items.

![HashMap Frequency Signature — Anagram Detection](editions/python/chapters/12-hashmaps-sliding-windows/visuals/hashmap_frequency.png){width=85%}

**Prefix Sum Array & Cumulative Matching**
An array where `pref[i]` stores the sum of elements from index $0$ to $i$. The trick `pref[j] - pref[i] = K` allows finding a subarray sum $K$ in $\mathcal{O}(1)$ time by rearranging to `pref[i] = pref[j] - K` and looking up previously seen prefix sums.

**Prefix Sum with HashMap**
A pattern combining prefix sums with a HashMap to count the occurrences of each prefix sum. This allows counting how many subarrays sum to a specific value $K$ in $\mathcal{O}(N)$ time.

**Two-Pointer for Sorted Arrays**
Using two pointers, usually starting at the beginning and end of a sorted array, that converge towards the middle. Used to find pairs summing to a target in $\mathcal{O}(N)$ time without extra space.

**Interval Merging and Insertion**
Sorting a collection of intervals by start time and iterating through them to combine overlapping ranges (where `current.start <= previous.end`). 

**Monotonic Stack/Queue**
A data structure that maintains elements in a strictly increasing or decreasing order. Useful for finding the "next greater element" or managing the maximum/minimum in a sliding window in $\mathcal{O}(N)$ time.

**Character Frequency Signature**
Using a fixed-size array (like `int[26]` for lowercase English letters) to count character occurrences. By converting this array to a string (or checking array equality), it acts as a canonical $\mathcal{O}(1)$ space key for anagrams.

**Modular Arithmetic in Prefix Sums**
Using the modulo operator with prefix sums. If `pref[i] % K == pref[j] % K`, then the subarray between $i$ and $j$ has a sum divisible by $K$.

### 'At Most K' to 'Exactly K' Conversion
This mathematical reduction calculates exact occurrences using cumulative bounds. It uses the formula `exactly(K) = atMost(K) - atMost(K-1)`.
Why it matters: This is the standard trick for counting subarrays with exactly K distinct elements.

### Index Negation Trick
This technique marks elements as 'seen' by negating the value at the corresponding index, such as `nums[abs(val)-1] = -nums[abs(val)-1]`. It only works for array values bounded within the range `[1, N]`.
Why it matters: It provides O(1) space duplicate or missing element detection without using extra data structures.

### Cyclic Sort / Index Placement
This sorting pattern places each value `v` precisely at its correct target index `nums[v-1]`. It continuously swaps elements until the current position holds the correct value.
Why it matters: It is the optimal strategy to find the first missing positive integer in O(N) time and O(1) space.

### Expand-Around-Center
This technique treats each index (and the space between indices) as a potential palindrome center. It then expands outwards as long as the mirrored characters match.
Why it matters: It is a simple and reliable O(N²) approach for the longest palindromic substring problem.

### Frequency Bucket Sort
This sorting alternative groups elements by their frequency into buckets ranging from `0` to `N`. You then scan these buckets in reverse order to collect the most frequent items.
Why it matters: It solves Top-K frequent elements problems in O(N) time without requiring a heap.

### Deferred Deletion / Lazy Invalidation
Instead of immediately removing items from a data structure, this technique marks entries as invalid. The actual cleanup happens later during traversal or retrieval.
Why it matters: It avoids ConcurrentModificationExceptions and heavily simplifies priority queue update patterns.

### Contribution Counting
Instead of iterating through all possible subarrays, this mathematical approach computes exactly how many subarrays a specific element contributes to. It aggregates the total across all individual element contributions.
Why it matters: It dramatically transforms O(N²) brute force summation logic into a highly optimal O(N) pass.

### Greedy Interval Scheduling
This algorithm sorts given intervals by their end times first. It then greedily picks the next non-overlapping interval to maximize total count.
Why it matters: It is a provably optimal approach for finding the maximum number of non-overlapping intervals.

* * *

## Reusable Code Templates

### Template A: Dynamic Sliding Window
```python
left = max_len = 0
for right in range(len(arr)):
    # 1. Add arr[right] to window state
    while False: # window state violates invariant
        # 2. Remove arr[left] from window state
        left += 1
    # 3. Update maxLen or minLen
    max_len = max(max_len, right - left + 1)
```

### Template B: Fixed-Size Sliding Window
```python
k, total_sum, max_val = 3, 0, 0
for i in range(len(arr)):
    total_sum += arr[i] # Add current element
    if i >= k - 1:
        max_val = max(max_val, total_sum) # Update result
        total_sum -= arr[i - (k - 1)]     # Remove leftmost element for next iteration
```

### Template C: Prefix Sum + HashMap Counter
```python
from collections import defaultdict
hash_map = defaultdict(int)
hash_map[0] = 1 # Base case for subarrays starting at index 0
total_sum = count = 0
for num in nums:
    total_sum += num
    if (total_sum - k) in hash_map:
        count += hash_map[total_sum - k]
    hash_map[total_sum] += 1
```

### Template D: HashMap Frequency Grouping
```python
from collections import defaultdict
hash_map = defaultdict(list)
for s in strs:
    count = [0] * 26
    for c in s: count[ord(c) - ord('a')] += 1
    key = str(count)
    hash_map[key].append(s)
```

* * *

## Solved Exemplar Problems

**1. Longest Substring Without Repeating Characters**
**Specification:** Given a string, find the length of the longest substring without repeating characters.

**Example:** `s = "abcabcbb"` -> Output: `3` ("abc")

**Pattern:** Dynamic Sliding Window + HashMap

**Explanation:** We expand the right pointer. If the character is in the set, we contract the left pointer until the duplicate is removed, ensuring the window always contains unique characters.
```python
def length_of_longest_substring(self, s: str) -> int:
    char_set = set()
    left = max_val = 0
    for right in range(len(s)):
        # Contract if duplicate found
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right]) # Add current char
        max_val = max(max_val, right - left + 1)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(min(N, M))
```

* * *

**2. Subarray Sum Equals K**
**Specification:** Find the total number of continuous subarrays whose sum equals to K.

**Example:** `nums = [1,1,1], k = 2` -> Output: `2`

**Pattern:** Prefix Sum + HashMap

**Explanation:** We maintain a running sum. If `sum - k` exists in our frequency map, it means there is a subarray ending at the current index that sums to K.
```python
def subarray_sum(self, nums: list[int], k: int) -> int:
    from collections import defaultdict
    hash_map = defaultdict(int)
    hash_map[0] = 1 # Base case
    total_sum = count = 0
    for num in nums:
        total_sum += num
        # Check if required prefix exists
        if (total_sum - k) in hash_map: count += hash_map[total_sum - k]
        hash_map[total_sum] += 1
    return count
# Time Complexity: O(N) | Space Complexity: O(N)
```

* * *

**3. Group Anagrams**
**Specification:** Group strings that are anagrams of each other.

**Example:** `["eat","tea","tan","ate","nat","bat"]` -> Output: `[["bat"],["nat","tan"],["ate","eat","tea"]]`

**Pattern:** HashMap Frequency Signature

**Explanation:** Generate a 26-element character count array for each string, convert it to a string key, and use it in a HashMap to group anagrams together.
```python
def group_anagrams(self, strs: list[str]) -> list[list[str]]:
    from collections import defaultdict
    hash_map = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s: count[ord(c) - ord('a')] += 1 # Build signature
        key = tuple(count)
        hash_map[key].append(s)
    return list(hash_map.values())
# Time Complexity: O(N * L) | Space Complexity: O(N * L)
```

* * *

**4. Find All Anagram Start Indices**
**Specification:** Find all start indices of p's anagrams in s.

**Example:** `s = "cbaebabacd", p = "abc"` -> Output: `[0, 6]`

**Pattern:** Fixed-Size Sliding Window + Frequency Array

**Explanation:** Use a window of size `p.length()`. Keep arrays of character frequencies for `p` and the current window in `s`. If they match, add the index.
```python
def find_anagrams(self, s: str, p: str) -> list[int]:
    res = []
    if len(s) < len(p): return res
    p_count, s_count = [0] * 26, [0] * 26
    for c in p: p_count[ord(c) - ord('a')] += 1
    for i in range(len(s)):
        s_count[ord(s[i]) - ord('a')] += 1
        if i >= len(p): s_count[ord(s[i - len(p)]) - ord('a')] -= 1 # Contract
        if p_count == s_count: res.append(i - len(p) + 1) # Match
    return res
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**5. Longest Substring with At Most K Distinct Characters**
**Specification:** Find the length of the longest substring with at most K distinct characters.

**Example:** `s = "eceba", k = 2` -> Output: `3` ("ece")

**Pattern:** Dynamic Sliding Window

**Explanation:** Use a HashMap to track character frequencies. When map size exceeds K, shrink window from left until size is K again.
```python
def length_of_longest_substring_k_distinct(self, s: str, k: int) -> int:
    from collections import defaultdict
    hash_map = defaultdict(int)
    left = max_val = 0
    for right in range(len(s)):
        c = s[right]
        hash_map[c] += 1
        while len(hash_map) > k: # Invariant broken
            left_char = s[left]
            left += 1
            hash_map[left_char] -= 1
            if hash_map[left_char] == 0: del hash_map[left_char]
        max_val = max(max_val, right - left + 1)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(K)
```

* * *

**6. Minimum Window Substring (Hard)**
*Note: This problem is universally classified as Hard on major platforms. While it uses the sliding window pattern from this chapter, its implementation complexity—managing two frequency maps, a `formed` counter, and a contraction loop—places it at the highest difficulty tier.*
**Specification:** Given strings s and t, find the minimum substring of s containing all characters in t.

**Example:** `s = "ADOBECODEBANC", t = "ABC"` -> Output: `"BANC"`

**Pattern:** Dynamic Sliding Window

**Explanation:** Track required characters in a map. Expand right until all required characters are in the window, then contract left to minimize the window.
```python
def min_window(self, s: str, t: str) -> str:
    char_map = [0] * 128
    for c in t: char_map[ord(c)] += 1
    left, count = 0, len(t)
    min_len, min_start = float('inf'), 0
    
    for right in range(len(s)):
        if char_map[ord(s[right])] > 0: count -= 1 # Found required char
        char_map[ord(s[right])] -= 1
        
        while count == 0: # All chars found
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left
            char_map[ord(s[left])] += 1
            if char_map[ord(s[left])] > 0: count += 1 # Removed required char
            left += 1
            
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**7. Group Shifted Strings**
**Specification:** Group strings that can be formed by shifting characters uniformly.

**Example:** `["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]` -> Output groups `["abc","bcd","xyz"]`, etc.

**Pattern:** Difference-Based Signature

**Explanation:** Calculate the relative distance between adjacent characters. Use this sequence of differences as the HashMap key.
```python
def group_strings(self, strings: list[str]) -> list[list[str]]:
    from collections import defaultdict
    hash_map = defaultdict(list)
    for s in strings:
        key = []
        for i in range(1, len(s)):
            diff = (ord(s[i]) - ord(s[i-1]) + 26) % 26 # Circular difference
            key.append(str(diff))
        hash_map[','.join(key)].append(s)
    return list(hash_map.values())
# Time Complexity: O(N * L) | Space Complexity: O(N * L)
```

* * *

**8. Contiguous Array Equal 0s and 1s**
**Specification:** Find the maximum length of a contiguous subarray with an equal number of 0s and 1s.

**Example:** `[0, 1, 0]` -> Output: `2`

**Pattern:** Prefix Sum (+1/-1 trick)

**Explanation:** Treat 0s as -1. If the running sum is seen again, it means the subarray between those two indices sums to 0, implying equal 0s and 1s.
```python
def find_max_length(self, nums: list[int]) -> int:
    hash_map = {0: -1}
    total_sum = max_val = 0
    for i, num in enumerate(nums):
        total_sum += -1 if num == 0 else 1 # Map 0 to -1
        if total_sum in hash_map:
            max_val = max(max_val, i - hash_map[total_sum])
        else:
            hash_map[total_sum] = i # Store first occurrence
    return max_val
# Time Complexity: O(N) | Space Complexity: O(N)
```

* * *

**9. Subarray Product Less Than K**
**Specification:** Count contiguous subarrays where the product is strictly less than K.

**Example:** `nums = [10,5,2,6], k = 100` -> Output: `8`

**Pattern:** Dynamic Sliding Window

**Explanation:** Maintain a running product. If product >= k, shrink from left. Number of valid subarrays ending at `right` is `right - left + 1`.
```python
def num_subarray_product_less_than_k(self, nums: list[int], k: int) -> int:
    if k <= 1: return 0
    prod, left, count = 1, 0, 0
    for right in range(len(nums)):
        prod *= nums[right]
        while prod >= k:
            prod //= nums[left]
            left += 1 # Shrink
        count += right - left + 1 # Add valid subarrays
    return count
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**10. Permutation in String**
**Specification:** Return true if s2 contains a permutation of s1.

**Example:** `s1 = "ab", s2 = "eidbaooo"` -> Output: `true`

**Pattern:** Fixed-Size Window Frequency Match

**Explanation:** Same logic as Anagram Start Indices. Maintain a window of size `s1.length()` and compare character counts.
```python
def check_inclusion(self, s1: str, s2: str) -> bool:
    if len(s1) > len(s2): return False
    s1_map, s2_map = [0] * 26, [0] * 26
    for c in s1: s1_map[ord(c) - ord('a')] += 1
    for i in range(len(s2)):
        s2_map[ord(s2[i]) - ord('a')] += 1
        if i >= len(s1): s2_map[ord(s2[i - len(s1)]) - ord('a')] -= 1
        if s1_map == s2_map: return True
    return False
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**11. Maximum Erasure Value**
**Specification:** Find the maximum score (sum) from a subarray of unique elements.

**Example:** `nums = [4,2,4,5,6]` -> Output: `17`

**Pattern:** Dynamic Sliding Window + HashSet

**Explanation:** Use a set to track uniqueness. Expand right, add to sum. If duplicate found, shrink from left, subtracting from sum until unique.
```python
def maximum_unique_subarray(self, nums: list[int]) -> int:
    char_set = set()
    total_sum = max_val = left = 0
    for right in range(len(nums)):
        while nums[right] in char_set:
            char_set.remove(nums[left])
            total_sum -= nums[left] # Remove duplicate
            left += 1
        char_set.add(nums[right])
        total_sum += nums[right]
        max_val = max(max_val, total_sum)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(N)
```

* * *

**12. Longest Repeating Character Replacement**
**Specification:** Longest substring of same letters after replacing at most k chars.

**Example:** `s = "AABABBA", k = 1` -> Output: `4`

**Pattern:** Window with Max Frequency Tracking

**Explanation:** If `window size - max_freq_char_count > k`, we have too many differing chars, so we shrink the window.
```python
def character_replacement(self, s: str, k: int) -> int:
    count = [0] * 26
    max_count = left = max_len = 0
    for right in range(len(s)):
        idx = ord(s[right]) - ord('A')
        count[idx] += 1
        max_count = max(max_count, count[idx])
        if right - left + 1 - max_count > k: # Invalid window
            count[ord(s[left]) - ord('A')] -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**13. Fruit Into Baskets**
**Specification:** Max fruit collected with 2 baskets (equivalent to max substring with <= 2 distinct characters).

**Example:** `[1,2,3,2,2]` -> Output: `4`

**Pattern:** Dynamic Sliding Window

**Explanation:** Keep a frequency map. When distinct fruit types exceed 2, increment left pointer to shrink.
```python
def total_fruit(self, fruits: list[int]) -> int:
    from collections import defaultdict
    count = defaultdict(int)
    left = max_val = 0
    for right in range(len(fruits)):
        count[fruits[right]] += 1
        while len(count) > 2:
            count[fruits[left]] -= 1
            if count[fruits[left]] == 0:
                del count[fruits[left]]
            left += 1
        max_val = max(max_val, right - left + 1)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**14. Continuous Subarray Sum Multiple of K**
**Specification:** Check if a subarray of length >= 2 has a sum multiple of K.

**Example:** `nums = [23,2,4,6,7], k = 6` -> Output: `true`

**Pattern:** Prefix Sum Modular Math

**Explanation:** If `pref[i] % k == pref[j] % k`, the sum between $i$ and $j$ is a multiple of $K$. Store remainder and its first seen index.
```python
def check_subarray_sum(self, nums: list[int], k: int) -> bool:
    hash_map = {0: -1}
    total_sum = 0
    for i, num in enumerate(nums):
        total_sum += num
        mod = total_sum if k == 0 else total_sum % k
        if mod in hash_map:
            if i - hash_map[mod] > 1: return True # Length >= 2
        else:
            hash_map[mod] = i
    return False
# Time Complexity: O(N) | Space Complexity: O(min(N, K))
```

* * *

**15. Max Consecutive Ones III**
**Specification:** Longest contiguous 1s after flipping at most K zeros.

**Example:** `nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2` -> Output: `6`

**Pattern:** Window with Zero-Flip Budget

**Explanation:** Expand window. If 0 encountered, decrease K. If K < 0, shrink window until a 0 is excluded.
```python
def longest_ones(self, nums: list[int], k: int) -> int:
    left = 0
    for right in range(len(nums)):
        if nums[right] == 0: k -= 1
        if k < 0: # Over budget
            if nums[left] == 0: k += 1
            left += 1
    return len(nums) - left # Trick to return max valid length seen
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**16. Find All Duplicates in Array**
**Specification:** Find elements appearing twice in an array containing integers in range [1, n].

**Example:** `[4,3,2,7,8,2,3,1]` -> Output: `[2,3]`

**Pattern:** Index Negation Trick

**Explanation:** Use the array itself as a hash table. Mark the number at index `abs(num) - 1` negative. If it's already negative, it's a duplicate.
```python
def find_duplicates(self, nums: list[int]) -> list[int]:
    res = []
    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0: res.append(abs(num)) # Found duplicate
        else: nums[idx] = -nums[idx] # Mark seen
    return res
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**17. Task Scheduler CPU Units**
**Specification:** Minimum CPU intervals to finish tasks given a cooldown of `n` between identical tasks.

**Example:** `tasks = ["A","A","A","B","B","B"], n = 2` -> Output: `8`

**Pattern:** Frequency Math

**Explanation:** Calculate idle slots based on the most frequent task. `maxIdle = (maxFreq - 1) * n`. Fill slots with other tasks.
```python
def least_interval(self, tasks: list[str], n: int) -> int:
    count = [0] * 26
    max_val = max_count = 0
    for c in tasks:
        idx = ord(c) - ord('A')
        count[idx] += 1
        if count[idx] == max_val:
            max_count += 1
        elif count[idx] > max_val:
            max_val = count[idx]
            max_count = 1
            
    empty_slots = (max_val - 1) * (n - (max_count - 1))
    available_tasks = len(tasks) - max_val * max_count
    idles = max(0, empty_slots - available_tasks)
    return len(tasks) + idles
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**18. Insert & Merge Overlapping Intervals**
**Specification:** Insert a new interval into a sorted list and merge if necessary.

**Example:** `[[1,3],[6,9]], new = [2,5]` -> Output: `[[1,5],[6,9]]`

**Pattern:** Interval Merging

**Explanation:** Three phases: Add all before new, merge overlapping with new, add all after new.
```python
def insert(self, intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:
    res = []
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < new_interval[0]:
        res.append(intervals[i]) # Before
        i += 1
    while i < n and intervals[i][0] <= new_interval[1]: # Merge
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    res.append(new_interval)
    while i < n:
        res.append(intervals[i]) # After
        i += 1
    return res
# Time Complexity: O(N) | Space Complexity: O(N)
```

* * *

**19. Top K Frequent Elements**
**Specification:** Return the K most frequent elements.

**Example:** `nums = [1,1,1,2,2,3], k = 2` -> Output: `[1,2]`

**Pattern:** HashMap + Min-Heap

**Explanation:** Count frequencies in a map, then keep a min-heap of size K based on frequencies.
```python
def top_k_frequent(self, nums: list[int], k: int) -> list[int]:
    from collections import Counter
    import heapq
    
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
# Time Complexity: O(N log K) | Space Complexity: O(N)
```

* * *

**20. First Missing Positive Integer**
**Specification:** Find the smallest missing positive integer in an unsorted array.

**Example:** `[3,4,-1,1]` -> Output: `2`

**Pattern:** Cyclic Sort (Index placement)

**Explanation:** Place number `x` at index `x-1`. Then scan to find the first index that doesn't have `i+1`.
```python
def first_missing_positive(self, nums: list[int]) -> int:
    i = 0
    while i < len(nums):
        # Swap to correct position if valid
        if 0 < nums[i] <= len(nums) and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        else:
            i += 1
            
    for i in range(len(nums)):
        if nums[i] != i + 1: return i + 1 # Missing
        
    return len(nums) + 1
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**21. Minimum Size Subarray Sum**
**Specification:** Min length of subarray with sum >= target.

**Example:** `target = 7, nums = [2,3,1,2,4,3]` -> Output: `2`

**Pattern:** Dynamic Window with Target Sum

**Explanation:** Keep expanding until sum >= target, then shrink to find minimum.
```python
def min_sub_array_len(self, target: int, nums: list[int]) -> int:
    left = total_sum = 0
    min_val = float('inf')
    for right in range(len(nums)):
        total_sum += nums[right]
        while total_sum >= target:
            min_val = min(min_val, right - left + 1)
            total_sum -= nums[left]
            left += 1
    return 0 if min_val == float('inf') else min_val
# Time Complexity: O(N) | Space Complexity: O(1)
```

* * *

**22. Substring with Concatenation of All Words**
**Specification:** Find starting indices of substrings that are a concatenation of all words in an array exactly once.

**Example:** `s = "barfoothefoobarman", words = ["foo","bar"]` -> Output: `[0, 9]`

**Pattern:** Fixed-Size Window with Inner HashMap

**Explanation:** Use a map for word counts. Slide a window of length `words.length * wordLen` and verify word counts inside.
```python
def find_substring(self, s: str, words: list[str]) -> list[int]:
    res = []
    if not s or not words: return res
    word_len = len(words[0])
    total_len = word_len * len(words)
    
    from collections import Counter
    counts = Counter(words)
    
    for i in range(len(s) - total_len + 1):
        seen = {}
        j = 0
        while j < len(words):
            w = s[i + j * word_len : i + (j + 1) * word_len]
            if w in counts:
                seen[w] = seen.get(w, 0) + 1
                if seen[w] > counts[w]: break
            else:
                break
            j += 1
        if j == len(words): res.append(i)
    return res
# Time Complexity: O(N * M * L) | Space Complexity: O(M)
```

* * *

**23. Contains Duplicate II**
**Specification:** Check if array has duplicates within distance k.

**Example:** `[1,2,3,1], k = 3` -> Output: `true`

**Pattern:** Sliding Window Set

**Explanation:** Keep a sliding set of size k. If add fails, duplicate found.
```python
def contains_nearby_duplicate(self, nums: list[int], k: int) -> bool:
    hash_set = set()
    for i in range(len(nums)):
        if i > k: hash_set.remove(nums[i - k - 1])
        if nums[i] in hash_set: return True
        hash_set.add(nums[i])
    return False
# Time Complexity: O(N) | Space Complexity: O(K)
```

* * *

**24. Count Number of Nice Subarrays**
**Specification:** Count subarrays with exactly k odd numbers.

**Example:** `nums = [1,1,2,1,1], k = 3` -> Output: `2`

**Pattern:** Prefix Sum of Odds

**Explanation:** Treat odds as 1s, evens as 0s. Same as subarray sum equals K.
```python
def number_of_subarrays(self, nums: list[int], k: int) -> int:
    from collections import defaultdict
    hash_map = defaultdict(int)
    hash_map[0] = 1
    total_sum = count = 0
    for num in nums:
        total_sum += num % 2
        count += hash_map[total_sum - k]
        hash_map[total_sum] += 1
    return count
# Time Complexity: O(N) | Space Complexity: O(N)
```

* * *

**25. Frequency of Most Frequent Element**
**Specification:** Max frequency of an element after incrementing at most K operations.

**Example:** `[1,2,4], k = 5` -> Output: `3`

**Pattern:** Sort + Sliding Window

**Explanation:** Sort first. To make all elements in window equal to `nums[right]`, we need `nums[right] * window_length - window_sum <= k`.
```python
def max_frequency(self, nums: list[int], k: int) -> int:
    nums.sort()
    left = total_sum = 0
    for right in range(len(nums)):
        total_sum += nums[right]
        if nums[right] * (right - left + 1) - total_sum > k:
            total_sum -= nums[left]
            left += 1
    return len(nums) - left
# Time Complexity: O(N log N) | Space Complexity: O(1)
```

* * *

**26. Subarrays with K Different Integers**
**Specification:** Count subarrays with exactly K distinct integers.

**Example:** `[1,2,1,2,3], K = 2` -> Output: `7`

**Pattern:** At-Most-K Trick

**Explanation:** Exactly(K) = AtMost(K) - AtMost(K-1).
```python
def subarrays_with_k_distinct(self, nums: list[int], k: int) -> int:
    return self._at_most_k(nums, k) - self._at_most_k(nums, k - 1)

def _at_most_k(self, nums: list[int], k: int) -> int:
    count = [0] * (len(nums) + 1)
    left = res = distinct = 0
    for right in range(len(nums)):
        if count[nums[right]] == 0: distinct += 1
        count[nums[right]] += 1
        while distinct > k:
            count[nums[left]] -= 1
            if count[nums[left]] == 0: distinct -= 1
            left += 1
        res += right - left + 1
    return res
# Time Complexity: O(N) | Space Complexity: O(N)
```

* * *

**27. Longest Palindromic Substring**
**Specification:** Find the longest substring that reads same backwards.

**Example:** `"babad"` -> Output: `"bab"`

**Pattern:** Expand Around Center

**Explanation:** Treat each character and between-character as a center and expand outwards to check for palindrome.
```python
def longest_palindrome(self, s: str) -> str:
    start = end = 0
    for i in range(len(s)):
        len1 = self._expand(s, i, i)
        len2 = self._expand(s, i, i + 1)
        length = max(len1, len2)
        if length > end - start:
            start = i - (length - 1) // 2
            end = i + length // 2
    return s[start:end + 1]

def _expand(self, s: str, l: int, r: int) -> int:
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1; r += 1
    return r - l - 1
# Time Complexity: O(N^2) | Space Complexity: O(1)
```

* * *

**28. 3Sum**
**Specification:** Find all unique triplets that sum to zero.

**Example:** `[-1,0,1,2,-1,-4]` -> Output: `[[-1,-1,2],[-1,0,1]]`

**Pattern:** Sort + Two Pointer

**Explanation:** Sort array. Iterate `i`, and use two pointers `L` and `R` to find pairs summing to `-nums[i]`. Skip duplicates.
```python
def three_sum(self, nums: list[int]) -> list[list[int]]:
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            total = nums[i] + nums[l] + nums[r]
            if total == 0:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l+1]: l += 1
                while l < r and nums[r] == nums[r-1]: r -= 1
                l += 1; r -= 1
            elif total < 0: l += 1
            else: r -= 1
    return res
# Time Complexity: O(N^2) | Space Complexity: O(1)
```

* * *

**29. 4Sum**
**Specification:** Find unique quadruplets summing to target.

**Example:** `nums = [1,0,-1,0,-2,2], target = 0` -> Output: `[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]`

**Pattern:** Sort + Nested Two Pointer

**Explanation:** Extend 3Sum by adding one more outer loop.
```python
def four_sum(self, nums: list[int], target: int) -> list[list[int]]:
    nums.sort()
    res = []
    for i in range(len(nums) - 3):
        if i > 0 and nums[i] == nums[i-1]: continue
        for j in range(i + 1, len(nums) - 2):
            if j > i + 1 and nums[j] == nums[j-1]: continue
            l, r = j + 1, len(nums) - 1
            while l < r:
                total = nums[i] + nums[j] + nums[l] + nums[r]
                if total == target:
                    res.append([nums[i], nums[j], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]: l += 1
                    while l < r and nums[r] == nums[r-1]: r -= 1
                    l += 1; r -= 1
                elif total < target: l += 1
                else: r -= 1
    return res
# Time Complexity: O(N^3) | Space Complexity: O(1)
```

* * *

**30. Number of Distinct Islands**
**Specification:** Count number of uniquely shaped islands in a grid.

**Example:** Grid with two identical 2x2 islands -> Output: `1`

**Pattern:** DFS + Path Signature Hashing

**Explanation:** Record the direction moved (U, D, L, R) during DFS traversal. Store path strings in a HashSet to deduplicate identical shapes.
```python
def num_distinct_islands(self, grid: list[list[int]]) -> int:
    hash_set = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                path = []
                self._dfs(grid, i, j, "S", path) # Start with 'S'
                hash_set.add("".join(path))
    return len(hash_set)

def _dfs(self, grid: list[list[int]], r: int, c: int, dir_str: str, path: list[str]) -> None:
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] == 0: return
    grid[r][c] = 0 # mark visited
    path.append(dir_str)
    self._dfs(grid, r + 1, c, "D", path)
    self._dfs(grid, r - 1, c, "U", path)
    self._dfs(grid, r, c + 1, "R", path)
    self._dfs(grid, r, c - 1, "L", path)
    path.append("B") # Backtrack to distinguish paths
# Time Complexity: O(R * C) | Space Complexity: O(R * C)
```

* * *

## Practice Problem Bank

**1. Contiguous Subarray Max Vowels**
**Specification:** Given string `s` and length `k`, find the maximum number of vowels in a substring of length `k`.

**Example:** `s = "abciiidef", k = 3` -> Output: `3` (for "iii")

**Constraints:** $1 \le s.length \le 10^5$, $1 \le k \le s.length$. Lowercase English letters.

**Strategic Hint:** Fixed-Size Sliding Window counting vowels as it slides.

**2. Number of Subarrays with Bounded Maximum**
**Specification:** Count subarrays such that the maximum value in the subarray is between `left` and `right` inclusive.

**Example:** `nums = [2,1,4,3], left = 2, right = 3` -> Output: `3` ([2], [2, 1], [3])

**Constraints:** $1 \le nums.length \le 10^5$, $0 \le nums[i] \le 10^9$.

**Strategic Hint:** Two-pointer tracking valid range start and last valid number seen.

**3. Grumpy Bookstore Owner**
**Specification:** Maximize customers satisfied over an array. The owner is grumpy at some indices. You have one `minutes` long secret technique to keep them not grumpy.

**Example:** `customers=[1,0,1,2,1,1,7,5], grumpy=[0,1,0,1,0,1,0,1], minutes=3` -> Output: `16`

**Constraints:** Arrays equal length $\le 2 \times 10^4$.

**Strategic Hint:** Fixed-size window tracking the max potential customer gain.

**4. Minimum Flips to Make Binary String Alternating**
**Specification:** You can remove the first char and append it to the end. Find the min operations to change the string into an alternating string of 0s and 1s.

**Example:** `"111000"` -> Output: `2`

**Constraints:** $1 \le s.length \le 10^5$.

**Strategic Hint:** Double the string string (`s+s`) and use a Fixed-Size Sliding Window of length $N$.

**5. Maximize the Confusion of an Exam**
**Specification:** Given a string of 'T' and 'F', flip at most $K$ answers to maximize consecutive identical answers.

**Example:** `"TTFF", k=2` -> Output: `4`

**Constraints:** $1 \le len \le 5 \times 10^4$.

**Strategic Hint:** Apply Max Consecutive Ones III logic separately for 'T' and 'F'.

**6. Longest Subarray of 1s After Deleting One Element**
**Specification:** You must delete exactly one element. Find the max continuous 1s remaining.

**Example:** `[1,1,0,1]` -> Output: `3`

**Constraints:** $1 \le nums.length \le 10^5$.

**Strategic Hint:** Dynamic Window with a zero-budget of exactly 1.

**7. Repeated DNA Sequences**
**Specification:** Find all 10-letter-long sequences occurring more than once.

**Example:** `"AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"` -> Output: `["AAAAACCCCC","CCCCCAAAAA"]`

**Constraints:** $1 \le s.length \le 10^5$.

**Strategic Hint:** Fixed window length 10 hashing strings or bitmask signatures.

**8. K-diff Pairs in an Array**
**Specification:** Count unique pairs $(i,j)$ such that $|nums[i] - nums[j]| == k$.

**Example:** `[3,1,4,1,5], k=2` -> Output: `2` (1,3 and 3,5)

**Constraints:** Array length $\le 10^4$.

**Strategic Hint:** HashMap counting frequencies. Check `num + k` for $k > 0$ and frequency $> 1$ for $k=0$.

**9. Check if Array Pairs Are Divisible by k**
**Specification:** Can we pair up all elements such that every pair sum is divisible by $k$?

**Example:** `[1,2,3,4,5,10,6,7,8,9], k=5` -> Output: `true`

**Constraints:** Array length even, $\le 10^5$.

**Strategic Hint:** Modulo arithmetic counts array. Count of `x` must equal count of `k - x`.

**10. Count Vowel Substrings of a String**
**Specification:** Substrings containing only vowels, and at least one of each ('a','e','i','o','u').

**Example:** `"aeiouu"` -> Output: `2`

**Constraints:** $1 \le s.length \le 100$.

**Strategic Hint:** Dynamic Window with vowel frequency tracking.

**11. Subarray Sums Divisible by K**
**Specification:** Count subarrays whose sum is divisible by $K$.

**Example:** `[4,5,0,-2,-3,1], k = 5` -> Output: `7`

**Constraints:** $1 \le nums.length \le 3 \times 10^4$.

**Strategic Hint:** Prefix Sum + Modulo HashMap grouping.

**12. Find the Longest Substring Containing Vowels in Even Counts**
**Specification:** Max length substring with all vowels appearing an even number of times.

**Example:** `"eleetminicoworoep"` -> Output: `13`

**Constraints:** $1 \le s.length \le 5 \times 10^5$.

**Strategic Hint:** Prefix Sum with Bitmask (5 bits) mapped to first occurrence indices.

**13. Matrix Block Sum**
**Specification:** Compute sum of elements in a submatrix defined by a distance $K$.

**Example:** $3\times3$ grid, $K=1$. Output is block sums.

**Constraints:** Matrix dimensions $\le 100$.

**Strategic Hint:** 2D Prefix Sum Array. `pref[i][j] = val + pref[i-1][j] + pref[i][j-1] - pref[i-1][j-1]`.

**14. Replace the Substring for Balanced String**
**Specification:** String has 'Q', 'W', 'E', 'R'. Replace a minimal substring to make counts exactly $N/4$.

**Example:** `"QWER"` -> Output: `0`. `"QQWE"` -> Output: `1`.

**Constraints:** length is multiple of 4.

**Strategic Hint:** Dynamic Window matching missing characters needed outside the window.

**15. Longest Substring Of All Vowels in Order**
**Specification:** Substring must contain all 5 vowels in alphabetical order.

**Example:** `"aeiaaioaaaaeiiiiouuuooaauuaeiu"` -> Output: `13`

**Constraints:** string size $\le 5 \times 10^5$.

**Strategic Hint:** Dynamic Window resetting when order is broken or char is not a vowel.

**16. Count Good Meals**
**Specification:** Number of pairs of items whose sum is a power of two.

**Example:** `[1,3,5,7,9]` -> Output: `4`

**Constraints:** Elements $\le 2^{20}$.

**Strategic Hint:** Two Sum with target looping through all 22 powers of two.

**17. Largest Subarray of 0's and 1's**
**Specification:** Exact same as 'Contiguous Array', formulated differently.

**Example:** `[0,1]` -> Output: `2`

**Constraints:** size $\le 10^5$.

**Strategic Hint:** Prefix sum converting 0 to -1, check map for first occurrence.

**18. Sort Characters By Frequency**
**Specification:** Sort string based on character frequencies descending.

**Example:** `"tree"` -> Output: `"eert"` or `"eetr"`

**Constraints:** $1 \le len \le 5 \times 10^5$.

**Strategic Hint:** HashMap for frequencies, then PriorityQueue or Bucket Sort.

**19. Number of Pairs of Strings With Concatenation Equal to Target**
**Specification:** Given array of strings, count pairs $(i,j)$ where `nums[i]+nums[j] == target`.

**Example:** `nums = ["777","7","77","77"], target = "7777"` -> Output: `4`

**Constraints:** $\le 100$ strings.

**Strategic Hint:** Hashmap string frequencies. Check target prefixes and suffixes.

**20. Arithmetic Slices**
**Specification:** Count contiguous subarrays forming arithmetic progressions of length $\ge 3$.

**Example:** `[1,2,3,4]` -> Output: `3`

**Constraints:** size $\le 5000$.

**Strategic Hint:** Dynamic Window or DP tracking consecutive diffs.

**21. Number of Submatrices That Sum to Target**
**Specification:** 2D version of Subarray Sum Equals K.

**Example:** $2\times2$ grid, target 0.

**Constraints:** Matrix $\le 100\times100$.

**Strategic Hint:** 2D Prefix Sums flattened into 1D for every pair of rows + HashMap.

**22. Count Trippets That Can Form Two Arrays of Equal XOR**
**Specification:** $a = arr[i] \dots arr[j-1]$, $b = arr[j] \dots arr[k]$. Count $(i,j,k)$ where $a == b$.

**Example:** `[2,3,1,6,7]` -> Output: `4`

**Constraints:** length $\le 300$.

**Strategic Hint:** Prefix XOR. $a == b \implies arr[i \dots k] == 0$.

**23. Maximum Number of Vowels in a Substring of Given Length**
**Specification:** Another variation of vowel counting with fixed K.

**Example:** `"leetcode", k=3` -> Output: `2`

**Constraints:** length $\le 10^5$.

**Strategic Hint:** Fixed window `O(N)` linear scan.

**24. Subarray With Given Sum**
**Specification:** Non-negative integers, find continuous subarray summing to S. (Return bounds).

**Example:** `[1,2,3,7,5], S=12` -> Output: `[2,4]` (1-based)

**Constraints:** Elements $> 0$.

**Strategic Hint:** Dynamic Window (since all positive, monotonic sum).

**25. Minimum Operations to Reduce X to Zero**
**Specification:** Remove elements from either left or right ends to make target X. Minimum ops.

**Example:** `nums = [1,1,4,2,3], x = 5` -> Output: `2`

**Constraints:** Elements $> 0$.

**Strategic Hint:** Find max length subarray summing to `TotalSum - X`.

**26. Distinct Numbers in Each Subarray**
**Specification:** Count distinct numbers in every window of size K.

**Example:** `[1,2,3,2,2,1,3], k=3` -> Output: `[3,2,2,2,3]`

**Constraints:** $1 \le n \le 10^5$.

**Strategic Hint:** Fixed size window with HashMap counting frequencies.

**27. Shortest Subarray with Sum at Least K**
**Specification:** Like minimum size subarray sum but array can have negatives!

**Example:** `[2,-1,2], k=3` -> Output: `3`

**Constraints:** Length $\le 10^5$.

**Strategic Hint:** Prefix sum + Monotonic Deque to maintain increasing prefix sums.

**28. Make Sum Divisible by P**
**Specification:** Remove smallest subarray so remaining array sum is divisible by P.

**Example:** `[3,1,4,2], p=6` -> Output: `1` (remove [4])

**Constraints:** Length $\le 10^5$.

**Strategic Hint:** Target remainder is `total_sum % P`. Find shortest subarray with this mod.

**29. Continuous Subarrays**
**Specification:** Subarrays where absolute diff between any two elements is $\le 2$.

**Example:** `[5,4,2,4]` -> Output: `8`

**Constraints:** Length $\le 10^5$.

**Strategic Hint:** Dynamic Window with TreeMap or Monotonic Queues to track min/max.

**30. Longest Subarray With Maximum Bitwise AND**
**Specification:** Find max bitwise AND possible, then find longest subarray with that value.

**Example:** `[1,2,3,3,2,2]` -> Output: `2`

**Constraints:** Length $\le 10^5$.

**Strategic Hint:** Max AND is just the max element. Find longest contiguous sequence of the max element.


# Hard-tier Mastery — Algorithmic Optimization: Binary Search Variants, Monotonic Structures, Dynamic Programming, and Graph Algorithms

This chapter covers Hard-tier of the General Coding Assessments (Hard difficulty, ~25 minutes target time). Hard-tier is the most challenging question testing optimal $\mathcal{O}(\log N)$ or $\mathcal{O}(N)$ solutions, DP state transitions, and graph algorithms.

## Essential Terminology & Vocabulary

### Rotated Sorted Array & Monotonic Partition Invariant

**1. Conceptual Definition: What is a Rotated Sorted Array?**
A **Rotated Sorted Array** is an array that was originally sorted in ascending order (with unique elements), but has been shifted (rotated) at some unknown pivot index $K$.

For example, consider the original sorted array:
```
Original Sorted Array: [0, 1, 2, 4, 5, 6, 7]
```

If we rotate this array at pivot index $K = 3$ (shifting elements from index 3 onwards to the front), we get:

```
Rotated Sorted Array: [4, 5, 6, 7, 0, 1, 2]
```

Notice what happened:

- The single monotonically increasing sequence is split into **two sorted sub-arrays**: $[4, 5, 6, 7]$ (the left segment) and $[0, 1, 2]$ (the right segment).
- The array is no longer sorted overall, so standard Binary Search (which assumes `nums[left] <= nums[right]`) fails if implemented naively.

![Binary Search on Rotated Sorted Array — Two Sorted Halves](editions/python/chapters/13-optimization-dp/visuals/rotated_sorted_array.png){width=85%}

* * *

**2. The Core Mathematical Invariant**
The key insight that allows us to achieve $\mathcal{O}(\log N)$ time complexity is the **Monotonic Partition Invariant**:

> **The Fundamental Invariant:** Whenever you split a Rotated Sorted Array into two halves using a midpoint `mid = left + (right - left) / 2`, **AT LEAST ONE OF THE TWO HALVES IS GUARANTEED TO BE STRICTLY MONOTONICALLY SORTED.**

> **Proof by Exhaustion.** Consider array `A[lo..hi]` with midpoint `mid = (lo + hi) / 2`. The rotation point (the index where `A[i] > A[i+1]`) can only exist in one contiguous segment.
> - **Case 1:** Rotation point is in `A[mid+1..hi]`. Then `A[lo..mid]` contains no rotation point, so `A[lo] ≤ A[lo+1] ≤ ... ≤ A[mid]` — the left half is sorted.
> - **Case 2:** Rotation point is in `A[lo..mid]`. Then `A[mid+1..hi]` contains no rotation point, so `A[mid+1] ≤ ... ≤ A[hi]` — the right half is sorted.
> - **Case 3:** No rotation point exists in `A[lo..hi]` (entire subarray is sorted). Both halves are sorted.
>
> In all cases, at least one half is sorted. ∎

- If `nums[left] <= nums[mid]`: The **LEFT half** `[left ... mid]` is monotonically sorted.
- If `nums[left] > nums[mid]`: The **RIGHT half** `[mid ... right]` is monotonically sorted.

* * *

**3. Step-by-Step Binary Search Decision Rule**
Because one half is always sorted, we can easily check if our `target` lies within the boundaries of that sorted half:

1. Calculate `mid = left + (right - left) / 2`.
2. If `nums[mid] == target`, return `mid` immediately.
3. Check which half is sorted:

   - **Case A: Left Half `[left ... mid]` is Sorted (`nums[left] <= nums[mid]`)**
     - Is `target` in range `[nums[left] ... nums[mid]]`?
       - If **Yes**: Eliminate the right half $\rightarrow$ `right = mid - 1`.
       - If **No**: Eliminate the left half $\rightarrow$ `left = mid + 1`.
   - **Case B: Right Half `[mid ... right]` is Sorted (`nums[left] > nums[mid]`)**
     - Is `target` in range `[nums[mid] ... nums[right]]`?
       - If **Yes**: Eliminate the left half $\rightarrow$ `left = mid + 1`.
       - If **No**: Eliminate the right half $\rightarrow$ `right = mid - 1`.

* * *

**4. Complete Worked Execution Trace (`nums = [4, 5, 6, 7, 0, 1, 2]`, `target = 0`)**

Let's trace searching for `target = 0`:

* **Iteration 1:**
  - `left = 0` (val `4`), `right = 6` (val `2`).
  - `mid = 0 + (6 - 0) / 2 = 3` $\rightarrow$ `nums[3] = 7`.
  - Is `nums[mid] == target`? `7 == 0` (False).
  - Check left half sortedness: `nums[0] (4) <= nums[3] (7)` $\rightarrow$ **Left Half `[4, 5, 6, 7]` IS SORTED.**
  - Is `target` (0) within `[4 ... 7]`? No (`0 < 4`).
  - Action: Eliminate left half $\rightarrow$ `left = mid + 1 = 4`.

* **Iteration 2:**
  - `left = 4` (val `0`), `right = 6` (val `2`).
  - `mid = 4 + (6 - 4) / 2 = 5` $\rightarrow$ `nums[5] = 1`.
  - Is `nums[mid] == target`? `1 == 0` (False).
  - Check left half sortedness: `nums[4] (0) <= nums[5] (1)` $\rightarrow$ **Left Half `[0, 1]` IS SORTED.**
  - Is `target` (0) within `[0 ... 1]`? Yes! (`0 >= 0` and `0 <= 1`).
  - Action: Eliminate right half $\rightarrow$ `right = mid - 1 = 4`.

* **Iteration 3:**
  - `left = 4` (val `0`), `right = 4` (val `0`).
  - `mid = 4 + (4 - 4) / 2 = 4` $\rightarrow$ `nums[4] = 0`.
  - Is `nums[mid] == target`? `0 == 0` (True!).
  - **Return index `4`!** (Exact $\mathcal{O}(\log N)$ solution reached in 3 steps).

### Binary Search on Answer Space (Parametric Binary Search)
**Definition:** A technique where we search for an optimal value (the "answer") within a known range `[low, high]` instead of searching for a specific element in an array. We use a monotonic predicate function (e.g., `canFulfill(mid)`) to determine whether a given value `mid` is feasible. 
**Why it matters:** It transforms optimization problems (e.g., "find the minimum capacity") into a series of simpler decision problems (e.g., "is capacity X sufficient?"), enabling $\mathcal{O}(N \log(\max - \min))$ solutions.
**When to use:** When the answer space is bounded, the feasibility function is monotonic (if $x$ is valid, $x+1$ is also valid, or vice versa), and calculating feasibility takes linear time $\mathcal{O}(N)$.

### Monotonic Stack & Deque
**Definition:** A stack or double-ended queue (deque) where elements are maintained in strictly increasing or strictly decreasing order. 
**Why it matters:** It provides $\mathcal{O}(1)$ amortized time complexity for range maximum/minimum lookups or finding the "next greater element". Elements are pushed and popped at most once.
**When to use:** Finding the next greater/smaller element, sliding window maximum/minimum, and calculating histogram areas.

### Dynamic Programming State Transition (1D, 2D, Interval DP)
**Definition:** The mathematical rule or formula that relates the solution of a larger problem to its smaller overlapping subproblems. 

- **1D DP:** The state depends on a single variable (e.g., index `i`). Transition: `dp[i] = dp[i-1] + dp[i-2]`.
- **2D DP:** The state depends on two variables (e.g., strings of length `i` and `j`). Transition: `dp[i][j] = ...`.
- **Interval DP:** The state is defined by a range `[i, j]`. Subproblems are smaller intervals within the range.

**Why it matters:** Properly defining the state and transition is the core of any DP solution. It turns exponential $\mathcal{O}(2^N)$ backtracking into polynomial time $\mathcal{O}(N)$ or $\mathcal{O}(N^2)$ solutions.

### Memoization vs Tabulation
**Definition:** The two primary methods for implementing Dynamic Programming.

| Feature | Memoization (Top-Down) | Tabulation (Bottom-Up) |
| --- | --- | --- |
| **Direction** | Start from the main problem, recursively call subproblems. | Start from base cases, iteratively build up to the main problem. |
| **State Storage** | Hash Map or Array. | N-dimensional Array. |
| **Overhead** | Recursive stack overhead (potential StackOverflow). | No recursive overhead, generally faster constant time. |
| **When to use** | When not all subproblems need to be evaluated. | When all subproblems will definitely be evaluated. |

### Knapsack Variants
**Definition:** A family of combinatorial optimization problems involving packing items into a capacity-constrained space to maximize value.

- **0/1 Knapsack:** Each item can be chosen at most once. Transition relies on picking or skipping: `dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i])`.
- **Unbounded Knapsack:** Each item can be chosen infinitely many times.
- **Subset Sum:** A specialized 0/1 Knapsack where we want to know if a subset sums exactly to `target`.

**Why it matters:** They form the basis for numerous resource allocation and subset combination problems in technical interviews.

### Topological Sort
**Definition:** A linear ordering of vertices in a Directed Acyclic Graph (DAG) such that for every directed edge $U \rightarrow V$, vertex $U$ comes before $V$ in the ordering.
**Why it matters:** Kahn's Algorithm (using an in-degree array and queue) processes dependencies efficiently in $\mathcal{O}(V + E)$ time.
**When to use:** Task scheduling, resolving prerequisites (like courses or build systems), finding dependency cycles.

### BFS Shortest Path
**Definition:** Breadth-First Search traversal to find the shortest path in an **unweighted** graph. It processes nodes level-by-level using a Queue.
**Why it matters:** It guarantees that the first time a target node is reached, it is via the shortest possible path (fewest edges). 
**When to use:** Shortest path on grids or unweighted graphs, state transitions requiring fewest moves (like word ladders or minimum jumps).

### Two-pointer
**Definition:** Using two indices (usually `left` and `right`) to traverse a sequence simultaneously. 
**Why it matters:** It optimally narrows down search spaces without requiring extra memory, often reducing $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.
**When to use:** Finding pairs in sorted arrays, bounding areas (like trapping rain water or container with most water), and cycle detection.

### Greedy
**Definition:** Making the locally optimal choice at each step with the hope that these local choices lead to a globally optimal solution.
**Why it matters:** When a greedy choice property can be proven (e.g., via contradiction or exchange arguments), the algorithm is extremely fast and space-efficient.
**When to use:** Interval scheduling, jump games, Huffman coding, minimum spanning trees.

### DP State Compression
When processing a dynamic programming grid where the current row only depends on the previous row, this technique replaces `dp[N][M]` with two 1D arrays `prev[]` and `curr[]`.
Why it matters: It halves memory usage and often reduces O(N*M) space to O(M).

### Binary Search Loop Termination
This deals with the crucial choice between `while (lo < hi)` and `while (lo <= hi)` loops, paired with `mid = lo + (hi - lo) / 2` to prevent overflow. Choosing the wrong termination condition causes infinite loops.
Why it matters: Boundary conditions and termination logic are the #1 source of binary search bugs.

### Interval DP Framework
This DP pattern defines the state `dp[i][j]` as the optimal solution for a subarray from `i` to `j`. It enumerates a split point `k` to divide the interval into smaller subproblems.
Why it matters: It is essential for solving burst balloons, matrix chain multiplication, and palindrome partitioning.

### Union-Find (Disjoint Set Union)
This data structure tracks elements partitioned into disjoint subsets. It combines a `find()` method using path compression with a `union()` method utilizing union-by-rank to achieve near O(1) amortized operations.
Why it matters: It is the optimal structure for connected components, cycle detection, and Kruskal's Minimum Spanning Tree.

### Dijkstra's Algorithm
This is an optimal pathfinding algorithm that utilizes a priority queue for a breadth-first search on weighted edges. It processes nodes in order of shortest accumulated distance in O((V+E) log V) time.
Why it matters: It is the gold standard for solving single-source shortest path problems with non-negative weights.

### Backtracking Template
This pattern generates all possible configurations by exhaustively exploring decision trees. It strictly follows a "choose, explore, unchoose" structured layout with early pruning.
Why it matters: It is universally used to generate all combinations, permutations, and subsets in optimization problems.

### LRU Cache Architecture
This system design paradigm combines a `HashMap<Key, Node>` with a doubly linked list. The hash map provides instant access, while the list maintains temporal usage order.
Why it matters: It allows O(1) get and put operations by seamlessly combining hash lookup with ordered eviction.

### Fibonacci DP Recognition
This refers to identifying when a problem's state perfectly maps to the linear recurrence `dp[i] = dp[i-1] + dp[i-2]`. The entire array state can be compressed into two variables.
Why it matters: Problems like climbing stairs, decode ways, and tiling can be instantly recognized and compressed to O(1) space.

![DP State Transition — Climbing Stairs with Space Optimization](editions/python/chapters/13-optimization-dp/visuals/dp_climbing_stairs.png){width=85%}

* * *

## Reusable Code Templates

### Template A: Binary Search
```python
# Standard Binary Search
def binary_search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target: return mid
        elif nums[mid] < target: left = mid + 1
        else: right = mid - 1
    return -1

# Binary Search on Answer Space (Leftmost valid)
def binary_search_answer_space(min_val: int, max_val: int) -> int:
    left, right = min_val, max_val
    best = -1
    while left <= right:
        mid = left + (right - left) // 2
        if is_valid(mid):
            best = mid
            right = mid - 1 # Try to find a smaller valid answer
        else:
            left = mid + 1
    return best
```

### Template B: Monotonic Stack
```python
def next_greater_element(self, nums: list[int]) -> list[int]:
    n = len(nums)
    result = [-1] * n
    stack = [] # stores indices
    for i in range(n):
        # Maintain strictly decreasing stack
        while stack and nums[i] > nums[stack[-1]]:
            prev_index = stack.pop()
            result[prev_index] = nums[i] # Found next greater!
        stack.append(i)
    return result
```

### Template C: 1D DP with State Compression
```python
def dp_state_compression(self, nums: list[int]) -> int:
    if not nums: return 0
    prev2 = 0 # dp[i-2]
    prev1 = nums[0] # dp[i-1]
    for i in range(1, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = curr
    return prev1
```

### Template D: BFS with Level Tracking
```python
def bfs_level(self, start: 'Node', target: 'Node') -> int:
    from collections import deque
    queue = deque([start])
    visited = {start}
    
    level = 0
    while queue:
        size = len(queue)
        for _ in range(size):
            curr = queue.popleft()
            if curr == target: return level
            
            for neighbor in curr.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        level += 1 # Increment level after exploring all nodes at current depth
    return -1
```

### Template E: Topological Sort (Kahn's Algorithm)
```python
def topological_sort(self, num_nodes: int, edges: list[list[int]]) -> list[int]:
    from collections import deque
    adj = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    
    for u, v in edges:
        adj[v].append(u) # v -> u
        in_degree[u] += 1
        
    queue = deque(i for i in range(num_nodes) if in_degree[i] == 0)
    
    order = []
    while queue:
        curr = queue.popleft()
        order.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    return order if len(order) == num_nodes else [] # Empty if cycle exists
```

* * *

## Solved Exemplar Problems

**1. Search in Rotated Sorted Array**
**Difficulty Classification:** This problem is classified as Medium on all major assessment platforms. It appears in this chapter because it demonstrates the advanced application of the Binary Search pattern **[PAT-10] Monotonic Partition Binary Search** with a modified invariant. For assessment preparation, treat this as a medium-tier warm-up before tackling the harder DP and graph problems in this chapter.
**Specification:** Given an integer array sorted in ascending order (with distinct values) and rotated at an unknown pivot, find the index of `target`.

**Example:** `nums = [4,5,6,7,0,1,2]`, `target = 0` $\rightarrow$ output `4`.

**Pattern:** Rotated Binary Search

**Explanation:** We use the monotonic partition invariant. At any midpoint, at least one half of the array is strictly sorted. We identify the sorted half and check if the target falls within its range.

```python
def search(self, nums: list[int], target: int) -> int:
    if not nums: return -1
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target: return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1 # Target is in the sorted left half
            else:
                left = mid + 1 # Target must be in the right half
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1 # Target is in the sorted right half
            else:
                right = mid - 1 # Target must be in the left half
    return -1
# Time Complexity: O(log N)
# Space Complexity: O(1)
```

* * *

**2. Sliding Window Maximum**
**Specification:** Return an array of the maximum values in every sliding window of size `K`.

**Example:** `nums = [1,3,-1,-3,5,3,6,7]`, `k = 3` $\rightarrow$ output `[3,3,5,5,6,7]`.

**Pattern:** Monotonic Deque

**Explanation:** We maintain a deque of indices such that the values are in strictly decreasing order. The front of the deque always holds the maximum element's index for the current window. We remove elements from the front that fall out of the window.

```python
def max_sliding_window(self, nums: list[int], k: int) -> list[int]:
    if not nums or k <= 0: return []
    n = len(nums)
    res = [0] * (n - k + 1)
    res_index = 0
    from collections import deque
    q = deque()
    
    for i in range(n):
        # Remove indices outside the current window
        if q and q[0] < i - k + 1:
            q.popleft()
        # Remove smaller elements (maintain decreasing order)
        while q and nums[q[-1]] < nums[i]:
            q.pop()
        q.append(i)
        
        # Record max for the window
        if i >= k - 1:
            res[res_index] = nums[q[0]]
            res_index += 1
            
    return res
# Time Complexity: O(N) since each element is pushed/popped at most once
# Space Complexity: O(K) for the deque
```

* * *

**3. Longest Common Subsequence**
**Specification:** Return the length of the longest common subsequence between two strings.

**Example:** `text1 = "abcde"`, `text2 = "ace"` $\rightarrow$ output `3` ("ace").

**Pattern:** 2D DP

> ⚠️ **Common Confusion: Subsequence ≠ Substring**
>
> A **substring** must be contiguous (`"BCD"` from `"ABCDE"`). A **subsequence** can skip characters but must preserve order (`"ACE"` from `"ABCDE"` — pick A, skip B, pick C, skip D, pick E). The order matters: `"ECA"` is **not** a valid subsequence of `"ABCDE"` because the characters appear in the wrong order.

![Subsequence vs Substring](editions/python/chapters/13-optimization-dp/visuals/subsequence_vs_substring.png){width=85%}

**Trace-Through:** For `text1 = "CAT"`, `text2 = "CART"`, the DP table builds the answer cell by cell. Each cell asks: "What is the longest common subsequence using only the first *i* characters of text1 and first *j* characters of text2?"

|  | "" | C | A | R | T |
|---|---|---|---|---|---|
| **""** | 0 | 0 | 0 | 0 | 0 |
| **C** | 0 | **1** ↖ | 1 ← | 1 ← | 1 ← |
| **A** | 0 | 1 ↑ | **2** ↖ | 2 ← | 2 ← |
| **T** | 0 | 1 ↑ | 2 ↑ | 2 ↑ | **3** ↖ |

- ↖ (diagonal + 1): Characters **match** — extend the LCS we had before both characters.
- ← or ↑ (max of left/above): Characters **don't match** — carry forward the best LCS from skipping one character.

The bold diagonal cells show: C matches C (1), A matches A (2), T matches T (3). The "R" in "CART" is simply skipped. **LCS = "CAT", length 3.**

**Explanation:** `dp[i][j]` represents the LCS of the prefixes of length `i` and `j`. If characters match, we add 1 to the result of `dp[i-1][j-1]`. If not, we take the max of skipping a character in either string.

```python
def longest_common_subsequence(self, text1: str, text2: str) -> int:
    if len(text1) < len(text2): return self.longest_common_subsequence(text2, text1)
    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        curr = [0] * (n + 1)
        
    return prev[n]
# Time Complexity: O(M * N)
# Space Complexity: O(min(M, N)) - Space compressed DP as taught in the vocabulary section.
```

* * *

**4. Burst Balloons**
> ⚠️ **Assessment Realism Note:** Interval DP problems like Burst Balloons are extremely unlikely in timed assessments (the O(N³) derivation requires 30+ minutes of focused work). This exemplar is included for comprehensive pattern coverage. For timed assessment practice, prioritize the multi-source BFS, 1D DP, and monotonic stack problems in this chapter.

**Specification:** Maximize coins by bursting balloons. Bursting `nums[i]` yields `nums[i-1] * nums[i] * nums[i+1]` coins.

**Example:** `nums = [3,1,5,8]` $\rightarrow$ output `167`.

**Pattern:** Interval DP

> ⚠️ **The Key Trick: Think BACKWARDS**
>
> The natural instinct is to simulate bursting balloons left-to-right, but that creates dependency chaos — bursting balloon `i` changes the neighbors of balloon `i+1`. Instead, ask: **"Which balloon do I burst LAST?"** If balloon `k` is the *last* to burst in interval `(i, j)`, then at that moment only `arr[i]` and `arr[j]` remain as its neighbors. This makes the left and right subproblems *independent*.

![Burst Balloons — Think Backwards](editions/python/chapters/13-optimization-dp/visuals/burst_balloons_trace.png){width=85%}

**Trace-Through:** For `nums = [3, 1, 5, 8]`, we pad with 1s: `arr = [1, 3, 1, 5, 8, 1]`.

- **Interval length 1** (single balloons): burst `3` alone → `1×3×1 = 3`. Burst `1` alone → `3×1×5 = 15`. Burst `5` alone → `1×5×8 = 40`. Burst `8` alone → `5×8×1 = 40`.
- **Interval length 2** (pairs): Try each as the *last* to burst. E.g., for `(3,1)`: if `3` is last → `1×3×5 + dp[1][2] = 15 + 15 = 30`. If `1` is last → `1×1×5 + dp[0][1] = 5 + 3 = 8`. Best = `30`.
- **Build up** to the full interval `dp[0][5]` = `167`.

The three nested loops enumerate: interval length → starting position → which balloon is last.

**Explanation:** We think backwards: what is the LAST balloon to be burst in an interval `[left, right]`? This allows us to split the problem into independent subproblems. `dp[i][j]` is the max coins obtained from bursting balloons strictly between `i` and `j`.

```python
def max_coins(self, nums: list[int]) -> int:
    n = len(nums)
    arr = [1] + nums + [1] # Padding with 1s
    
    dp = [[0] * (n + 2) for _ in range(n + 2)]
    
    # len_ is the length of the interval strictly between i and j
    for len_ in range(1, n + 1):
        for i in range(n - len_ + 1):
            j = i + len_ + 1
            # k is the index of the LAST balloon to burst in (i, j)
            for k in range(i + 1, j):
                coins = arr[i] * arr[k] * arr[j] + dp[i][k] + dp[k][j]
                dp[i][j] = max(dp[i][j], coins)
                
    return dp[0][n + 1]
# Time Complexity: O(N^3)
# Space Complexity: O(N^2)
```

* * *

**5. Maximum Product Subarray**
**Specification:** Find a contiguous non-empty subarray with the maximum product.

**Example:** `nums = [2,3,-2,4]` $\rightarrow$ output `6` (subarray `[2,3]`).

**Pattern:** 1D DP (Min/Max Tracking)

**Explanation:** Since multiplying two negative numbers yields a positive number, we must track BOTH the maximum product and the minimum product ending at the current position.

```python
def max_product(self, nums: list[int]) -> int:
    if not nums: return 0
    max_val = min_val = result = nums[0]
    
    for i in range(1, len(nums)):
        # If current is negative, max and min will swap roles
        if nums[i] < 0:
            max_val, min_val = min_val, max_val
            
        max_val = max(nums[i], max_val * nums[i])
        min_val = min(nums[i], min_val * nums[i])
        result = max(result, max_val)
        
    return result
# Time Complexity: O(N)
# Space Complexity: O(1)
```

* * *

**6. Median of Two Sorted Arrays**
**Specification:** Find the median of two sorted arrays in $\mathcal{O}(\log(M+N))$ time.

**Example:** `nums1 = [1,3]`, `nums2 = [2]` $\rightarrow$ output `2.0`.

**Pattern:** Binary Search on Partitions

**Explanation:** We binary search for the correct partition index in the smaller array such that the left halves of both arrays contain exactly half the total elements, and the largest element on the left is $\le$ the smallest element on the right.

```python
def find_median_sorted_arrays(self, A: list[int], B: list[int]) -> float:
    if len(A) > len(B): return self.find_median_sorted_arrays(B, A) # ensure A is smaller
    m, n = len(A), len(B)
    left, right = 0, m
    
    while left <= right:
        i = (left + right) // 2 # partition A
        j = (m + n + 1) // 2 - i # partition B
        
        max_left_a = float('-inf') if i == 0 else A[i - 1]
        min_right_a = float('inf') if i == m else A[i]
        max_left_b = float('-inf') if j == 0 else B[j - 1]
        min_right_b = float('inf') if j == n else B[j]
        
        if max_left_a <= min_right_b and max_left_b <= min_right_a:
            # Correct partition found
            if (m + n) % 2 == 0:
                return (max(max_left_a, max_left_b) + min(min_right_a, min_right_b)) / 2.0
            else:
                return max(max_left_a, max_left_b)
        elif max_left_a > min_right_b:
            right = i - 1 # move partition left in A
        else:
            left = i + 1 # move partition right in A
            
    return 0.0
# Time Complexity: O(log(min(M, N)))
# Space Complexity: O(1)
```

* * *

**7. Trapping Rain Water**
**Specification:** Calculate how much rain water can be trapped after raining.

**Example:** `height = [0,1,0,2,1,0,1,3,2,1,2,1]` $\rightarrow$ output `6`.

**Pattern:** Two-Pointer

**Explanation:** The amount of water above a bar depends on `min(max_left, max_right)`. We use two pointers from both ends, safely moving the pointer that points to the strictly smaller max bound, adding water along the way.

```python
def trap(self, height: list[int]) -> int:
    if not height: return 0
    left, right = 0, len(height) - 1
    left_max = right_max = total_water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max: left_max = height[left]
            else: total_water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max: right_max = height[right]
            else: total_water += right_max - height[right]
            right -= 1
            
    return total_water
# Time Complexity: O(N)
# Space Complexity: O(1)
```

* * *

**8. Daily Temperatures**
*Note: While placed in this chapter for its use of the Monotonic Stack pattern **[PAT-09] Monotonic Stack ("The Waiting Room")**, this problem is a Medium-difficulty gateway to the pattern. Use it as a warm-up before tackling the harder exemplars below.*
**Specification:** Find the number of days you have to wait after each day to get a warmer temperature.

**Example:** `[73,74,75,71,69,72,76,73]` $\rightarrow$ output `[1,1,4,2,1,1,0,0]`.

**Pattern:** Monotonic Stack

**Explanation:** We maintain a stack of indices representing days where we haven't found a warmer day yet (decreasing order). When we find a warmer day, we pop from the stack and compute the wait time.

```python
def daily_temperatures(self, temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    res = [0] * n
    stack = []
    
    for i in range(n):
        # While current temp is greater than temp at stack top
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_index = stack.pop()
            res[prev_index] = i - prev_index
        stack.append(i)
        
    return res
# Time Complexity: O(N)
# Space Complexity: O(N)
```

* * *

**9. Edit Distance / Levenshtein**
**Specification:** Minimum insertions, deletions, substitutions to convert `word1` to `word2`.

**Example:** `word1 = "horse"`, `word2 = "ros"` $\rightarrow$ output `3`.

**Pattern:** 2D DP

> ⚠️ **The Three Operations — Mapped to Table Directions**
>
> At each cell, you choose the cheapest of three operations: **Replace** (↖ diagonal + 1), **Delete** from word1 (↑ up + 1), **Insert** into word1 (← left + 1). If characters already match, the diagonal costs 0 (no operation needed).

![Edit Distance Trace](editions/python/chapters/13-optimization-dp/visuals/edit_distance_trace.png){width=85%}

**Trace-Through:** Convert `"CAT"` → `"CUT"` (answer: 1 — just replace A with U).

|  | "" | C | U | T |
|---|---|---|---|---|
| **""** | 0 | 1 | 2 | 3 |
| **C** | 1 | **0** ↖ | 1 | 2 |
| **A** | 2 | 1 | **1** ↖ | 2 |
| **T** | 3 | 2 | 2 | **1** ↖ |

- **Row 0 / Col 0** (base cases): Converting "" → "CUT" costs 3 inserts. Converting "CAT" → "" costs 3 deletes.
- **dp[1][1]:** C = C → match! Free! Diagonal `dp[0][0]` = 0.
- **dp[2][2]:** A ≠ U → mismatch. `1 + min(dp[1][1], dp[1][2], dp[2][1])` = `1 + min(0, 1, 1)` = **1** (replace A→U).
- **dp[3][3]:** T = T → match! Diagonal `dp[2][2]` = 1. **Answer: 1 edit.**

**Real-world use:** Spell checkers, DNA alignment, fuzzy string matching, and `git diff` all use variants of this algorithm.

**Explanation:** `dp[i][j]` is the edit distance between `word1` prefix length `i` and `word2` prefix length `j`. If characters match, cost is `dp[i-1][j-1]`. Otherwise, cost is `1 + min(insert, delete, replace)`.

```python
def min_distance(self, word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] # No op
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], # Replace
                                   dp[i - 1][j],     # Delete
                                   dp[i][j - 1])     # Insert
                                   
    return dp[m][n]
# Time Complexity: O(M * N)
# Space Complexity: O(M * N)
```

* * *

**10. LRU Cache**
**Specification:** Design a cache with Least Recently Used eviction policy supporting `get` and `put` in $\mathcal{O}(1)$ time.

**Pattern:** HashMap + Doubly Linked List

> ⚠️ **"Why no timestamp?" — Position IS the Timestamp**
>
> A common question is: "Shouldn't we store a timestamp for when each item was last used?" The answer is no — the **position in the linked list** is the timestamp. The node closest to HEAD was used most recently. The node closest to TAIL was used longest ago. Every `get()` or `put()` moves that node to the HEAD. No clock needed — the list order *is* the chronological record.

![LRU Cache — Position is the Timestamp](editions/python/chapters/13-optimization-dp/visuals/lru_cache_diagram.png){width=85%}

**Trace-Through:** Cache capacity = 2.

| Operation | HashMap | Linked List (HEAD → TAIL) | Why |
|:-----------------|:-------------------|:--------------------------|:-------------------------------------------------------|
| `put(1, "A")` | {1→A} | **[1]** | First entry, goes to head |
| `put(2, "B")` | {1→A, 2→B} | **[2, 1]** | Newest at head |
| `get(1)` | {1→A, 2→B} | **[1, 2]** | Accessed 1 → move to head |
| `put(3, "C")` | {1→A, 3→C} | **[3, 1]** | Full! Evict tail (2). Add 3 at head |
| `get(2)` | returns -1 | **[3, 1]** | Key 2 was evicted |

Notice: after `get(1)`, key 1 moved to head, saving it from eviction. Key 2, untouched at the tail, got evicted when capacity was exceeded. **The list position told us which was "least recently used" without any timestamps.**

**Explanation:** The HashMap provides $\mathcal{O}(1)$ access to nodes. The Doubly Linked List maintains the eviction order. Moving a node to the head of the list designates it as most recently used.

```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache: return -1
        node = self.cache[key]
        self._remove(node)
        self._insert(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        if len(self.cache) == self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
            
        new_node = Node(key, value)
        self._insert(new_node)
        self.cache[key] = new_node

    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert(self, node: Node) -> None:
        node.next = self.head.next
        node.next.prev = node
        self.head.next = node
        node.prev = self.head
# Time Complexity: O(1) for both get and put
# Space Complexity: O(Capacity)
```

* * *

**11. Maximal Rectangle in Binary Matrix**
**Specification:** Find the largest rectangle containing only `1`s in a 2D binary matrix.

**Example:** Input matrix $\rightarrow$ Output `6` (formed by the 2x3 rectangle of 1s in rows 1-2, cols 2-4):
```text
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
```

**Pattern:** Histogram Reduction + Monotonic Stack

> ⚠️ **The Two-Step Intuition: Row Histograms + Monotonic Stack**
>
> **Step 1 (Matrix $\rightarrow$ Histograms):** Process the matrix row by row. At each row, compute column heights. If `matrix[r][c] == '1'`, `heights[c] += 1`; if `'0'`, `heights[c] = 0`. Each row forms a 1D histogram.
>
> **Step 2 (Largest Rectangle in Histogram):** For any bar `K` of height `H`, how far can it stretch left and right? It stretches until it hits a **strictly shorter bar** on the left and right.
> - We maintain a stack of indices with **increasing heights**.
> - When we see a **shorter bar** at index `i`, the bar at `stack.peek()` cannot stretch right any further!
> - Pop the bar `h = heights[stack.pop()]`. Its right bound is `i`, its left bound is the new `stack.peek()`. 
> - $\text{Width} = i - \text{stack.peek()} - 1$. $\text{Area} = h \times \text{width}$.
> - A dummy bar of height `0` at `i = n` forces all remaining bars off the stack at the end.

![Maximal Rectangle & Histogram Stack](editions/python/chapters/13-optimization-dp/visuals/maximal_rectangle_histogram.png){width=85%}

**Trace-Through (Monotonic Stack for Heights `[3, 1, 3, 2, 2]`):**

| Index `i` | Height `h` | Action | Stack State | Area Calculated |
|:---------:|:----------:|:-----------------------------|:------------|:----------------|
| 0 | 3 | Push 0 | `[0]` | — |
| 1 | 1 | `1 < 3` $\rightarrow$ Pop 0 (h=3) | `[]` | `height=3, width=1` $\rightarrow$ **3** |
| 1 | 1 | Push 1 | `[1]` | — |
| 2 | 3 | Push 2 | `[1, 2]` | — |
| 3 | 2 | `2 < 3` $\rightarrow$ Pop 2 (h=3) | `[1]` | `height=3, width=3-1-1=1` $\rightarrow$ **3** |
| 3 | 2 | Push 3 | `[1, 3]` | — |
| 4 | 2 | Push 4 | `[1, 3, 4]` | — |
| 5 (sentinel) | 0 | `0 < 2` $\rightarrow$ Pop 4 (h=2) | `[1, 3]` | `height=2, width=5-3-1=1` $\rightarrow$ **2** |
| 5 (sentinel) | 0 | `0 < 2` $\rightarrow$ Pop 3 (h=2) | `[1]` | `height=2, width=5-1-1=3` $\rightarrow$ **6** ✅ |
| 5 (sentinel) | 0 | `0 < 1` $\rightarrow$ Pop 1 (h=1) | `[]` | `height=1, width=5` $\rightarrow$ **5** |

**Explanation:** We treat each row as the base of a histogram and update heights. We then run the $\mathcal{O}(N)$ "Largest Rectangle in Histogram" algorithm using a monotonic stack on each row.

```python
def maximal_rectangle(self, matrix: list[list[str]]) -> int:
    if not matrix or not matrix[0]: return 0
    cols = len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for row in matrix:
        # Update histogram heights
        for c in range(cols):
            heights[c] = heights[c] + 1 if row[c] == '1' else 0
        max_area = max(max_area, self._max_histogram(heights))
        
    return max_area

def _max_histogram(self, heights: list[int]) -> int:
    stack = []
    max_val = 0
    n = len(heights)
    
    for i in range(n + 1):
        h = 0 if i == n else heights[i]
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_val = max(max_val, height * width)
        stack.append(i)
        
    return max_val
# Time Complexity: O(R * C)
# Space Complexity: O(C)
```

* * *

**12. Word Ladder**
**Specification:** Find the shortest sequence of word mutations from `beginWord` to `endWord`, changing one letter at a time, using a dictionary.

**Example:** `begin = "hit", end = "cog", list = ["hot","dot","dog","lot","log","cog"]` $\rightarrow$ output `5`.

**Pattern:** BFS

**Explanation:** We use BFS because we want the shortest path in an unweighted graph. For each word, we generate all valid next mutations and enqueue them, tracking the level.

```python
def ladder_length(self, begin_word: str, end_word: str, word_list: list[str]) -> int:
    word_set = set(word_list)
    if end_word not in word_set: return 0
    
    from collections import deque
    queue = deque([begin_word])
    level = 1
    
    while queue:
        for _ in range(len(queue)): # Level-by-level processing
            curr = queue.popleft()
            for j in range(len(curr)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == curr[j]: continue
                    next_word = curr[:j] + c + curr[j+1:]
                    if next_word == end_word: return level + 1
                    if next_word in word_set: # remove serves as 'visited' check
                        word_set.remove(next_word)
                        queue.append(next_word)
        level += 1
        
    return 0
# Time Complexity: O(M^2 * N) where M is word length, N is number of words
# Space Complexity: O(M * N)
```

* * *

**13. Coin Change**
**Specification:** Find the minimum number of coins needed to make up a given amount.

**Example:** `coins = [1,2,5]`, `amount = 11` $\rightarrow$ output `3`.

**Pattern:** 1D DP (Unbounded Knapsack)

**Explanation:** `dp[i]` is the minimum coins needed for amount `i`. We iterate through amounts and coins, taking the min of using the coin or not: `dp[i] = min(dp[i], dp[i - coin] + 1)`.

```python
def coin_change(self, coins: list[int], amount: int) -> int:
    dp = [amount + 1] * (amount + 1) # Fill with max invalid value
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
                
    return -1 if dp[amount] > amount else dp[amount]
# Time Complexity: O(Amount * N)
# Space Complexity: O(Amount)
```

* * *

**14. House Robber**
**Specification:** Maximum money you can rob from houses where you cannot rob adjacent houses.

**Example:** `[2,7,9,3,1]` $\rightarrow$ output `12`.

**Pattern:** 1D DP with State Compression

**Explanation:** The transition is `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`. We only need to store the previous two values, saving space.

```python
def rob(self, nums: list[int]) -> int:
    if not nums: return 0
    prev1 = 0 # max so far excluding current
    prev2 = 0 # max so far including current (-2)
    
    for num in nums:
        temp = max(prev1, prev2 + num) # rob or don't rob
        prev2 = prev1
        prev1 = temp
        
    return prev1
# Time Complexity: O(N)
# Space Complexity: O(1)
```

* * *

**15. Regular Expression Matching**
**Specification:** Implement regex matching with support for `.` (any single char) and `*` (zero or more of the preceding char).

**Example:** `s = "ab", p = ".*"` $\rightarrow$ output `true`.

**Pattern:** 2D DP

**Explanation:** Complex transition logic based on whether we see a `*`. We either treat `*` as zero occurrences (`dp[i][j-2]`) or multiple occurrences (`dp[i-1][j]` if the preceding char matches).

```python
def is_match(self, s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Match empty string with patterns like a*b*
    for j in range(1, n + 1):
        if p[j - 1] == '*': dp[0][j] = dp[0][j - 2]
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1] # Single char match
            elif p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2] # Match zero times
                # If preceding char matches, match one or more times
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
                    
    return dp[m][n]
# Time Complexity: O(M * N)
# Space Complexity: O(M * N)
```

* * *

**16. Course Schedule II**
**Specification:** Return the ordering of courses you should take to finish all courses given prerequisite pairs `[course, prereq]`.

**Example:** `num = 4, prereqs = [[1,0],[2,0],[3,1],[3,2]]` $\rightarrow$ output `[0,1,2,3]`.

**Pattern:** Topological Sort (Kahn's)

**Explanation:** We count the in-degree of each course. A course with in-degree 0 has no prerequisites and can be taken. We enqueue it, take it, and decrement the in-degree of its neighbors.

```python
def find_order(self, num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    in_degree = [0] * num_courses
    adj = [[] for _ in range(num_courses)]
    
    for dest, src in prerequisites:
        adj[src].append(dest)
        in_degree[dest] += 1
        
    from collections import deque
    q = deque(i for i in range(num_courses) if in_degree[i] == 0)
    
    res = []
    while q:
        curr = q.popleft()
        res.append(curr)
        for nxt in adj[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                q.append(nxt)
                
    return res if len(res) == num_courses else [] # If not all courses taken, cycle exists
# Time Complexity: O(V + E)
# Space Complexity: O(V + E)
```

* * *

**17. Partition Equal Subset Sum**
**Specification:** Determine if an array can be partitioned into two subsets with equal sums.

**Example:** `nums = [1,5,11,5]` $\rightarrow$ output `true`.

**Pattern:** 0/1 Knapsack DP

**Explanation:** The problem translates to: "Is there a subset that sums exactly to `total_sum / 2`?" We use a 1D DP array where `dp[j]` is true if a sum `j` is achievable.

```python
def can_partition(self, nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2 != 0: return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        # Iterate backwards to avoid reusing the same element
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
            
    return dp[target]
# Time Complexity: O(N * Target)
# Space Complexity: O(Target)
```

* * *

**18. Decode Ways**
**Specification:** Given a string of digits, return the number of ways it can be decoded (`A=1`, `Z=26`).

**Example:** `s = "226"` $\rightarrow$ output `3` (BZ, VF, BBF).

**Pattern:** 1D DP

**Explanation:** Very similar to Fibonacci. The number of ways to decode up to `i` is the ways to decode up to `i-1` (if single digit valid) plus the ways to decode up to `i-2` (if two digits valid).

```python
def num_decodings(self, s: str) -> int:
    if not s or s[0] == '0': return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, n + 1):
        one_digit = int(s[i - 1:i])
        two_digits = int(s[i - 2:i])
        
        if 1 <= one_digit <= 9:
            dp[i] += dp[i - 1]
        if 10 <= two_digits <= 26:
            dp[i] += dp[i - 2]
            
    return dp[n]
# Time Complexity: O(N)
# Space Complexity: O(N) which can be optimized to O(1)
```

* * *

**19. Stock Span**
**Specification:** Design a class that calculates the stock's span (consecutive days prior where price was $\le$ today).

**Example:** `[100, 80, 60, 70, 60, 75, 85]` $\rightarrow$ output `[1, 1, 1, 2, 1, 4, 6]`.

**Pattern:** Monotonic Stack

**Explanation:** Maintain a stack of pairs `{price, span}`. If the incoming price is greater than the top of the stack, pop the stack and accumulate the span. This maintains a strictly decreasing stack.

```python
class StockSpanner:
    def __init__(self):
        # Array holds [price, span]
        self.stack = []
        
    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1] # Accumulate previous spans
        self.stack.append([price, span])
        return span
# Time Complexity: Amortized O(1) per next() call
# Space Complexity: O(N)
```

* * *

**20. Longest Increasing Subsequence**
**Specification:** Find the length of the longest strictly increasing subsequence in an array.

**Example:** `nums = [10,9,2,5,3,7,101,18]` $\rightarrow$ output `4` (`[2,3,7,101]`).

**Pattern:** DP + Binary Search

**Explanation:** We maintain an array `tails` where `tails[i]` stores the smallest tail of all increasing subsequences of length `i+1`. We binary search the position to update in `tails`.

```python
def length_of_lis(self, nums: list[int]) -> int:
    tails = [0] * len(nums)
    size = 0
    for x in nums:
        left, right = 0, size
        while left != right:
            mid = left + (right - left) // 2
            if tails[mid] < x:
                left = mid + 1
            else:
                right = mid
        tails[left] = x
        if left == size: size += 1 # Found a larger element, expand LIS
    return size
# Time Complexity: O(N log N)
# Space Complexity: O(N)
```

* * *

**21. Find Minimum in Rotated Sorted Array**
**Specification:** Return the minimum element in a rotated sorted array in $\mathcal{O}(\log N)$.

**Example:** `[3,4,5,1,2]` $\rightarrow$ output `1`.

**Pattern:** Binary Search

**Explanation:** If `nums[mid] > nums[right]`, the minimum is in the right half. Else, the minimum is in the left half (including mid).
```python
def find_min(self, nums: list[int]) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]: left = mid + 1
        else: right = mid
    return nums[left]
# Time Complexity: O(log N)
# Space Complexity: O(1)
```

* * *

**22. Kth Smallest Element in Sorted Matrix**
**Specification:** Find the K-th smallest element in a matrix where rows and columns are sorted.

**Example:** `matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8` $\rightarrow$ output `13`.

**Pattern:** Binary Search on Answer Space

**Explanation:** Binary search the value space `[min, max]`. Count how many elements are $\le$ mid. If count $< k$, `left = mid + 1`. Else `right = mid`.
```python
def kth_smallest(self, matrix: list[list[int]], k: int) -> int:
    n = len(matrix)
    left, right = matrix[0][0], matrix[n-1][n-1]
    while left < right:
        mid = left + (right - left) // 2
        count = self._count_less_equal(matrix, mid)
        if count < k: left = mid + 1
        else: right = mid
    return left

def _count_less_equal(self, matrix: list[list[int]], target: int) -> int:
    n, i, j, count = len(matrix), len(matrix) - 1, 0, 0
    while i >= 0 and j < n:
        if matrix[i][j] <= target:
            count += i + 1
            j += 1
        else:
            i -= 1
    return count
# Time Complexity: O(N log(Max - Min))
# Space Complexity: O(1)
```

* * *

**23. Jump Game II**
**Specification:** Return minimum jumps to reach the last index. You can jump up to `nums[i]` steps from index `i`.

**Example:** `[2,3,1,1,4]` $\rightarrow$ output `2`.

**Pattern:** Greedy BFS levels

**Explanation:** We maintain the farthest reach for the current jump level. When `i == currentEnd`, we must make a jump and update `currentEnd = farthest`.
```python
def jump(self, nums: list[int]) -> int:
    jumps = current_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps
# Time Complexity: O(N)
# Space Complexity: O(1)
```

* * *

**24. Unique Paths**
**Specification:** Count ways to reach bottom-right from top-left moving only right and down.

**Example:** `m = 3, n = 7` $\rightarrow$ output `28`.

**Pattern:** 2D DP

**Explanation:** `dp[i][j] = dp[i-1][j] + dp[i][j-1]`.
```python
def unique_paths(self, m: int, n: int) -> int:
    dp = [[0] * n for _ in range(m)]
    for i in range(m): dp[i][0] = 1
    for j in range(n): dp[0][j] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
# Time Complexity: O(M * N)
# Space Complexity: O(M * N) (can be optimized to O(N))
```

* * *

**25. Maximum Subarray / Kadane's Algorithm**
**Specification:** Find contiguous subarray with largest sum.

**Example:** `[-2,1,-3,4,-1,2,1,-5,4]` $\rightarrow$ output `6`.

**Pattern:** DP / Greedy

**Explanation:** At each step, either add the current element to the previous sum, or start a new subarray if the previous sum is negative.
```python
def max_sub_array(self, nums: list[int]) -> int:
    max_sum = current_sum = nums[0]
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    return max_sum
# Time Complexity: O(N)
# Space Complexity: O(1)
```

* * *

**26. Climbing Stairs**
**Specification:** Number of ways to climb `n` stairs (taking 1 or 2 steps).

**Example:** `n = 3` $\rightarrow$ output `3`.

**Pattern:** Fibonacci DP

**Explanation:** `dp[i] = dp[i-1] + dp[i-2]`.
```python
def climb_stairs(self, n: int) -> int:
    if n <= 2: return n
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
# Time Complexity: O(N)
# Space Complexity: O(1)
```

* * *

**27. Largest Rectangle in Histogram**
**Specification:** Find area of largest rectangle in histogram.

**Example:** `[2,1,5,6,2,3]` $\rightarrow$ output `10`.

**Pattern:** Monotonic Stack

**Explanation:** Stack stores indices of strictly increasing heights. Pop when a smaller height is found, calculating area using the popped height as the bottleneck.
```python
def largest_rectangle_area(self, heights: list[int]) -> int:
    stack = []
    max_area = 0
    n = len(heights)
    for i in range(n + 1):
        h = 0 if i == n else heights[i]
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area
# Time Complexity: O(N)
# Space Complexity: O(N)
```

* * *

**28. Merge K Sorted Lists**
**Specification:** Merge K sorted linked lists into one sorted list.

**Example:** `[[1,4,5],[1,3,4],[2,6]]` $\rightarrow$ output `[1,1,2,3,4,4,5,6]`.

**Pattern:** Min-Heap

**Explanation:** Put all list heads into a PriorityQueue. Extract the min, append to result, and insert the next node from the extracted list.
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def merge_k_lists(self, lists: list[ListNode]) -> ListNode:
    import heapq
    
    # Python heapq requires a way to break ties if vals are equal.
    # We can use id(node) or an index.
    pq = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(pq, (head.val, i, head))
            
    dummy = ListNode(0)
    curr = dummy
    
    while pq:
        val, i, min_node = heapq.heappop(pq)
        curr.next = min_node
        curr = curr.next
        if min_node.next:
            heapq.heappush(pq, (min_node.next.val, i, min_node.next))
            
    return dummy.next
# Time Complexity: O(N log K)
# Space Complexity: O(K)
```

* * *

**29. Longest Valid Parentheses**
**Specification:** Find length of longest valid (well-formed) parentheses substring.

**Example:** `")()())"` $\rightarrow$ output `4`.

**Pattern:** DP

**Explanation:** `dp[i]` is the length of longest valid substring ending at `i`. If `s[i] == ')'` and `s[i-1] == '('`, `dp[i] = dp[i-2] + 2`. If `s[i-1] == ')'`, match earlier part.
```python
def longest_valid_parentheses(self, s: str) -> int:
    max_len = 0
    dp = [0] * len(s)
    for i in range(1, len(s)):
        if s[i] == ')':
            if s[i - 1] == '(':
                dp[i] = (dp[i - 2] if i >= 2 else 0) + 2
            elif i - dp[i - 1] > 0 and s[i - dp[i - 1] - 1] == '(':
                dp[i] = dp[i - 1] + (dp[i - dp[i - 1] - 2] if (i - dp[i - 1]) >= 2 else 0) + 2
            max_len = max(max_len, dp[i])
    return max_len
# Time Complexity: O(N)
# Space Complexity: O(N)
```

* * *

**30. Container With Most Water**
**Specification:** Find two lines that together with x-axis forms a container holding the most water.

**Example:** `[1,8,6,2,5,4,8,3,7]` $\rightarrow$ output `49`.

**Pattern:** Two-pointer

**Explanation:** Area is `width * min(h[L], h[R])`. Move the pointer pointing to the shorter line to potentially find a taller line.
```python
def max_area(self, height: list[int]) -> int:
    max_area = 0
    left, right = 0, len(height) - 1
    while left < right:
        w = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, w * h)
        if height[left] < height[right]: left += 1
        else: right -= 1
    return max_area
# Time Complexity: O(N)
# Space Complexity: O(1)
```

* * *

## Practice Problem Bank

**31. Capacity To Ship Packages Within D Days**
**Specification:** A conveyor belt has packages that must be shipped in D days. The i-th package has weight `weights[i]`. Each day, you load the ship with packages in the order given up to the ship's max weight capacity. Return the least weight capacity of the ship.

**Example:** `weights = [1,2,3,4,5,6,7,8,9,10], D = 5` $\rightarrow$ output `15`.

**Constraints:** `1 <= D <= weights.length <= 5*10^4`

**Strategic Hint:** Use Binary Search on Answer Space (`[max(weights), sum(weights)]`).

**32. Russian Doll Envelopes**
**Specification:** Given a 2D array of envelopes `[width, height]`, you can put one inside another if both width and height of the inner are strictly smaller. Find the maximum number of envelopes you can Russian doll.

**Example:** `envelopes = [[5,4],[6,4],[6,7],[2,3]]` $\rightarrow$ output `3`.

**Constraints:** `1 <= envelopes.length <= 10^5`

**Strategic Hint:** Sort by width ASC and height DESC, then apply Longest Increasing Subsequence logic with DP + Binary Search.

**33. Course Schedule**
**Specification:** There are `numCourses` to take. Some courses have prerequisites. Determine if it is possible to finish all courses.

**Example:** `numCourses = 2, prerequisites = [[1,0],[0,1]]` $\rightarrow$ output `false`.

**Constraints:** `1 <= numCourses <= 2000`

**Strategic Hint:** Use Kahn's Algorithm for Topological Sort to detect cycles in the DAG.

**34. Next Greater Element II**
**Specification:** Given a circular integer array, return the next greater number for every element. If it doesn't exist, return -1.

**Example:** `nums = [1,2,1]` $\rightarrow$ output `[2,-1,2]`.

**Constraints:** `1 <= nums.length <= 10^4`

**Strategic Hint:** Use a Monotonic Stack and loop through the array twice to simulate circularity.

**35. Koko Eating Bananas**
**Specification:** Koko wants to eat all bananas in `H` hours. Return her minimum eating speed `K` bananas per hour.

**Example:** `piles = [3,6,7,11], H = 8` $\rightarrow$ output `4`.

**Constraints:** `1 <= piles.length <= 10^4`

**Strategic Hint:** Use Binary Search on Answer Space with bounds `[1, max(piles)]`.

**36. Palindrome Partitioning II**
**Specification:** Given a string, partition it such that every substring is a palindrome. Return the minimum cuts needed.

**Example:** `s = "aab"` $\rightarrow$ output `1` ("aa", "b").

**Constraints:** `1 <= s.length <= 2000`

**Strategic Hint:** 1D DP where `dp[i]` is min cuts for suffix `s[i..n]`. Expand from centers to find palindromes.

**37. Search a 2D Matrix**
**Specification:** Write an efficient algorithm that searches for a value in an `m x n` matrix. Each row is sorted from left to right, and the first integer of each row is greater than the last integer of the previous row.

**Example:** `matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3` $\rightarrow$ output `true`.

**Constraints:** `m == matrix.length, n == matrix[i].length, 1 <= m, n <= 100`

**Strategic Hint:** Treat the 2D matrix as a flat 1D array and use standard Binary Search.

**38. Minimum Path Sum**
**Specification:** Given a `m x n` grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

**Example:** `grid = [[1,3,1],[1,5,1],[4,2,1]]` $\rightarrow$ output `7`.

**Constraints:** `1 <= m, n <= 200`

**Strategic Hint:** 2D DP modifying the grid in-place: `grid[i][j] += min(grid[i-1][j], grid[i][j-1])`.

**39. Perfect Squares**
**Specification:** Given an integer `n`, return the least number of perfect square numbers that sum to `n`.

**Example:** `n = 12` $\rightarrow$ output `3` (4 + 4 + 4).

**Constraints:** `1 <= n <= 10^4`

**Strategic Hint:** 1D DP similar to Coin Change where coins are perfect squares up to `sqrt(n)`.

**40. Combination Sum IV**
**Specification:** Given an array of distinct integers and a target, return the number of possible combinations that add up to target.

**Example:** `nums = [1,2,3], target = 4` $\rightarrow$ output `7`.

**Constraints:** `1 <= nums.length <= 200`

**Strategic Hint:** 1D DP where `dp[i] += dp[i - num]` for all valid `num` in `nums`.

**41. Split Array Largest Sum**
**Specification:** Split an array into `k` non-empty contiguous subarrays such that the largest sum among these subarrays is minimized.

**Example:** `nums = [7,2,5,10,8], k = 2` $\rightarrow$ output `18`.

**Constraints:** `1 <= nums.length <= 1000`

**Strategic Hint:** Binary Search on Answer Space where `left = max(nums)` and `right = sum(nums)`.

**42. Trapping Rain Water II**
**Specification:** Given an `m x n` integer matrix of heights, return the volume of water it can trap after raining.

**Example:** `heightMap = [[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]` $\rightarrow$ output `4`.

**Constraints:** `1 <= m, n <= 200`

**Strategic Hint:** Use a Min-Heap starting with boundary cells and simulate a rising water level using BFS.

**43. Maximize Distance to Closest Person**
**Specification:** In a row of seats, 1 means occupied, 0 means empty. Find a seat to maximize distance to the closest person.

**Example:** `seats = [1,0,0,0,1,0,1]` $\rightarrow$ output `2`.

**Constraints:** `2 <= seats.length <= 20000`

**Strategic Hint:** Two-pointer approach counting zeros between ones, with edge cases for edges of the array.

**44. Minimum Window Substring**
**Specification:** Given two strings `s` and `t`, return the minimum window substring of `s` such that every character in `t` is included in the window.

**Example:** `s = "ADOBECODEBANC", t = "ABC"` $\rightarrow$ output `"BANC"`.

**Constraints:** `1 <= s.length, t.length <= 10^5`

**Strategic Hint:** Sliding Window Two-Pointer with a character frequency map to track fulfillment.

**45. Largest Divisible Subset**
**Specification:** Given a set of distinct positive integers, find the largest subset such that every pair `(Si, Sj)` satisfies `Si % Sj == 0` or `Sj % Si == 0`.

**Example:** `nums = [1,2,3]` $\rightarrow$ output `[1,2]` or `[1,3]`.

**Constraints:** `1 <= nums.length <= 1000`

**Strategic Hint:** Sort first. Use 1D DP `dp[i]` representing the max subset ending with `nums[i]`, tracking parents to reconstruct.

**46. Interleaving String**
**Specification:** Given strings `s1`, `s2`, and `s3`, find whether `s3` is formed by an interleaving of `s1` and `s2`.

**Example:** `s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"` $\rightarrow$ output `true`.

**Constraints:** `0 <= s1.length, s2.length <= 100`

**Strategic Hint:** 2D DP where `dp[i][j]` means if `s3.substring(0, i+j)` can be formed by `s1.substring(0, i)` and `s2.substring(0, j)`.

**47. Shortest Path in Binary Matrix**
**Specification:** Find the shortest clear path from top-left to bottom-right in a grid.

**Example:** `grid = [[0,1],[1,0]]` $\rightarrow$ output `2`.

**Constraints:** `1 <= n <= 100`

**Strategic Hint:** BFS with Level Tracking since we want the shortest path in an unweighted grid with 8 directions.

**48. Task Scheduler**
**Specification:** Given an array of CPU tasks and a cooldown `n`, return the least number of intervals needed to finish all tasks.

**Example:** `tasks = ["A","A","A","B","B","B"], n = 2` $\rightarrow$ output `8`.

**Constraints:** `1 <= tasks.length <= 10^4`

**Strategic Hint:** Greedy approach or Math formula based on the frequency of the most common task.

**49. 132 Pattern**
**Specification:** Given an array of integers, find if there is a 132 pattern (`i < j < k` and `nums[i] < nums[k] < nums[j]`).

**Example:** `nums = [3,1,4,2]` $\rightarrow$ output `true`.

**Constraints:** `1 <= nums.length <= 2 * 10^5`

**Strategic Hint:** Traverse backwards maintaining a Monotonic Stack to find the `nums[k]` value while keeping track of max `nums[k]`.

**50. Minimum Size Subarray Sum**
**Specification:** Return the minimal length of a contiguous subarray of which the sum is greater than or equal to `target`.

**Example:** `target = 7, nums = [2,3,1,2,4,3]` $\rightarrow$ output `2`.

**Constraints:** `1 <= nums.length <= 10^5`

**Strategic Hint:** Sliding Window Two-Pointer. Expand right until sum is met, then contract left to minimize.

**51. Frog Jump**
**Specification:** A frog crosses a river with stones. If the last jump was `k` units, the next jump must be `k-1`, `k`, or `k+1`. Can it reach the last stone?

**Example:** `stones = [0,1,3,5,6,8,12,17]` $\rightarrow$ output `true`.

**Constraints:** `2 <= stones.length <= 2000`

**Strategic Hint:** 2D DP or Hash Map of sets where `map.get(stone)` contains all possible jump lengths that reached this stone.

**52. Rotting Oranges**
**Specification:** Every minute, any fresh orange adjacent to a rotten one becomes rotten. Return minimum minutes until no cell has a fresh orange.

**Example:** `grid = [[2,1,1],[1,1,0],[0,1,1]]` $\rightarrow$ output `4`.

**Constraints:** `1 <= m, n <= 10`

**Strategic Hint:** Multi-source BFS starting with all rotten oranges in the queue at minute 0.

**53. Word Break**
**Specification:** Given a string and a dictionary, determine if the string can be segmented into a space-separated sequence of dictionary words.

**Example:** `s = "leetcode", wordDict = ["leet", "code"]` $\rightarrow$ output `true`.

**Constraints:** `1 <= s.length <= 300`

**Strategic Hint:** 1D DP `dp[i]` is true if `s.substring(0, i)` can be broken down.

**54. Max Consecutive Ones III**
**Specification:** Given a binary array and an integer `k`, return the max number of consecutive `1`s if you can flip at most `k` `0`s.

**Example:** `nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2` $\rightarrow$ output `6`.

**Constraints:** `1 <= nums.length <= 10^5`

**Strategic Hint:** Sliding Window. The window can contain at most `k` zeros.

**55. Jump Game**
**Specification:** Determine if you can reach the last index starting from the first.

**Example:** `nums = [2,3,1,1,4]` $\rightarrow$ output `true`.

**Constraints:** `1 <= nums.length <= 10^4`

**Strategic Hint:** Greedy approach tracking the maximum reachable index `maxReach = max(maxReach, i + nums[i])`.

**56. Wiggle Subsequence**
**Specification:** Find length of longest subsequence that alternates between strictly increasing and strictly decreasing.

**Example:** `nums = [1,7,4,9,2,5]` $\rightarrow$ output `6`.

**Constraints:** `1 <= nums.length <= 1000`

**Strategic Hint:** 1D DP tracking the longest ending with an "up" transition and a "down" transition.

**57. Predict the Winner**
**Specification:** Two players pick numbers from either end of an array. Determine if Player 1 can guarantee a win.

**Example:** `nums = [1, 5, 2]` $\rightarrow$ output `false`.

**Constraints:** `1 <= nums.length <= 20`

**Strategic Hint:** Interval DP `dp[i][j]` representing the max score difference a player can achieve taking from `[i, j]`.

**58. Find K-th Smallest Pair Distance**
**Specification:** Find the K-th smallest distance among all pairs `(nums[i], nums[j])` in an array.

**Example:** `nums = [1,3,1], k = 1` $\rightarrow$ output `0`.

**Constraints:** `n <= 10^4`

**Strategic Hint:** Binary Search on Answer Space with sliding window to count pairs with distance $\le$ `mid`.

**59. Cheapest Flights Within K Stops**
**Specification:** Find the cheapest price from `src` to `dst` with up to `k` stops.

**Example:** `n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1` $\rightarrow$ output `200`.

**Constraints:** `1 <= n <= 100`

**Strategic Hint:** Bellman-Ford or BFS with level tracking (up to K levels) tracking minimum costs.

**60. Remove K Digits**
**Specification:** Given a string representing a non-negative integer, remove `k` digits to form the smallest possible integer.

**Example:** `num = "1432219", k = 3` $\rightarrow$ output `"1219"`.

**Constraints:** `1 <= num.length <= 10^5`

**Strategic Hint:** Monotonic Stack (increasing). Pop strictly larger digits while `k > 0`.


# Mastering Problem Decomposition: The Capstone

> *"Every problem you will ever face in a technical assessment is a composition of patterns you already know. The art is in the seeing."*

---

## From Patterns to Synthesis

Throughout the preceding chapters, you have meticulously studied and mastered the 24 Canonical Patterns. You understand Sliding Windows, Monotonic Stacks, Prefix Sums, and Topological Sorts in isolation. However, demonstrating proficiency in individual patterns is merely the baseline expectation. To excel in elite technical assessments, you must transition from pattern recognition to pattern synthesis.

Real assessment problems—especially those found in equal-weight peer assessments and single-deep-problem architectural interviews—rarely map cleanly to a single, textbook pattern. Instead, they are complex compositions requiring the seamless integration of two, three, or even more distinct patterns. The complexity lies not in the patterns themselves, but in their orchestration.

This capstone chapter is your synthesis training ground. It is designed to elevate your analytical capabilities, teaching you how to systematically dissect intricate problems, identify the interlocking sub-components, and construct robust, optimal solutions through the deliberate composition of the canonical patterns.

## The Cognitive Derivation Process

When you encounter a truly novel problem—one that doesn't immediately map to a known pattern—follow this derivation process:

1. **Generate the smallest non-trivial example** (n=3 or n=4) and solve it BY HAND on paper. Track what your brain does.
2. **Identify the decision you make at each step.** Are you choosing the maximum? The nearest? The first valid? This reveals the algorithm class (greedy, search, optimization).
3. **Ask: "What information do I need from the past, and what do I need about the future?"** If you need past information → prefix arrays or DP. If you need future information → suffix arrays, reverse iteration, or monotonic stacks.
4. **Ask: "Can I solve a smaller version of this problem and combine the results?"** If yes → divide and conquer or recursive DP.
5. **Ask: "Does the order of processing matter?"** If no → consider sorting first. If yes → the original order is a constraint you must preserve.

This is NOT pattern matching. This is the fundamental analytical skill that GENERATES pattern recognition.

## The Problem Analysis Canvas

To navigate complex problem spaces effectively, we must formalize the 5-step decomposition framework introduced in Chapter 2 into a rigorous, repeatable structure. The **Problem Analysis Canvas** is a mental and textual template you should apply to every problem you encounter. In an assessment setting, writing this canvas out in comments serves as both your architectural blueprint and a clear signal of your structured thinking to evaluators.

### The Canvas

| Analysis Phase | Your Response |
|---|---|
| **Restatement** | [What is actually being asked, stripped of narrative?] |
| **Inputs** | [Types, ranges, constraints, formats] |
| **Outputs** | [Expected return type and format] |
| **Constraints** | [N range → target time/space complexity] |
| **Edge Cases** | [Empty inputs, single elements, identical elements, overflows] |
| **Sub-Problems** | [Break the core problem into 2-4 independent components] |
| **Pattern Mapping** | [Which PAT-XX resolves each sub-problem?] |
| **Complexity Target** | [Final Time $O(\dots)$ and Space $O(\dots)$ bounds] |
| **Approach** | [Pseudocode or high-level bulleted steps] |

By rigidly adhering to this canvas, you eliminate the panic of the blank screen and replace it with a systematic diagnostic process.

![Problem Analysis Canvas — Structured Decomposition Framework](editions/python/chapters/14-mastering-decomposition/visuals/problem_analysis_canvas.jpg){width=85%}

## Decomposition Walkthroughs

The following sections provide comprehensive step-by-step decomposition analyses across varying levels of complexity. We will analyze the problems, deconstruct them using the canvas methodology, and map them to our canonical patterns.

### Tier 1: Single-Pattern Problems (Warm-Up)

Tier 1 problems form the foundation of technical assessments. They are characterized by a direct, one-to-one mapping with a specific pattern. The challenge here is swift recognition and flawless execution.

#### Example 1: The Target Sum Search
**Problem:** Given a sorted array of integers, determine if any two distinct numbers sum to a specific target value.

**Analysis:**
*   **Restatement:** Find a pair in a sorted array that equals a target sum.
*   **Constraints:** Array is sorted. We need a solution better than $O(N^2)$.
*   **Sub-Problems:** We need to efficiently search for a complement value for each element.
*   **Pattern Mapping:** The array is sorted, and we are looking for a pair. This immediately triggers **[PAT-06] Converging Two-Pointers**.
*   **Approach:** Place pointers at the start and end. If the sum is too large, decrement the right pointer. If too small, increment the left. Time $O(N)$, Space $O(1)$.

#### Example 2: First Unique Character
**Problem:** Find the first non-repeating character in a string and return its index.

**Analysis:**
*   **Restatement:** Identify the earliest character in a sequence that appears exactly once.
*   **Sub-Problems:** 1. Count occurrences of all characters. 2. Find the first character with a count of one.
*   **Pattern Mapping:** Counting occurrences over a finite set (characters) maps to **[PAT-01] Direct Indexing & Frequency Buckets** (or Hash Map).
*   **Approach:** One pass to populate frequency array. Second pass over the string to check frequencies and return the first index where frequency is 1. Time $O(N)$, Space $O(1)$ (bounded by alphabet size).

#### Example 3: In-Place Array Rotation
**Problem:** Rotate an array to the right by $k$ positions, modifying the array in-place.

**Analysis:**
*   **Restatement:** Shift all elements right by $k$, wrapping around, without using extra $O(N)$ space.
*   **Sub-Problems:** Shifting elements in-place without a buffer requires structured swaps.
*   **Pattern Mapping:** Modifying array order in-place often utilizes **[PAT-02] In-Place Mutation & Two-Pointer Compaction**.
*   **Approach:** Reverse the entire array. Reverse the first $k$ elements. Reverse the remaining $N-k$ elements. Time $O(N)$, Space $O(1)$.

#### Example 4: The Missing Sequence
**Problem:** Find the missing number in an array containing $n$ distinct numbers taken from the range $0$ to $n$.

**Analysis:**
*   **Restatement:** Identify the single absent integer in a contiguous sequence.
*   **Pattern Mapping:** Comparing a sequence to an expected aggregate relies on mathematical invariants (e.g., Gauss's sum formula or XOR accumulation).
*   **Approach:** Calculate the expected sum using $n(n+1)/2$. Subtract the actual sum of the array. The difference is the missing number. Time $O(N)$, Space $O(1)$.

### Tier 2: Dual-Pattern Compositions (Assessment Core)

Tier 2 problems are the standard for rigorous technical screens. They cannot be solved by applying a single pattern in isolation; they require identifying two overlapping structures and combining them harmoniously.

#### Example 1: Distinct Substrings
**Problem:** Find the length of the longest substring containing at most $K$ distinct characters.

**Analysis:**
*   **Restatement:** Find the maximum contiguous subarray length bounded by a character diversity constraint.
*   **Sub-Problems:** 1. Iterate over all possible contiguous subarrays efficiently. 2. Track the number of distinct characters currently in view.
*   **Pattern Mapping:** "Longest substring" and "contiguous" strongly imply **[PAT-04] Dynamic Sliding Window (Variable Size)**. "Tracking distinct characters" implies **[PAT-01] Direct Indexing & Frequency Buckets**.
*   **Approach:** Use a sliding window with a left and right pointer. Expand right, updating a frequency map. If the map size exceeds $K$, increment left, decrementing frequencies until the map size is valid again. Keep track of the maximum window size.

#### Example 2: The Kth Largest
**Problem:** Find the Kth largest element in an unsorted array efficiently without sorting the entire array.

**Analysis:**
*   **Restatement:** Locate a specific rank-order element in unsorted data.
*   **Constraints:** Sorting takes $O(N \log N)$. Can we achieve $O(N)$ average time?
*   **Sub-Problems:** 1. Partition the array around a pivot. 2. Decide which partition to explore based on the pivot's final index.
*   **Pattern Mapping:** Partitioning logic maps to QuickSelect, which is a variation of **[PAT-11] Binary Search on Solution Range**, combined with **[PAT-02] In-Place Mutation & Two-Pointer Compaction**. Alternatively, managing the top K elements maps to **[PAT-25] Priority Queue / Min-Max Heap**.
*   **Approach (Heap):** Maintain a Min-Heap of size K. Iterate the array; push elements. If heap exceeds K, pop. The root of the heap is the Kth largest. Time $O(N \log K)$.

#### Example 3: Merging Multiple Streams
**Problem:** Merge $K$ sorted linked lists into a single sorted linked list.

**Analysis:**
*   **Restatement:** Combine multiple ordered sequences into one ordered sequence.
*   **Sub-Problems:** 1. Continuously identify the smallest current element across $K$ heads. 2. Append to a new list and advance the corresponding pointer.
*   **Pattern Mapping:** Finding the minimum among $K$ dynamic candidates is exactly what a **[PAT-25] Priority Queue / Min-Max Heap** is for. Processing them sequentially visually resembles **[PAT-13] Level-by-Level BFS Wavefront**.
*   **Approach:** Push the head of each list into a Min-Heap. While heap is not empty, pop the smallest node, append to result, and if the popped node has a `next`, push `next` into the heap.

#### Example 4: Substring Anagrams
**Problem:** Given a text and a pattern string, find all starting indices in the text where the substring is an anagram of the pattern.

**Analysis:**
*   **Restatement:** Find all contiguous subarrays of length $P$ in text that have the exact same character frequencies as the pattern.
*   **Sub-Problems:** 1. Maintain a rolling view of length $P$. 2. Compare the frequency signature of the view against the pattern's signature.
*   **Pattern Mapping:** "Rolling view of fixed length" dictates a **[PAT-05] Fixed-Size Monotonic Deque Window** (or simply a fixed-size window approach). "Frequency signature" maps to **[PAT-01] Direct Indexing & Frequency Buckets**.
*   **Approach:** Compute the target frequency array for the pattern. Use a sliding window of length $P$ over the text, maintaining a rolling frequency array. Compare the arrays at each step. Time $O(N)$.

#### Example 5: Course Prerequisites
**Problem:** Given $N$ courses and a list of prerequisite pairs, determine if it is possible to finish all courses.

**Analysis:**
*   **Restatement:** Detect if a directed graph of dependencies contains any cycles.
*   **Sub-Problems:** 1. Model the dependencies as a graph. 2. Traverse the graph to ensure all nodes can be visited without encountering back-edges.
*   **Pattern Mapping:** Dependency resolution strictly maps to **[PAT-16] Topological Sort (Kahn's & DFS)**. The traversal mechanism is inherently Level-by-Level BFS.
*   **Approach:** Build an adjacency list and an in-degree array. Push nodes with in-degree 0 to a queue. Process BFS, decrementing in-degrees of neighbors. If a neighbor hits 0, queue it. If the count of processed nodes equals $N$, no cycles exist.

### Tier 3: Multi-Pattern Synthesis (Capstone Challenges)

Tier 3 problems represent the apex of algorithmic assessments. These problems require deep architectural insight, combining three or more patterns, or employing a pattern in a highly unconventional manner.

#### Example 1: The Word Ladder
**Problem:** Given a start word, an end word, and a dictionary, find the length of the shortest transformation sequence from start to end, where only one letter can be changed at a time.

**Analysis:**
*   **Restatement:** Find the shortest path between two nodes in an unweighted graph where edges represent single-character mutations.
*   **Pattern Mapping:** "Shortest path in unweighted graph" guarantees **[PAT-13] Level-by-Level BFS Wavefront**. Generating valid edges requires character substitution logic. To optimize, we can use **[PAT-14] Multi-Source BFS Parallel Spreading** or Bidirectional BFS.
*   **Approach:** Treat words as nodes. For the current word, substitute each character with 'a'-'z' to find valid neighbors in the dictionary. Enqueue valid, unseen neighbors. BFS guarantees the first time we reach the end word is the shortest path.

#### Example 2: Trapping Rainwater
**Problem:** Given an array representing building heights, calculate the total volume of trapped rainwater.

**Analysis:** (As seen in Chapter 2, but expanded)
*   **Restatement:** Water at index $i$ is $\min(\text{max\_left}, \text{max\_right}) - \text{height}[i]$.
*   **Pattern Mapping:** We need boundary maximums. This can be solved via **[PAT-03] Prefix Sums & Range Query Invariants** (Time $O(N)$, Space $O(N)$). To optimize space, we synthesize it with **[PAT-06] Converging Two-Pointers** (Time $O(N)$, Space $O(1)$).
*   **Approach (Two-Pointer):** Maintain `left`, `right`, `left_max`, `right_max`. Move the pointer corresponding to the smaller maximum, safely calculating trapped water as we guarantee the other side is bounded by a larger height.

#### Example 3: Largest Rectangle in Histogram
**Problem:** Find the area of the largest rectangle that can be formed within a histogram.

**Analysis:**
*   **Restatement:** For every bar, find the maximum contiguous width where all bars are at least as tall as the current bar. Area = height * width.
*   **Pattern Mapping:** We need to find the "next smaller element" to the left and right to define the width boundaries. This is the textbook definition of a **[PAT-09] Monotonic Stack ("The Waiting Room")**.
*   **Approach:** Maintain an increasing monotonic stack of indices. When encountering a shorter bar, pop from the stack. The popped bar is the height. The current index is the right boundary; the new top of the stack is the left boundary. Synthesize with sentinel logic (append a 0 height at the end) to flush the stack efficiently.

#### Example 4: Minimum Window Substring
**Problem:** Find the minimum contiguous substring in $S$ that contains all characters of $T$ in any order.

**Analysis:**
*   **Restatement:** Find the shortest subarray that satisfies a strict subset frequency requirement.
*   **Pattern Mapping:** "Shortest contiguous substring" → **[PAT-04] Dynamic Sliding Window (Variable Size)**. "Contains all characters" → **[PAT-01] Direct Indexing & Frequency Buckets**. Furthermore, we need a **Convergence Condition** to know when the window is valid without iterating the map every time.
*   **Approach:** Maintain a `target_map` for $T$ and a `window_map`. Use a `matched_chars` integer to track how many unique characters in $T$ have their frequency met in the window. Expand right. When `matched_chars == target_map.size()`, the window is valid. Record length, then shrink left until it becomes invalid.

#### Example 5: Median of Two Sorted Arrays
**Problem:** Find the median of two sorted arrays of different lengths in $O(\log(M+N))$ time.

**Analysis:**
*   **Restatement:** Partition two sorted arrays such that the left halves contain the smaller half of the combined elements, and the right halves contain the larger half.
*   **Pattern Mapping:** The $O(\log)$ constraint on sorted arrays demands **[PAT-10] Monotonic Partition Binary Search**. We are binary searching the partition index of the smaller array.
*   **Approach:** Binary search on the smaller array to find partition $X$. The partition $Y$ in the larger array is determined by the total required elements in the left half. Check if `max(left_X, left_Y) <= min(right_X, right_Y)`. If true, median is found. If `left_X > right_Y`, move partition $X$ left.

#### Example 6: Bursting Balloons
**Problem:** Given $N$ balloons with values, bursting balloon $i$ yields `nums[i-1] * nums[i] * nums[i+1]` coins. Find the maximum coins obtainable by bursting all balloons.

**Analysis:**
*   **Restatement:** Find the optimal sequence of dependent operations that maximizes a cumulative score.
*   **Pattern Mapping:** The outcome of bursting a balloon depends on which balloons are left. This is overlapping subproblems typically solved using **[PAT-21] 2D Grid Path Optimization** concepts adapted for intervals (Interval DP). The synthesis secret here is **Reverse Thinking**: instead of choosing which balloon to burst first, choose which balloon to burst *last* in the interval.
*   **Approach:** DP state: $dp[i][j]$ is max coins obtained from bursting balloons between index $i$ and $j$ exclusive. Iterate over interval lengths, then start points. For each interval, guess which balloon $k$ is the *last* to burst. Transition: $dp[i][j] = \max(dp[i][j], dp[i][k] + dp[k][j] + \text{nums}[i] \times \text{nums}[k] \times \text{nums}[j])$.

## The Pattern Recognition Decision Tree (Expanded)

![Pattern Selection Decision Matrix](editions/python/chapters/14-mastering-decomposition/visuals/decomposition_decision.jpg){width=85%}

To facilitate rapid decomposition during an assessment, utilize this expanded diagnostic decision tree. When analyzing a problem, ask yourself these guiding questions in sequence:

1.  **What is the primary data structure?**
    *   *Array/String:* Sequential patterns (Pointers, Windows, Prefix Arrays, Monotonic Stacks).
    *   *Matrix/Grid:* 2D traversal (BFS/DFS), Dynamic Programming.
    *   *Graph:* Connectivity, Shortest Path, Topological Sort.
    *   *Tree:* Recursion, Level-Order traversal.
    *   *LinkedList:* Fast/Slow Pointers, In-place reversal.

2.  **What is the query type?**
    *   *Search/Find:* Binary Search, Hash Maps.
    *   *Count/Frequency:* Hash Maps, Arrays as Maps.
    *   *Optimize (Max/Min):* Greedy, Dynamic Programming, Binary Search on Answer.
    *   *Transform:* In-place swaps, Reversals.
    *   *Validate (True/False):* Two-Pointers, Stack (matching).

3.  **What are the constraints?**
    *   $N \le 20 \dots 100$: Backtracking, $O(N^3)$, Brute Force often acceptable.
    *   $N \le 10^4$: $O(N^2)$ might pass, but $O(N \log N)$ is expected.
    *   $N \le 10^5 \dots 10^6$: $O(N \log N)$ or strictly $O(N)$ required. Hash Maps, Sliding Windows, Two Pointers.
    *   $N \ge 10^9$: $O(\log N)$ or $O(1)$ required. Binary Search, Math formulas.

4.  **Is ordering important?**
    *   *Sorted:* Binary Search family, Two-Pointer Converging.
    *   *Unsorted (but order matters):* Sliding Window, Monotonic Stack.
    *   *Unsorted (order doesn't matter):* Hash Maps, Sorting as a preprocessing step.

5.  **Does it involve a window or contiguous subarray?**
    *   Fixed size → Fixed Sliding Window.
    *   Variable size with constraint → Dynamic Sliding Window.

6.  **Does it ask for 'next greater/smaller' elements?**
    *   Immediately points to Monotonic Stack.

7.  **Does it have overlapping subproblems or ask for combinations?**
    *   Optimization/Counting over subsets → Dynamic Programming family.

8.  **Does it involve connectivity or paths?**
    *   Shortest path unweighted → BFS.
    *   Dependencies/Prerequisites → Topological Sort.
    *   Component grouping → Union-Find or DFS.

## Common Decomposition Mistakes

Even with a structured framework, engineers often fall victim to specific decomposition anti-patterns under pressure. Be vigilant against these errors:

*   **Jumping to Code Without Analysis:** The most fatal error. Writing code before the canvas is complete leads to structural dead-ends and unrecoverable bugs.
*   **Over-Decomposing:** Breaking a simple problem into too many abstract layers. If a sub-problem requires only three lines of logic, it does not need a helper function or a complex object model. Keep it localized.
*   **Pattern Forcing:** Attempting to forcefully map a problem to a familiar pattern (e.g., trying to use Dynamic Programming when a simple Greedy approach works). Let the constraints dictate the pattern, not your preference.
*   **Ignoring Constraints:** Designing an elegant $O(N^2)$ solution when $N = 10^5$. Always validate your target complexity against the input constraints *before* committing to a pattern.
*   **Premature Optimization:** Trying to write the perfect $O(N)$ $O(1)$ space solution immediately. It is almost always better to articulate a correct $O(N^2)$ approach first, guarantee correctness conceptually, and then optimize it by swapping sub-pattern implementations.

## Practice Exercises

Apply the Problem Analysis Canvas to the following 15 problem statements. Do not write code. Your goal is strictly to identify the constraints, decompose the problem, and map the appropriate patterns.

1.  Given a matrix of 1s (land) and 0s (water), count the number of islands. *(Hint: Graph Traversal)*
2.  Find the maximum sum of any contiguous subarray of size $k$. *(Hint: PAT-05)*
3.  Determine if a string has all unique characters without using extra data structures. *(Hint: Sorting or Bit Manipulation)*
4.  Given an array of intervals, merge all overlapping intervals. *(Hint: Sorting + Linear Scan)*
5.  Find the lowest common ancestor of two nodes in a Binary Search Tree. *(Hint: BST property + Traversal)*
6.  Serialize and deserialize a binary tree. *(Hint: Pre-order or Level-order traversal)*
7.  Given a list of strings, group the anagrams together. *(Hint: String Signature + Hash Map)*
8.  Implement a data structure that supports insert, delete, and getRandom in $O(1)$ time. *(Hint: Array + Hash Map synthesis)*
9.  Find the length of the longest strictly increasing subsequence in an array. *(Hint: DP or Binary Search Synthesis)*
10. Given a directed graph, find the shortest path from a source to all other nodes where edges have positive weights. *(Hint: Dijkstra's Algorithm)*
11. Check if a binary tree is perfectly balanced. *(Hint: Post-order traversal)*
12. Given a string, find the longest palindromic substring. *(Hint: Expand around center or DP)*
13. Search for a target value in a 2D matrix where rows and columns are sorted. *(Hint: Specialized Two-Pointer from a corner)*
14. Calculate the edit distance between two strings. *(Hint: 2D Dynamic Programming)*
15. Find all valid combinations of $k$ numbers that sum up to $n$. *(Hint: Backtracking)*

> ⭐ **STAR Moment: The Synthesis Mindset**
>
> The engineers who consistently score in the top percentile on technical assessments are not the ones who have memorized the most solutions. They are the ones who can see the hidden structure in novel problems. Every new problem is a remix of patterns you already know. Train your eyes to see the composition, and no assessment will ever surprise you.


# 20 Timed Algorithmic Mock Assessment Sets

## How to Use This Chapter

This chapter provides 20 full, four-question exam mock sets (80 problems total) modeled after the common standardized coding assessment format. Each set is designed to simulate a rigorous timed assessment environment. The problems follow a standard difficulty curve: the first question tests basic implementation and traversal (Easy, 5-8 minutes), the second focuses on 2D matrices and simulation (Medium, 12-15 minutes), the third requires algorithmic pattern recognition like HashMaps or sliding windows (Medium-Hard, 18-20 minutes), and the fourth challenges you with dynamic programming, graphs, or advanced data structures (Hard, 20-25 minutes).

To get the most out of these mock assessments, strictly time yourself. Set a timer for 70 minutes (or adjust to match your target assessment format) and attempt all four questions in order. Do not look up syntax or external resources. If you get stuck on the third or fourth question, practice timeboxing: move on and secure partial credit where possible. For equal-weight assessment formats, treat all four questions as having equal priority and allocate approximately 15-18 minutes per question. After time expires, review your performance. Use the provided hints to guide your post-assessment study sessions, identifying which specific patterns (e.g., sliding window, BFS, monotonic stack) require further review.

![Assessment Pacing Strategy and Time Allocation](editions/python/chapters/15-mock-assessment-sets/visuals/pacing_strategy.jpg){width=85%}

Remember, there is no code in this chapter—this is your practice arena. Read the specifications, analyze the test cases, check the constraints, and write your own optimal solutions.

* * *

## Set 1: Warm-Up Fundamentals

* **Q1 (Easy): Vowel Starting Words**
  * *Specification:* Given a string of text containing words separated by single spaces, calculate the total number of words that begin with a vowel. Vowels are defined as 'a', 'e', 'i', 'o', and 'u', and the check should be case-insensitive. Ignore any punctuation, assuming the string consists only of alphabetical characters and spaces. Return the final integer count of qualifying words.
  * *Sample Test Case:* Input: `"Apple banana Orange umbrella"` -> Output: `3`.
  * *Constraints:* String length $1 \le L \le 10^5$.
  * *Hint:* Use standard string splitting to isolate words, then check the first character of each token against a predefined set of vowels.

* **Q2 (Medium): Rotate Rectangular Image**
  * *Specification:* You are given an $M \times N$ 2D matrix representing an image, where each cell holds a pixel value. Your task is to rotate the image 90 degrees clockwise. Unlike square matrix rotation, this matrix is rectangular, meaning the dimensions of the resulting matrix will swap to $N \times M$. You must allocate a new matrix to hold the rotated values and populate it correctly.
  * *Sample Test Case:* Input: `[[1, 2, 3], [4, 5, 6]]` -> Output: `[[4, 1], [5, 2], [6, 3]]`.
  * *Constraints:* $1 \le M, N \le 1000$.
  * *Hint:* The element at `matrix[r][c]` in the original matrix moves to `new_matrix[c][M - 1 - r]` in the rotated matrix.

* **Q3 (Medium-Hard): K-Frequency Substring**
  * *Specification:* Given a string and an integer K, find the length of the longest contiguous substring where no character appears more than K times. You must process the string and keep track of character frequencies dynamically. If the frequency of any character exceeds K, you must shrink the valid sequence until the condition is met again. Return the maximum length observed.
  * *Sample Test Case:* Input: `s = "abaccc", K = 2` -> Output: `4` (The substring "abac").
  * *Constraints:* String length $1 \le L \le 10^5$, $1 \le K \le L$.
  * *Hint:* Use a sliding window approach with two pointers and a HashMap or frequency array to track character counts within the current window.

* **Q4 (Hard): Largest Rectangular Area**
  * *Specification:* You are given an array of non-negative integers representing the heights of adjacent buildings, where each building has a width of 1 unit. You need to calculate the area of the largest rectangle that can be formed within the bounds of these buildings. The rectangle must be completely contained within the histograms. Return the maximum possible area.
  * *Sample Test Case:* Input: `[2, 1, 5, 6, 2, 3]` -> Output: `10` (Formed by heights 5 and 6).
  * *Constraints:* Array length $1 \le N \le 10^5$, building heights $0 \le H \le 10^4$.
  * *Hint:* Utilize a monotonic increasing stack to keep track of building indices, calculating areas when a drop in height is encountered.

* * *

## Set 2: Timed Mock Assessment 2

* **Q1 (Easy): Array Prefix Sum**
  * *Specification:* Given an array, calculate its running sum in place.
  * *Sample Test Case:* Input: `[1,2,3] -> [1,3,6]`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* [PAT-03] Prefix Sums

* **Q2 (Medium): Prefix Sum Range**
  * *Specification:* Process range sum queries on an array quickly.
  * *Sample Test Case:* Input: `[1,2,3], query(0,2) -> 6`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* [PAT-03] Prefix array

* **Q3 (Medium-Hard): BFS Shortest Path**
  * *Specification:* Find the shortest path to exit a grid maze.
  * *Sample Test Case:* Input: `grid -> 4 steps`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-13] BFS Wavefront

* **Q4 (Hard): Dijkstra Shortest**
  * *Specification:* Find network delay time for a signal to reach all nodes.
  * *Sample Test Case:* Input: `nodes=4, edges -> 2`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-18] Dijkstra Priority Queue


* * *

## Set 3: Timed Mock Assessment 3

* **Q1 (Easy): Palindrome Check**
  * *Specification:* Verify if a string is a palindrome, ignoring non-alphanumeric characters.
  * *Sample Test Case:* Input: `"A man, a plan" -> True`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* [PAT-06] Converging Pointers

* **Q2 (Medium): Binary Search Rotated**
  * *Specification:* Find an element in a sorted array that has been rotated.
  * *Sample Test Case:* Input: `[4,5,1,2,3], target=1 -> 2`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* [PAT-10] Partition Search

* **Q3 (Medium-Hard): DFS Component Count**
  * *Specification:* Count the number of connected components (islands) in a 2D grid.
  * *Sample Test Case:* Input: `grid -> 3 islands`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-15] DFS Flood Fill

* **Q4 (Hard): Topological Sort Complex**
  * *Specification:* Find the longest path in a Directed Acyclic Graph representing tasks.
  * *Sample Test Case:* Input: `tasks -> 10 days`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-16] Topo Sort / DP


* * *

## Set 4: Timed Mock Assessment 4

* **Q1 (Easy): In-place Transformation**
  * *Specification:* Move all zeros in an array to the end while maintaining relative order of other elements.
  * *Sample Test Case:* Input: `[0,1,0,3] -> [1,3,0,0]`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* [PAT-02] Write/Read pointers

* **Q2 (Medium): State Machine String**
  * *Specification:* Parse a string to extract a valid integer, handling signs and overflow.
  * *Sample Test Case:* Input: `"-42" -> -42`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* Deterministic finite automaton

* **Q3 (Medium-Hard): Tree Traversal**
  * *Specification:* Serialize and deserialize a binary tree.
  * *Sample Test Case:* Input: `[1,2,3] -> str -> [1,2,3]`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* Preorder traversal

* **Q4 (Hard): Union Find Network**
  * *Specification:* Find the redundant connection in a graph that should be a tree.
  * *Sample Test Case:* Input: `[[1,2],[1,3],[2,3]] -> [2,3]`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-17] Disjoint Set Union


* * *

## Set 5: Timed Mock Assessment 5

* **Q1 (Easy): Simple Math**
  * *Specification:* Return the sum of digits of a given integer until it becomes a single digit.
  * *Sample Test Case:* Input: `38 -> 2`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* Modulo arithmetic

* **Q2 (Medium): Matrix Zeroes**
  * *Specification:* If a cell is 0, set its entire row and column to 0 in-place.
  * *Sample Test Case:* Input: `[[1,0],[1,1]] -> [[0,0],[1,0]]`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* Row/Col marker tracking

* **Q3 (Medium-Hard): Course Schedule II**
  * *Specification:* Return the ordering of courses you should take to finish all courses.
  * *Sample Test Case:* Input: `num=2, req=[[1,0]] -> [0,1]`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-16] Topological Sort

* **Q4 (Hard): Word Ladder**
  * *Specification:* Find the length of the shortest transformation sequence from beginWord to endWord.
  * *Sample Test Case:* Input: `hit -> cog: 5`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-13] BFS Wavefront


* * *

## Set 6: Timed Mock Assessment 6

* **Q1 (Easy): Anagram Validation**
  * *Specification:* Determine if two strings are valid anagrams of one another.
  * *Sample Test Case:* Input: `"listen", "silent" -> True`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* [PAT-01] Frequency buckets

* **Q2 (Medium): Subarray Sum K**
  * *Specification:* Find the total number of continuous subarrays whose sum equals k.
  * *Sample Test Case:* Input: `[1,1,1], k=2 -> 2`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* [PAT-03] Prefix HashMap

* **Q3 (Medium-Hard): Word Search**
  * *Specification:* Check if a word exists in a grid of characters.
  * *Sample Test Case:* Input: `board, "ABCCED" -> True`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-12] DFS Backtracking

* **Q4 (Hard): Longest Valid Parentheses**
  * *Specification:* Find the length of the longest valid (well-formed) parentheses substring.
  * *Sample Test Case:* Input: `")()())" -> 4`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-08] Stack or Two Pointers


* * *

## Set 7: Timed Mock Assessment 7

* **Q1 (Easy): Array Intersection**
  * *Specification:* Find the common elements between two sorted arrays.
  * *Sample Test Case:* Input: `[1,2,3], [2,3,4] -> [2,3]`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* Two pointers matching

* **Q2 (Medium): Sort Colors**
  * *Specification:* Sort an array of 0s, 1s, and 2s in-place (Dutch National Flag).
  * *Sample Test Case:* Input: `[2,0,2,1,1,0] -> [0,0,1,1,2,2]`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* Three pointers

* **Q3 (Medium-Hard): Clone Graph**
  * *Specification:* Return a deep copy (clone) of a graph.
  * *Sample Test Case:* Input: `node 1 -> cloned node 1`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* HashMap + BFS/DFS

* **Q4 (Hard): Monotonic Stack Max Area**
  * *Specification:* Find the largest rectangle in a binary matrix of 0s and 1s.
  * *Sample Test Case:* Input: `matrix -> 6`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-09] Monotonic Stack


* * *

## Set 8: Timed Mock Assessment 8

* **Q1 (Easy): Missing Number**
  * *Specification:* Find the missing number in an array of size N containing numbers from 0 to N.
  * *Sample Test Case:* Input: `[0,1,3] -> 2`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* Sum formula or XOR

* **Q2 (Medium): Peak Element**
  * *Specification:* Find a peak element (strictly greater than neighbors) in O(log N) time.
  * *Sample Test Case:* Input: `[1,2,3,1] -> 2`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* Binary Search on gradient

* **Q3 (Medium-Hard): Evaluate Division**
  * *Specification:* Evaluate queries based on equation relationships a/b = 2.
  * *Sample Test Case:* Input: `a/b=2, b/c=3 -> a/c=6`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* Graph DFS with path weights

* **Q4 (Hard): Minimum Spanning Tree**
  * *Specification:* Given a weighted undirected graph, find the MST weight using Kruskal's algorithm with Union-Find.
  * *Sample Test Case:* Input: `edges -> weight`
  * *Constraints:* V \le 10^4, E \le 5 \times 10^4.
  * *Hint:* [PAT-17] Disjoint Set Union + greedy edge sorting.


* * *

## Set 9: Timed Mock Assessment 9

* **Q1 (Easy): Merge Sorted Arrays**
  * *Specification:* Merge two sorted arrays into a new sorted array.
  * *Sample Test Case:* Input: `[1,3], [2,4] -> [1,2,3,4]`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* Two pointer merge

* **Q2 (Medium): Group Anagrams**
  * *Specification:* Group an array of strings into anagram sets.
  * *Sample Test Case:* Input: `["eat","tea","tan"] -> [["eat","tea"],["tan"]]`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* Frequency string as HashMap key

* **Q3 (Medium-Hard): Time Based Key-Value Store**
  * *Specification:* Create a map that supports setting and getting values by timestamps.
  * *Sample Test Case:* Input: `set(k,v,1), get(k,1) -> v`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* HashMap + Binary Search

* **Q4 (Hard): Trapping Rain Water**
  * *Specification:* Compute how much water it can trap after raining.
  * *Sample Test Case:* Input: `[0,1,0,2,1,0,1,3] -> 6`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* Two Pointers or [PAT-09] Stack


* * *

## Set 10: Timed Mock Assessment 10

* **Q1 (Easy): Longest Prefix**
  * *Specification:* Find the longest common prefix string amongst an array of strings.
  * *Sample Test Case:* Input: `["flower", "flow"] -> "flow"`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* Vertical string scanning

* **Q2 (Medium): Max Area Container**
  * *Specification:* Find two lines that together with the x-axis form a container holding the most water.
  * *Sample Test Case:* Input: `[1,8,6,2,5,4,8,3,7] -> 49`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* [PAT-06] Converging Two-Pointers

* **Q3 (Medium-Hard): LRU Cache**
  * *Specification:* Design a cache with Least Recently Used eviction strategy.
  * *Sample Test Case:* Input: `put(1,1), get(1) -> 1`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* HashMap + Doubly Linked List

* **Q4 (Hard): Burst Balloons**
  * *Specification:* Maximize coins by bursting balloons strategically.
  * *Sample Test Case:* Input: `[3,1,5,8] -> 167`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* Divide & Conquer DP


* * *

## Set 11: Timed Mock Assessment 11

* **Q1 (Easy): Valid Parentheses Basic**
  * *Specification:* Check if a string with just () is balanced.
  * *Sample Test Case:* Input: `"(())" -> True`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* Counter tracking

* **Q2 (Medium): Generate Parentheses**
  * *Specification:* Generate all combinations of n pairs of well-formed parentheses.
  * *Sample Test Case:* Input: `n=2 -> ["(())","()()"]`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* [PAT-12] Backtracking

* **Q3 (Medium-Hard): Merge Intervals**
  * *Specification:* Merge all overlapping intervals.
  * *Sample Test Case:* Input: `[[1,3],[2,6]] -> [[1,6]]`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-23] Sweep-Line Sort

* **Q4 (Hard): Find Median from Data Stream**
  * *Specification:* Design a class to calculate the median of numbers from a data stream.
  * *Sample Test Case:* Input: `add(1), add(2), median -> 1.5`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* Two Heaps (Min/Max)


* * *

## Set 12: Timed Mock Assessment 12

* **Q1 (Easy): Count Elements**
  * *Specification:* Count elements in array that have x+1 present in the array.
  * *Sample Test Case:* Input: `[1,2,3] -> 2`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* HashSet lookup

* **Q2 (Medium): Valid Sudoku**
  * *Specification:* Determine if a 9x9 Sudoku board is valid.
  * *Sample Test Case:* Input: Standard sudoku validation
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* HashMap/Array bitmasking

* **Q3 (Medium-Hard): Construct Binary Tree**
  * *Specification:* Build a tree from preorder and inorder traversal arrays.
  * *Sample Test Case:* Input: `pre=[3,9], in=[9,3] -> Tree`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* Divide and conquer

* **Q4 (Hard): Minimum Window Substring**
  * *Specification:* Find the minimum window in S which will contain all characters in T.
  * *Sample Test Case:* Input: `S="ADOBECODEBANC", T="ABC" -> "BANC"`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-04] Dynamic Sliding Window


* * *

## Set 13: Timed Mock Assessment 13

* **Q1 (Easy): Majority Element**
  * *Specification:* Find the element that appears more than n/2 times.
  * *Sample Test Case:* Input: `[2,2,1,1,1,2,2] -> 2`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* Boyer-Moore Voting

* **Q2 (Medium): Longest Consecutive Sequence**
  * *Specification:* Find the length of the longest consecutive elements sequence in O(N).
  * *Sample Test Case:* Input: `[100,4,200,1,3,2] -> 4`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* HashSet building blocks

* **Q3 (Medium-Hard): Design Add and Search Words**
  * *Specification:* Design a data structure that supports adding words and searching with '.' wildcards.
  * *Sample Test Case:* Input: `add("bad"), search("b.d") -> True`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-24] Trie with DFS

* **Q4 (Hard): 2D DP Pathing**
  * *Specification:* Find minimum path sum in grid moving down/right.
  * *Sample Test Case:* Input: `[[1,3,1],[1,5,1]] -> 7`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-21] 2D DP Grid


* * *

## Set 14: Timed Mock Assessment 14

* **Q1 (Easy): First Unique Character**
  * *Specification:* Find the first non-repeating character in a string.
  * *Sample Test Case:* Input: `"leetcode" -> 0`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* [PAT-01] Frequency counting

* **Q2 (Medium): Top K Frequent Elements**
  * *Specification:* Return the k most frequent elements in an array.
  * *Sample Test Case:* Input: `[1,1,1,2,2,3], k=2 -> [1,2]`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* HashMap and Min-Heap

* **Q3 (Medium-Hard): Permutations**
  * *Specification:* Return all possible permutations of an array of distinct integers.
  * *Sample Test Case:* Input: `[1,2] -> [[1,2],[2,1]]`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-12] Backtracking

* **Q4 (Hard): Course Schedule III**
  * *Specification:* Given N courses with (duration, deadline), maximize courses completed.
  * *Sample Test Case:* Input: `courses -> max`
  * *Constraints:* N \le 10^4.
  * *Hint:* [PAT-25] Priority Queue / Greedy with heap.


* * *

## Set 15: Timed Mock Assessment 15

* **Q1 (Easy): Detect Capital**
  * *Specification:* Verify if the capitalization of a word is correct (all caps, all lower, or title).
  * *Sample Test Case:* Input: `"USA" -> True`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* String traversal

* **Q2 (Medium): Product of Array Except Self**
  * *Specification:* Return array such that answer[i] is product of all elements except nums[i].
  * *Sample Test Case:* Input: `[1,2,3,4] -> [24,12,8,6]`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* Left/Right prefix products

* **Q3 (Medium-Hard): Pacific Atlantic Water Flow**
  * *Specification:* Find grid coordinates where water can flow to both Pacific and Atlantic oceans.
  * *Sample Test Case:* Input: `grid -> [[0,4],[1,3]]`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-15] DFS from borders

* **Q4 (Hard): Word Search II**
  * *Specification:* Given an M×N board of characters and a list of words, find all words that can be formed by sequentially adjacent cells (horizontally or vertically). Each cell may only be used once per word.
  * *Sample Test Case:* Input:
    ```
    board = [
      ["o","a","a","n"],
      ["e","t","a","e"],
      ["i","h","k","r"],
      ["i","f","l","v"]
    ]
    words = ["oath","pea","eat","rain"]
    Output: ["eat","oath"]
    ```
  * *Constraints:* M, N \le 12, words.length \le 3 \times 10^4, words[i].length \le 10.
  * *Hint:* Combine Trie prefix tree with DFS backtracking for efficient multi-word search.


* * *

## Set 16: Timed Mock Assessment 16

* **Q1 (Easy): Reverse Words**
  * *Specification:* Reverse the order of words in a string.
  * *Sample Test Case:* Input: `"the sky is blue" -> "blue is sky the"`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* Split and reverse

* **Q2 (Medium): Search 2D Matrix**
  * *Specification:* Search for a value in a sorted 2D matrix in O(log(MN)).
  * *Sample Test Case:* Input: `matrix, target=3 -> True`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* Virtual 1D Binary Search

* **Q3 (Medium-Hard): Accounts Merge**
  * *Specification:* Merge user accounts that share common email addresses.
  * *Sample Test Case:* Input: `[[John, a@a.com, b@b.com]] -> merged`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-17] Union-Find

* **Q4 (Hard): Longest Increasing Path**
  * *Specification:* Find the longest increasing path in a matrix.
  * *Sample Test Case:* Input: `[[9,9,4],[6,6,8]] -> 4`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* DFS + Memoization


* * *

## Set 17: Timed Mock Assessment 17

* **Q1 (Easy): Contains Duplicate**
  * *Specification:* Return true if any value appears at least twice in the array.
  * *Sample Test Case:* Input: `[1,2,3,1] -> True`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* HashSet

* **Q2 (Medium): Minimum Size Subarray Sum**
  * *Specification:* Find minimal length of subarray with sum >= target.
  * *Sample Test Case:* Input: `target=7, [2,3,1,2,4,3] -> 2`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* [PAT-04] Dynamic Sliding Window

* **Q3 (Medium-Hard): Daily Temperatures**
  * *Specification:* Find how many days to wait for a warmer temperature.
  * *Sample Test Case:* Input: `[73,74,75,71] -> [1,1,0,0]`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-09] Monotonic Stack

* **Q4 (Hard): Sliding Window Maximum**
  * *Specification:* Return the max sliding window of size k.
  * *Sample Test Case:* Input: `[1,3,-1,-3,5,3], k=3 -> [3,3,5,5]`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-05] Monotonic Deque


* * *

## Set 18: Timed Mock Assessment 18

* **Q1 (Easy): Remove Element**
  * *Specification:* Remove all instances of a specific value in-place.
  * *Sample Test Case:* Input: `[3,2,2,3], val=3 -> len=2`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* [PAT-02] Mutation

* **Q2 (Medium): Kth Largest Element**
  * *Specification:* Find the kth largest element in an unsorted array.
  * *Sample Test Case:* Input: `[3,2,1,5,6,4], k=2 -> 5`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* Min-Heap or QuickSelect

* **Q3 (Medium-Hard): Reorder List**
  * *Specification:* Reorder a linked list to L0 -> Ln -> L1 -> Ln-1.
  * *Sample Test Case:* Input: `1->2->3->4 -> 1->4->2->3`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* Middle finding + Reverse + Merge

* **Q4 (Hard): Serialize N-ary Tree**
  * *Specification:* Design an algorithm to serialize and deserialize an N-ary tree.
  * *Sample Test Case:* Input: `tree -> string -> tree`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* DFS Preorder


* * *

## Set 19: Timed Mock Assessment 19

* **Q1 (Easy): String Reversal**
  * *Specification:* Reverse a given string preserving whitespace and capitalization constraints.
  * *Sample Test Case:* Input: `"Hello" -> "olleH"`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* [PAT-02] Two pointers

* **Q2 (Medium): Matrix Spiral**
  * *Specification:* Traverse a 2D matrix in spiral order and return the elements.
  * *Sample Test Case:* Input: `[[1,2],[3,4]] -> [1,2,4,3]`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* Boundary tracking simulation

* **Q3 (Medium-Hard): Sliding Window Max**
  * *Specification:* Find the maximum string length without repeating characters.
  * *Sample Test Case:* Input: `"abcabc" -> 3`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-04] Dynamic Sliding Window

* **Q4 (Hard): 1D DP Robber**
  * *Specification:* Find max value you can rob without triggering adjacent alarms in a circular street.
  * *Sample Test Case:* Input: `[2,3,2] -> 3`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-19] DP State Machine


* * *

## Set 20: Timed Mock Assessment 20

* **Q1 (Easy): Frequency Counting**
  * *Specification:* Find the most frequent character in a given string. Break ties alphabetically.
  * *Sample Test Case:* Input: `"abac" -> 'a'`
  * *Constraints:* Array/String length $1 \le N \le 10^5$.
  * *Hint:* [PAT-01] Frequency Array

* **Q2 (Medium): Two Pointer Target**
  * *Specification:* Find two numbers in a sorted array that add up to target.
  * *Sample Test Case:* Input: `[2,7,11,15], target=9 -> [0,1]`
  * *Constraints:* Appropriate bounds $1 \le N \le 10^4$.
  * *Hint:* [PAT-06] Converging Pointers

* **Q3 (Medium-Hard): HashMap Multi-key**
  * *Specification:* Find the longest subarray with equal numbers of 0s and 1s.
  * *Sample Test Case:* Input: `[0,1,0] -> 2`
  * *Constraints:* Bounds $1 \le N \le 10^5$.
  * *Hint:* [PAT-03] Prefix Sums Hash

* **Q4 (Hard): Alien Dictionary**
  * *Specification:* Given sorted alien words, derive character ordering.
  * *Sample Test Case:* Input: `words -> ordering`
  * *Constraints:* words \le 300, word length \le 100.
  * *Hint:* Topological Sort on character graph.


* * *

## Exam Day 10-Point Speed & Debugging Survival Guide

Before jumping into the 20 Mock Sets, review this executive checklist of top speed traps and invariant bugs to ensure zero lost points on test day:

1. **String Concatenation in Loops ($O(N^2)$ TLE Trap):**  
   Never do `s += c` inside a loop in Java or C#. Creating new String objects on every iteration turns $O(N)$ into $O(N^2)$ time limit exceeded. Always use `StringBuilder` (or `char[]`).
2. **Negative Modulo in Java/C#:**  
   In Java and C#, `-5 % 3` returns `-2` (preserves sign), causing negative array index crashes. Always use the circular safe modulo formula: `(index % N + N) % N`.
3. **Monotonic Stack Width Invariant:**  
   In histogram / largest rectangle problems, after popping height `h = heights[stack.pop()]`, the width is **NOT** `i - poppedIdx + 1`! The true left boundary is `stack.peek()` after popping. Use: `int w = stack.isEmpty() ? i : (i - stack.peek() - 1);`.
4. **Monotonic Stack Sentinel vs. `if (i < n)` Rule:**  
   - *Daily Temperatures / Next Greater:* Pop when `current > top`. Un-popped elements at `i == n` never found a warmer day—leave answer as default 0 using `if (i < n)`.
   - *Histogram Max Area:* Use ghost bar `0` at `i == n`. Do **NOT** skip calculation when `i == n`! The bar extends to the right edge `n - 1`.
5. **Plus One / Add Last Digit Invariant:**  
   Don't write complex `% 10` / `/ 10` / `write--` loops. Walk right-to-left: if `digits[i] < 9`, increment and `return digits;` immediately! If loop finishes, return `new int[N+1]` with `res[0] = 1`.
6. **Character Frequency Indexing (`int[26]` vs `int[10]` vs `int[128]`):**  
   - Lowercase `a-z`: `counts[c - 'a']++` (size 26).
   - Digits `'0'-'9'`: `counts[c - '0']++` (size 10).
   - Mixed ASCII: `counts[c]++;` (size 128 direct ASCII indexing, no HashMap allocation needed).
   - Common Character Count: `common += Math.min(count1[i], count2[i]);` across 0..25.
7. **Matrix Rotation 90° Clockwise Formulas:**  
   - *Rectangular $R \times C \rightarrow C \times R$:* `target[j][R - 1 - i] = matrix[i][j]`
   - *Square $N \times N$ In-Place:* Transpose (`swap(matrix[i][j], matrix[j][i])` for `j > i`), then reverse each row horizontally (`swap(matrix[i][j], matrix[i][N - 1 - j])` for `j < N / 2`).
8. **Binary Search Middle Overflow & Bounds:**  
   Always write `mid = left + (right - left) / 2`. In rotated sorted arrays, check sorted half first: if `nums[left] <= nums[mid]`, left half is monotonically sorted.
9. **Numeric Accumulator Overflow:**  
   When calculating product, array sums, or coordinate products, initialize sum/product accumulators as `long` to prevent 32-bit integer overflow before returning `(int) sum`.
10. **Array Bounds Guarding:**  
    Always check `array != null && array.length > 0` before accessing index `0`, and ensure loops end at `i < array.length` (or `i <= array.length` when using a sentinel).


\part{System Design \& Architecture at Scale}


# System Architecture and Design Fundamentals

> *"A system is not a collection of services, but a web of communication boundaries. If your boundaries are wrong, your microservices are just a distributed monolith."*


## System Design in the Senior Interview

In senior system design interviews, candidates are often asked to design large-scale, low-latency platforms like an ad click aggregator, a video streaming service, or a trading exchange. 

A common pitfall is immediately drawing boxes for databases, load balancers, and caches without grounding the architecture in business specifications. 

To stand out, you must apply **Domain-Driven Design (DDD)**. Define your bounded contexts clearly, design your aggregates to protect business invariants, and construct sequence flows showing exactly how data travels across services while keeping latency low.

In this chapter, we will design the architecture of **ZenithTrade**, a high-frequency order matching exchange, mapping out its service boundaries and order lifecycle.


## Domain-Driven Design (DDD) Boundaries

To design a clean distributed system, you must first establish your domain boundaries using DDD principles.

### Bounded Contexts
A bounded context defines the boundary within which a particular domain model applies. In ZenithTrade, we separate the system into three main bounded contexts:

1.  **Exchange Context (ZenithTrade):** Deals with orders, bid/ask books, matching execution, and price feeds.
2.  **Ledger Context (AuraPay):** Handles balance preservation, double-entry transfers, and deposit/withdrawal checks.
3.  **Identity Context (ChiramTrust):** Manages user credentials, authentication scopes, and KYC compliance.

> **Crucial Mistake:** Do not mix context models. An `Order` inside the Exchange context should not contain details about a user's ledger overdraft limits. Decouple them and bridge them using events or APIs.

### Aggregates, Entities, and Value Objects

-   **Aggregates:** A cluster of associated objects treated as a single unit for data changes (e.g., an `OrderBook`). Changes to orders must go through the `OrderBook` root to protect sorting invariants.
-   **Entities:** Objects with a distinct identity that persists over time (e.g., a `LedgerAccount` with a unique UUID).
-   **Value Objects:** Immutable objects with no identity defined solely by their attributes (e.g., a `Money` value object containing `amount` and `currency`). Value objects have no setters; they are replaced entirely, making them thread-safe.

![DDD Bounded Context Map](editions/python/chapters/16-system-architecture/visuals/ddd_contexts.png){width=85%}


## Monolithic vs. Microservices vs. Event-Driven

Choosing an architectural style is a trade-off between latency, complexity, and operational overhead.

### Monolithic Architecture

-   **Description:** All components (matching, ledger, user management) run inside a single process, sharing memory.
-   **Pros:** Ultra-low latency (nanosecond-range in-memory operations), simple testing, and transactional database integrity.
-   **Cons:** Hard to scale across multiple teams; deployment failures crash the entire system.
-   **Use case:** The core matching engine loop of ZenithTrade must be monolithic and in-memory to meet microsecond latency specs.

### Microservices Architecture

-   **Description:** Services run in independent processes, communicating via synchronous protocols (gRPC, HTTP/REST).
-   **Pros:** Decoupled deployments, scaling independent workloads (e.g., scaling the API Gateway without scaling the matching engine).
-   **Cons:** High network latency (millisecond range), complex distributed transactions, and data consistency challenges.

### Event-Driven Architecture (EDA)

-   **Description:** Services communicate asynchronously by publishing and subscribing to events (Kafka, RabbitMQ).
-   **Pros:** High decoupling, loose runtime dependencies, and high resilience.
-   **Cons:** Eventual consistency. If the matching engine publishes a "TradeExecuted" event, the ledger balances might not update for several milliseconds.

![Monolithic vs Microservices vs Event-Driven Architecture](editions/python/chapters/16-system-architecture/visuals/arch_styles.png){width=80%}

![System Evolution — Scaling from Monolith to Microservices](editions/python/chapters/16-system-architecture/visuals/system_evolution.jpg){width=85%}


## Scaling Out: Partitioning & Consistent Hashing

A single matching engine instance cannot handle all trading instruments globally. To scale ZenithTrade horizontally, we must partition (shard) the matching workload.

### Consistent Hashing for Instrument Sharding

![Consistent Hashing Ring — Distributed Key Routing](editions/python/chapters/16-system-architecture/visuals/consistent_hashing.jpg){width=85%}

Instead of traditional modulo sharding (`hash(instrumentId) % nodeCount`), which causes massive data reshuffling when nodes are added or removed, ZenithTrade utilizes a **Consistent Hash Ring**:

1.  **The Ring:** The hash space is mapped onto a circular ring (e.g., 0 to $2^{32} - 1$).
2.  **Node Mapping:** Matching Engine instances (nodes) are hashed and placed at specific coordinates on the ring. We map multiple "virtual nodes" per physical machine to ensure uniform distribution of load.
3.  **Key Mapping:** Incoming orders are routed based on `hash(instrumentId)` (e.g., `BTC-USD`, `ETH-EUR`). The order is handled by the first matching engine node encountered walking clockwise from the key's hash coordinate.
4.  **Rebalancing:** When a new matching engine node is added to the cluster, it only takes a portion of keys from its immediate clockwise neighbor, keeping rebalancing traffic to a minimum.


## Command Query Responsibility Segregation (CQRS)

In financial systems, read traffic (users querying active order books, historical trades, and account balances) is several orders of magnitude higher than write traffic (executing transactions or submitting orders). Applying **CQRS** prevents read queries from degrading write performance:

-   **Command Path (Write):** Optimized for low latency and consistency. Incoming orders are processed by the in-memory matching engine, writing state changes sequentially to a Write-Ahead Log (WAL) or transactional ledger database.
-   **Query Path (Read):** Optimized for high-throughput queries. State change events (e.g., `OrderPlaced`, `TradeExecuted`) are published to Kafka and consumed by read-projection workers. These workers update read-optimized views in Elasticsearch (for historical search) or Redis (for fast order book rendering).
-   **Consistency Trade-off:** The read model is **eventually consistent** (typically lagging the command path by a few milliseconds), which is acceptable for user displays as long as the write path remains strictly consistent.


## CAP Theorem & Distributed Trade-offs

The CAP Theorem states that in a distributed system, you can only guarantee two out of three properties during a network partition: **Consistency (C)**, **Availability (A)**, or **Partition Tolerance (P)**. Because network partitions are inevitable in real-world infrastructure, system design is a choice between **CP** and **AP**:

![CAP Theorem — Consistency, Availability, and Partition Tolerance Trade-offs](editions/python/chapters/16-system-architecture/visuals/cap_theorem.jpg){width=85%}

-   **The Ledger Context (CP Choice):** AuraPay is designed as a **CP** system. In financial bookkeeping, correctness is non-negotiable. If a network partition occurs between ledger replicas, we must reject transaction requests (sacrificing availability) rather than risk allowing double-spending or balance mismatch (sacrificing consistency). Consensus protocols like Raft or Paxos are used to coordinate commits across healthy replicas.
-   **The Market Feed Context (AP Choice):** The ZenithTrade public price feed (ticker data) is designed as an **AP** system. If a partition occurs, it is better to continue broadcasting the latest available price data (even if slightly stale) to users than to shut down the feed entirely.


## API Design & Idempotency

When designing APIs for microservices, you must handle network failures gracefully. If a client submits a payment request but the connection drops before receiving a response, the client will retry the request. Without **Idempotency**, this leads to double-billing.

### Idempotency Keys in REST & gRPC

1.  **Client-Generated Key:** The client generates a unique UUID (e.g., `Idempotency-Key: f81d4fae-7dec-11d0-a765-00a0c91e6bf6`) and sends it in the request header.
2.  **API Gateway Check:** The API Gateway intercepts the request and queries Redis to see if the key exists:

    -   **Case 1 (New Request):** The gateway stores the key in Redis with a status of `PENDING` and routes the request.
    -   **Case 2 (Duplicate Pending):** If the key is found and its status is `PENDING`, the gateway returns a `409 Conflict` (request is currently processing).
    -   **Case 3 (Duplicate Completed):** If the key is found and its status is `COMPLETED`, the gateway returns the cached response payload directly, bypassing the backend services entirely.
3.  **Backend Commit:** Once the transaction settles, the service updates the status in Redis to `COMPLETED` and writes the response payload, ensuring the cache has a defined TTL (Time To Live, e.g., 24 hours).


## ZenithTrade Order Lifecycle Sequence

The following sequence diagram maps out how an order is submitted, validated, matched inside the memory buffer, and settled inside the ledger:

![ZenithTrade Order Lifecycle Sequence](editions/python/chapters/16-system-architecture/visuals/order_lifecycle.png){width=95%}

### Explaining the Sequence:

1.  **Gateway Ingest:** The API Gateway validates rate limits, checks for duplicate requests using the `Idempotency-Key`, and passes the request to the Exchange Context.
2.  **Order Validator:** Before an order enters the book, the validator calls the AuraPay ledger to verify that the client has sufficient funds (Pre-condition check).
3.  **In-Memory Matching:** The OrderBook matches buy and sell orders. Since this is CPU-intensive, it runs in memory.
4.  **Ledger Settlement:** Once matched, a double-entry transaction settles the trade inside the AuraPay database.
5.  **Asynchronous Notification:** The client is notified via WebSockets, completely out of the blocking execution thread path.


## API Rate Limiting Strategies

Rate limiting is essential for protecting APIs from abuse and cascading failures. The following four algorithms are foundational in system design interviews.

### Token Bucket Algorithm
The token bucket algorithm maintains a bucket that holds a maximum number of tokens (capacity). Tokens are added to the bucket at a fixed rate. Each incoming request consumes one token; if the bucket is empty, the request is dropped. It is widely used because it allows controlled bursts of traffic while enforcing a sustained long-term rate.

**Parameters:**

- **Capacity (Burst Size):** Maximum number of tokens the bucket can hold.
- **Refill Rate:** Rate at which new tokens are generated.

**When to use:** API gateways and per-user throttling (e.g., Stripe, Amazon API Gateway).

```python
import time
import threading

class TokenBucket:
    def __init__(self, max_tokens: int, refill_rate_per_second: int):
        self.max_tokens = max_tokens
        self.refill_rate_per_second = refill_rate_per_second
        self.tokens = max_tokens
        self.timestamp_nanos = time.monotonic_ns()
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        with self.lock:
            now = time.monotonic_ns()
            elapsed = now - self.timestamp_nanos
            refilled = min(self.max_tokens, 
                self.tokens + elapsed * self.refill_rate_per_second // 1_000_000_000)
            
            if refilled <= 0:
                return False
                
            self.tokens = refilled - 1
            self.timestamp_nanos = now
            return True
```


### Leaky Bucket Algorithm
In the leaky bucket algorithm, incoming requests enter a FIFO queue (the bucket). The system processes requests from the queue at a strictly constant rate. If the queue is full, new requests are discarded. Unlike the token bucket, it entirely smooths out bursts, ensuring a perfectly constant output rate.

**When to use:** Network traffic shaping and scenarios requiring steady-throughput processing.

### Fixed Window Counter
The fixed window counter algorithm counts incoming requests per discrete time window (e.g., 00:00 to 00:01). If the counter exceeds the threshold, requests are dropped. It is simple but suffers from the **boundary burst** problem: a user can send 100 requests at 00:00:59 and another 100 requests at 00:01:01, effectively pushing 200 requests in a two-second span across the boundary.

**When to use:** Simple scenarios where edge-case bursts are acceptable.

### Sliding Window Log / Counter
This approach addresses the boundary burst issue. A Sliding Window Log tracks individual request timestamps, discarding older ones to precisely enforce the rate over a rolling window. A Sliding Window Counter optimizes memory by keeping weighted counters of the previous and current overlapping windows.

**Trade-off:** Higher memory usage (for logs) or slight approximations (for counters).
**When to use:** Strict rate limiting scenarios where boundary bursts are unacceptable.

| Algorithm | Burst Handling | Memory | Accuracy | Complexity |
|---|---|---|---|---|
| Token Bucket | Allows controlled bursts | O(1) | Good | Low |
| Leaky Bucket | Smooths all bursts | O(N) queue | Good | Medium |
| Fixed Window | Boundary bursts possible | O(1) | Approximate | Low |
| Sliding Window | No boundary bursts | O(N) timestamps | Exact | High |


## Caching Architectures

Caching is a critical component for reducing database load and improving read latencies. The following patterns are essential for system design interviews.

### Cache-Aside (Lazy Loading)
In a Cache-Aside pattern, the application is fully responsible for managing the cache. For every read, the application first checks the cache. On a cache miss, it reads from the database, writes the result to the cache, and then returns the data.

**Pros:** Only requested data is cached, avoiding unnecessary memory usage. The system remains available (reading directly from the DB) even if the cache fails.
**Cons:** Introduces a cache miss penalty (latency spike) and risks serving stale data if not carefully invalidated.

```python
class UserService:
    def __init__(self, db_repository, redis_client):
        self.db_repository = db_repository
        self.redis_client = redis_client

    def get_user(self, user_id: str):
        cache_key = f"user:{user_id}"
        user = self.redis_client.get(cache_key)
        
        if user is None:
            # Cache miss: read from DB
            user = self.db_repository.find_by_id(user_id)
            if user is None:
                raise ValueError("User not found")
            # Populate cache
            self.redis_client.set(cache_key, user)
            
        return user
```


### Write-Through Cache
Under Write-Through caching, the application writes data to the cache and the database simultaneously (often abstracted so the application only writes to the cache, which synchronously updates the DB).

**Pros:** The cache is always strongly consistent with the database.
**Cons:** Higher write latency due to the dual synchronous writes. Also caches data that might never be read again.

### Write-Behind (Write-Back) Cache
With Write-Behind caching, the application writes exclusively to the cache, which acknowledges the write immediately. The cache then asynchronously flushes the data to the persistent database in the background.

**Pros:** Ultra-low write latency and reduced database load via batching.
**Cons:** High risk of data loss if the cache node crashes before flushing to the database.
**When to use:** High-write-throughput systems where occasional data loss is an acceptable trade-off.

### Cache Eviction Policies
When the cache reaches its memory limit, older data must be removed:

- **LRU (Least Recently Used):** Evicts the item that hasn't been accessed for the longest time. The most common and generally applicable policy.
- **LFU (Least Frequently Used):** Evicts the item with the lowest access frequency. Better for highly skewed, long-term access patterns.
- **TTL (Time To Live):** Automatically expires keys after a set duration, acting as a natural safeguard against stale data.

| Pattern | Consistency | Write Latency | Read Latency | Complexity |
|---|---|---|---|---|
| Cache-Aside | Eventual | Normal | Fast (on hit) | Low |
| Write-Through | Strong | Higher | Fast | Medium |
| Write-Behind | Eventual | Ultra-low | Fast | High |

### Cache Stampede Prevention
A cache stampede occurs when a highly requested cache entry expires (TTL elapses). Suddenly, hundreds of concurrent requests experience a cache miss and hit the database simultaneously, potentially bringing it down.

- **Solution 1: Mutex/Lock:** Implement a distributed lock so that only one thread experiencing the miss queries the database and refreshes the cache; other threads wait for the cache to be populated.
- **Solution 2: Early Expiry with Jitter:** Refresh the cache slightly before the actual TTL expires, utilizing a background worker.
- **Solution 3: Probabilistic Early Expiry:** Each incoming request has a small, random probability of refreshing the cache just before it naturally expires, spreading the DB load gracefully.


## Consumer-Scale System Design Archetypes

While this book's case studies emphasize financial systems with strict consistency requirements, many interviews target consumer-scale platforms. Here are the key architectural patterns for the most common system design questions:

**Design a Social Media Feed (Twitter/X Timeline)**
- Fan-out-on-write vs fan-out-on-read trade-off
- Celebrity problem: hybrid approach for users with >10K followers
- Timeline cache per user (Redis sorted sets by timestamp)
- Media storage: object store (S3) with CDN distribution
- Key metric: Feed generation < 200ms for 99th percentile

**Design a Ride-Sharing Service (Uber/Lyft)**
- Geospatial indexing: QuadTree or Geohash for driver location
- Driver-rider matching: nearest-neighbor search with ETA ranking
- Real-time location updates: WebSocket with 3-second heartbeats
- Surge pricing: demand/supply ratio per geohash cell
- Key metric: Match latency < 5 seconds in urban areas

**Design a Video Streaming Platform (Netflix/YouTube)**
- Adaptive bitrate streaming (HLS/DASH) with multiple encodings
- CDN edge caching: hot content pushed to 200+ PoPs globally
- Recommendation engine: collaborative filtering + content-based hybrid
- Upload pipeline: async transcoding queue (multiple resolutions)
- Key metric: Start-to-play < 2 seconds, rebuffer ratio < 0.5%

**Design a URL Shortener (bit.ly)**
- Base62 encoding of auto-increment ID (or MD5 hash truncation)
- Read-heavy (100:1 read/write ratio) → heavy caching layer
- 301 (permanent) vs 302 (temporary) redirect trade-offs for analytics
- Key metric: Redirect latency < 10ms at 100K QPS

For each archetype, the candidate should follow the same spec-driven approach used throughout this book: define the invariants (what must ALWAYS be true), identify the data flow, and select patterns from the canonical set.


## System Design Mock Interview: Sharded Order Matching Engine

To demonstrate how a senior candidate should navigate a system design round, here is a transcript-style mock interview.

### Requirements Gathering (The First 5 Minutes)
**Interviewer:** *"I want you to design a high-frequency order matching engine for a cryptocurrency exchange. How would you approach this?"*

**Candidate:** *"Before drawing components, I want to establish our core functional and non-functional requirements to set our design boundaries."*

*Functional Requirements:*

- Users can place Limit Orders (buy/sell a specific quantity at a specific price) and Market Orders.
- The matching engine must match buy and sell orders based on Price-Time priority.
- Trade execution must trigger account balance updates in a ledger.

*Non-Functional Requirements:*

- **Ultra-Low Latency:** Order matching must execute with sub-millisecond latency (p99 < 1ms).
- **High Throughput:** The system must handle 100,000 requests per second (RPS) peak load.
- **Strict Consistency:** The matching engine and ledger must prevent double-spending and guarantee double-entry correctness. We choose a **CP** model for the ledger.
- **High Availability:** The system must remain available even if a node crashes.

### High-Level Estimations (Scale & Math)
**Candidate:** *"Let's calculate our network and storage needs. At 100,000 RPS, if an average order payload is 200 bytes, our network ingest rate at the gateway is:"*

```
Ingest Bandwidth = 100,000 * 200 bytes = 20 MB/s = 160 Mbps
```

*"This is easily handled by standard network infrastructure. However, processing 100,000 matches per second in a single SQL database is impossible due to disk I/O bottlenecks. Therefore, our primary design boundary is that **the active matching engine must run entirely in-memory**, keeping reads and writes decoupled from disk operations during the matching loop."*

### API & Schema Design
**Candidate:** *"Let's define the API payload for placing a limit order. We'll use gRPC over HTTP/2 for low latency:"*

```protobuf
message PlaceOrderRequest {
    string idempotency_key = 1;
    string account_id = 2;
    string instrument_id = 3; // e.g., "BTC-USD"
    enum Side { BUY = 0; SELL = 1; }
    Side side = 4;
    double price = 5;
    double quantity = 6;
}
```

**Interviewer:** *"How do you handle the precision of prices and quantities? Double float values are prone to rounding errors."*

**Candidate:** *"Excellent point. In banking and exchange systems, floating-point arithmetic is a major risk because operations like `0.1 + 0.2` can result in precision loss. To enforce our correctness invariants, we represent prices and quantities as integers representing the smallest atomic units (e.g., satoshis for BTC, or multiplying USD by $10^8$ to store as integers), or utilize the `BigDecimal` type at our database and application borders."*

### Deep Dive: In-Memory Data Structures
**Interviewer:** *"How would you design the Order Book in memory to achieve sub-millisecond matching latency?"*

**Candidate:** *"To match orders quickly based on Price-Time priority, we need fast insertion, fast deletion (for cancellations), and fast retrieval of the highest bid and lowest ask. We will design the `OrderBook` using two collections: `bids` and `asks`."*

- *Bids Book:* Sorted descending by price.
- *Asks Book:* Sorted ascending by price.

*"For each book, we use a **TreeMap** (or Red-Black Tree) where the key is the price level, and the value is a **doubly-linked list** of orders at that price level (FIFO queue). This gives us:"*

- *Lookup/Match peak:* $O(1)$ to access the head of the tree.
- *Insert/Cancel:* $O(\log P)$ where $P$ is the number of distinct price levels, which is highly optimized.

### Horizontal Scaling: Partitioning the Exchange
**Interviewer:** *"How do you scale this matching engine when the number of instruments and users grows beyond a single machine's capacity?"*

**Candidate:** *"We partition our matching engine horizontally by **Instrument ID** (e.g., `BTC-USD`, `ETH-USD`, `SOL-USDT`). Because orders for different instruments do not interact, we can run completely isolated Matching Engine instances on different machines."*

*"We will use a **Consistent Hash Ring** at the API Gateway layer to route incoming orders. The gateway hashes the `instrument_id` and forwards the request to the designated matching node. This prevents hotspots and ensures that adding a new matching node only impacts a fraction of the ring."*

### Reliability and Failover
**Interviewer:** *"If an in-memory matching engine node crashes, how do you recover the state without losing orders?"*

**Candidate:** *"We use a **Write-Ahead Log (WAL)** pattern with active-passive replication. Every incoming order is written to an append-only log on disk (SSD) sequentially before it is processed by the matching engine. Since sequential writes are extremely fast (disk I/O is minimized), this preserves low latency."*

*"Additionally, each matching partition runs as a Raft consensus group containing one Leader and two Followers. The Leader streams the WAL to the Followers. If the Leader crashes, the Followers elect a new Leader, which replays the log from its last committed index to rebuild the in-memory state. This guarantees no order loss and sub-second failover recovery."*



## Modern Infrastructure Patterns (2024+)

Modern system design interviews increasingly expect familiarity with container orchestration and cloud-native patterns:

**Kubernetes Pod Autoscaling:** Horizontal Pod Autoscaler (HPA) scales replicas based on CPU/memory or custom metrics. For AuraPay's payment gateway, HPA with target CPU utilization of 70% ensures elastic scaling during Black Friday traffic spikes.

**Sidecar Proxy Pattern (Envoy/Istio):** Instead of application-level circuit breakers (like Resilience4j), modern architectures delegate traffic management to sidecar proxies. Each microservice pod gets an Envoy sidecar that handles circuit breaking, retry budgets, and mutual TLS — without any application code changes.

**Observability with eBPF:** Extended Berkeley Packet Filter enables kernel-level observability without code instrumentation. Tools like Cilium and Pixie capture request latencies, error rates, and network flows at the kernel level, providing distributed tracing with zero application overhead.

**Serverless Trade-offs:** Lambda/Cloud Functions eliminate infrastructure management but introduce cold start latency (100ms-2s), vendor lock-in, and debugging complexity. Use for event-driven workloads (image processing, webhook handling), not for latency-critical paths.

> ⭐ **STAR Moment: Bounded Context Isolation**
> 
> During system design interviews, explain that microservice division should mirror DDD Bounded Contexts. Say: *"We will isolate the ZenithTrade Matching Engine from the AuraPay Ledger. If the ledger experiences a database write lag, our matching engine can continue to accept and queue orders in memory, preventing system-wide downtime."* This shows you design for fault isolation.


# Enterprise Integration and Resiliency

> *"In a distributed system, failure is not an anomaly; it is a normal state of operation. Designing for reliability is the science of preventing local failures from becoming global disasters."*


## The Distributed Transaction Dilemma

In monolithic architectures, maintaining data consistency is straightforward: you open a database transaction, perform updates, and commit. If any step fails, the database rolls back all changes.

In a microservices architecture, however, a single business action can span multiple service boundaries. For instance, when a user purchases stock on ZenithTrade:

1.  The Exchange service matches the order.
2.  The AuraPay ledger service updates the account balance.
3.  The Custody service updates securities ownership.

Since these services use independent databases, you cannot use a database transaction (2PC - Two-Phase Commit is generally avoided in high-performance cloud environments due to lock overhead and latency). If the ledger debit succeeds but the custody credit fails, the system enters an inconsistent state.

In a senior architecture interview, you must explain how to resolve this. You will be evaluated on your understanding of the **Saga Pattern** and the **Transactional Outbox Pattern**.


## The Dual-Write Anti-Pattern

A common architectural flaw is the **Dual-Write**. This occurs when a service attempts to modify a database and send a message to a message broker (like Kafka or RabbitMQ) within the same API request:

```python
# Anti-pattern: Dual-Write
def complete_transaction(tx: TransactionRecord) -> None:
    database.save(tx) # Database Write
    kafka_producer.send("transaction-topic", tx) # Network Call
```


This is highly unreliable:

- If the database write succeeds but the message broker is temporarily down or network packet loss occurs, the message is lost, and downstream services (like Auditing or Risk Engine) are never notified.
- If you reverse the order and send the message first, the database write might fail (e.g., due to a constraint violation), but the rest of the system will process the event, leading to phantom actions.

### The Transactional Outbox Pattern
To guarantee **At-Least-Once Delivery**, you must write the business data and an event record to an "outbox" table *in the same local database transaction*. Because they use the same database, either both writes succeed, or both fail.

A background process (or CDC log tailer like Debezium) then polls the outbox table, publishes the events to the message broker, and marks them as processed.

The following code illustrates this Outbox Publisher worker:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

@dataclass
class OutboxEvent:
    id: UUID
    aggregate_type: str
    aggregate_id: UUID
    event_type: str
    payload: str
    created_at: datetime
    processed: bool

class MessageBrokerClient(ABC):
    @abstractmethod
    def publish(self, topic: str, payload: str):
        pass

class OutboxRepository(ABC):
    @abstractmethod
    def find_unprocessed_and_lock(self, limit: int) -> List[OutboxEvent]:
        pass

    @abstractmethod
    def mark_as_processed(self, event_id: UUID):
        pass

class TransactionalOutboxPublisher:
    """
    Service that polls the database Outbox table and publishes events to the broker.
    Guarantees At-Least-Once delivery of domain events.
    """
    def __init__(self, outbox_repository: OutboxRepository, broker_client: MessageBrokerClient):
        self.outbox_repository = outbox_repository
        self.broker_client = broker_client

    def publish_pending_events(self):
        # Retrieve unprocessed events under lock
        pending_events = self.outbox_repository.find_unprocessed_and_lock(100)

        for event in pending_events:
            try:
                # Publish to broker (external network call)
                topic = f"events.{event.aggregate_type.lower()}"
                self.broker_client.publish(topic, event.payload)

                # Mark as processed in the database
                self.outbox_repository.mark_as_processed(event.id)
            except Exception as e:
                # If publishing fails, we log and skip.
                # It will be retried on the next poll cycle (At-Least-Once).
                print(f"Failed to publish outbox event {event.id}: {str(e)}. Will retry.")
```


![Transactional Outbox Pattern](editions/python/chapters/17-resiliency/visuals/outbox_pattern.png){width=85%}

If the message broker fails during publication, the event remains unmarked in the database and will be retried in the next execution cycle. This ensures that the message is eventually delivered at least once.


## Event Sourcing

For financial ledgers (like AuraPay) where correctness and auditability are paramount, storing only the "current state" of an account is insufficient. A senior candidate should discuss **Event Sourcing**:

### State vs. Stream

- **State-Based Storage:** Storing a row `Account(id=101, balance=500.00)`. If a balance mismatch occurs, it is impossible to trace *why* the balance is incorrect without parsing external database logs.
- **Event-Sourced Storage:** Storing a stream of immutable events: `[Deposited(50.00), Deposited(70.00), Debited(20.00)]`. The current balance is a derived projection computed by folding/aggregating these events over time:

```
Current Balance = Sum(Credit Events) - Sum(Debit Events)
```

### Key Invariants & Advantages

1. **Mathematical Auditability:** Every balance change is linked to an immutable event. Historians can reconstruct the ledger state at any specific millisecond.
2. **Side-Effect Isolation:** Commands generate events. Events are appended to the event store (a sequential, write-only database) and then published to message brokers to trigger downstream read-projections, fully separating writes from read overhead.


## Distributed Sagas

A **Saga** is a sequence of local transactions. Each local transaction updates the database within a single service. If a step fails, the Saga orchestrator or participants execute a series of **compensating transactions** that undo the changes made by the preceding steps.

> **Why is it called a "Saga"?** The term comes from a **1987 research paper** by Hector Garcia-Molina and Kenneth Salem at Princeton University. They chose "Saga" because, like an epic literary saga with many chapters, a distributed transaction is a long-running story told through a sequence of smaller, self-contained episodes. If the story goes wrong at any chapter, you cannot un-tell the earlier chapters — you must write new compensating chapters to undo their effects. The metaphor is surprisingly precise.

There are two primary ways to design a Saga:

### Choreography-Based Saga
In a choreography-based saga, there is no central coordinator. Each service performs its transaction and emits an event. Other services listen to these events and perform their tasks.

-   **Pros:** Decoupled, no single point of failure, simple to implement for small workflows.
-   **Cons:** Hard to understand as the number of services grows; risks of cyclic dependencies.

### Orchestration-Based Saga
In an orchestration-based saga, a central service (the orchestrator) coordinates the workflow. It tells the participants what local transactions to execute and in what order. If a failure occurs, the orchestrator issues the rollbacks.

-   **Pros:** Clear visibility into the state of the transaction; easier to debug and manage complex flows.
-   **Cons:** Introduces a central point of failure; requires a state-machine engine.

![Saga Orchestration vs Choreography](editions/python/chapters/17-resiliency/visuals/saga_comparison.png){width=90%}


## Distributed Rate Limiting

To protect microservices from cascading failures or brute-force spikes, you must implement rate limiting. In a distributed environment, rate limits cannot be stored in-memory on a single application node.

### Redis Sliding Window Rate Limiter
We use Redis to store request timestamps. A sliding window rate limiter maintains a sorted set for each user:

1.  **Add Request:** Add current timestamp to sorted set using `ZADD`.
2.  **Prune Old Requests:** Remove timestamps older than the sliding window (e.g., current time minus 1 minute) using `ZREMRANGEBYSCORE`.
3.  **Count Volume:** Count active timestamps using `ZCARD`.
4.  **Enforce Limit:** If the count exceeds the threshold, reject the request. Otherwise, allow it and set a key TTL (`EXPIRE`) to reclaim memory when the client goes inactive.

![Redis Sliding Window Rate Limiting](editions/python/chapters/17-resiliency/visuals/rate_limiter.png){width=70%}


## Microservice Resiliency Patterns

When designing distributed systems, you must prevent cascading failures where one slow service consumes all resources on upstream callers.

```text
[Client] ---> [API Gateway] ---> [Exchange Service] ---> [Slow Ledger Service]
                                 (Threads Exhausted)
```

### Circuit Breakers
A **Circuit Breaker** wraps remote calls. It monitors failure rates.

-   **Closed State:** Requests pass through.
-   **Open State:** When the failure rate crosses a threshold (e.g., 50% failures over 10 seconds), the circuit trips (opens). Subsequent requests fail fast immediately, preventing resource exhaustion on the caller.
-   **Half-Open State:** After a timeout, the breaker allows a few probe requests to pass. If they succeed, it closes; if they fail, it opens again.

![Circuit Breaker State Machine](editions/python/chapters/17-resiliency/visuals/circuit_breaker.png){width=85%}

> **Why is it called a "Circuit Breaker"?** The pattern is borrowed directly from **electrical engineering**. In your home's breaker panel, a circuit breaker trips (opens) when it detects excessive current, preventing an electrical fire. Michael Nygard popularized the software version in his 2007 book *Release It!*, mapping the electrical metaphor to distributed systems: when a downstream service is failing, "trip the breaker" to fail fast and protect the calling system from cascading overload. The three states (Closed, Open, Half-Open) mirror how a physical breaker resets after the fault clears.

### Bulkheads
Named after the watertight compartments of a ship's hull. The **Bulkhead Pattern** isolates resources (like thread pools or memory) allocated to specific services. If the Ledger Service slows down, only the thread pool dedicated to the Ledger will exhaust its threads. The rest of the Exchange Service (such as market data streaming) remains completely unaffected.

> **Why "Bulkhead"?** On a cargo ship, bulkheads are vertical walls that divide the hull into sealed compartments. If one compartment floods, the bulkheads prevent water from spreading to adjacent compartments — the ship stays afloat. In software, we partition thread pools and connection pools the same way: one failing dependency can drain its own pool without sinking the entire application.

### Mock Interview Transcript: Cascading Failures

> **Interviewer:** Your payment service is experiencing cascading failures. Walk me through your approach to stop the bleeding and restore stability.
> **Candidate:** First, we need to halt the cascade. I would ensure we have circuit breakers wrapping our downstream calls to the payment gateway. If the failure rate spikes, the breaker trips to the open state, immediately returning an error instead of blocking threads. 
> **Interviewer:** Good. But if the breaker is open, all payments fail. Do you have a fallback?
> **Candidate:** We can implement a fallback strategy, like queuing the payment request in an outbox or Kafka topic for deferred processing, or serving a cached "payment pending" response to the user.
> **Interviewer:** What happens if your fallback also fails, say the queue broker is unreachable?
> **Candidate:** Actually, let me reconsider... If the fallback infrastructure is also down, we must fail gracefully. We return a clear 503 Service Unavailable to the client. We shouldn't try complex secondary fallbacks because that introduces more points of failure during an incident. We'd rely on bulkhead isolation to ensure this doesn't bring down unrelated services, like the user profile service.
> **Interviewer:** Makes sense. How do you decide the timeout thresholds before tripping the circuit breaker?
> **Candidate:** We shouldn't guess. We derive them from our SLAs and historical p99 latencies. If p99 is normally 200ms, a timeout of 500ms might be appropriate. For retries, we'd use exponential backoff with jitter to avoid overwhelming the recovering service.
> **Interviewer:** And how do you test this?
> **Candidate:** We'd use chaos engineering, deliberately injecting latency into the payment gateway in a staging environment to observe the breaker state transitions and bulkhead thread pools.

**Technical Summary:** The candidate effectively utilized circuit breakers to fail fast, bulkhead isolation to protect the broader system, and exponential backoff for retries. They correctly identified that complex fallbacks can exacerbate outages and demonstrated a data-driven approach to setting timeout thresholds using p99 metrics.


## Microservices Observability

A resilient architecture is impossible to manage without deep visibility into execution paths. In technical interviews, discuss the **Three Pillars of Observability**:

### Structured Logging & Trace Propagation
Never write plain text logs. All logs must be output as structured JSON. To trace a single request as it hops across multiple microservices (API Gateway $\to$ Exchange $\to$ Ledger), utilize **Trace Context Propagation**:

- When a request enters the API Gateway, the gateway checks for a `traceparent` HTTP header (W3C standard). If missing, it generates a unique `trace_id` (128-bit).
- The gateway includes this `trace_id` in all outgoing HTTP requests, gRPC metadata, or Kafka message headers.
- Every service logs the current `trace_id` along with its log statements. In centralized log management systems (like ELK Stack or Datadog), searching for a single `trace_id` brings up the exact execution timeline across all services.

### Distributed Tracing
Utilize OpenTelemetry to capture spans (timed execution blocks). Spans record database queries, network latencies, and function call execution times, creating visualization traces to pinpoint latency hotspots.

### Metrics Collection
Expose endpoints (e.g., Prometheus Prometheus JMX/Micrometer) to collect performance metrics:

- **System Metrics:** CPU usage, memory utilization, JVM garbage collection frequency, thread counts.
- **Application Metrics:** API request rates, HTTP 5xx error counts, database connection pool saturation, and circuit breaker states.


> ⭐ **STAR Moment: Compensating Transactions vs Rollback**
> 
> In a system design interview, make sure to emphasize that a Saga cannot "rollback" in the traditional database sense, because the initial transactions have already been committed. Instead, we must write explicit **compensating transactions** (e.g., if a debit was committed, the compensation is a credit). You must design these compensating operations to be **idempotent**, as they may be retried multiple times during a network partition.


# Database Design, Compliance, and Security

> *"In financial systems, a database is not just a storage system; it is the ultimate source of truth, legal compliance, and operational trust."*


## Database Architecture in Interviews

When designing systems in interviews, candidates frequently treat databases as simple black boxes, choosing "MySQL" or "MongoDB" arbitrarily. 

At a senior level, you must justify your database selection based on transactional guarantees (ACID), storage engines (B-Tree vs. LSM-Tree), and compliance boundaries (PCI-DSS, SOC2). If your system processes card transactions or sensitive personal identifiable information (PII), you must articulate how to encrypt and tokenize this data to prevent costly leaks.

In this chapter, we explore how to design a secure, compliant storage architecture for AuraPay, focusing on PCI-DSS card tokenization and database index tuning.


## ACID vs. NoSQL: Choosing the Right Engine

For financial transaction ledgers, the choice of database is crucial. 

### Relational Databases (RDBMS)
RDBMS engines (PostgreSQL, MySQL, Oracle) utilize **ACID** transactions (Atomicity, Consistency, Isolation, Durability).

-   **Why it's essential:** In a double-entry book-keeping system, a debit and credit must succeed or fail together. An RDBMS ensures that a database failure halfway through a transaction rolls back both sides of the ledger.
-   **Storage Engine (B-Tree):** RDBMS platforms typically use B-Tree indexes. B-Trees are optimized for read-heavy workloads with rapid random access but can suffer from write amplification during high-velocity insert/update operations.

### NoSQL & NewSQL Databases

-   **NoSQL (Cassandra, DynamoDB):** Trade consistency for scalability (BASE model - Basically Available, Soft state, Eventual consistency). They use LSM-Tree (Log-Structured Merge-tree) storage engines, which write sequentially to memory buffers (MemTable) before flushing to disk (SSTable), providing very high write speeds but slow random reads.
-   **NewSQL (Spanner, CockroachDB):** Provide the scale of NoSQL with the ACID guarantees of an RDBMS using distributed consensus protocols (Raft/Paxos) and atomic clocks.

![B-Tree vs LSM-Tree Storage Engines](editions/python/chapters/18-database-compliance/visuals/btree_vs_lsm.png){width=85%}

> **Why is it called \"PostgreSQL\"?** The name traces back to the 1970s. UC Berkeley professor Michael Stonebraker created a relational database called **Ingres**. In 1986, he started a successor project called **Post-Ingres** (i.e., \"after Ingres\"), later shortened to **Postgres**. When SQL support was added in 1996, the name became **PostgreSQL** \u2014 literally \"Post-Ingres with SQL.\" The elephant logo? Chosen simply because elephants *never forget* \u2014 a fitting mascot for a database.

> **Why is it called \"Redis\"?** The name is an acronym: **RE**mote **DI**ctionary **S**erver. Italian developer Salvatore Sanfilippo (known online as *antirez*) created it in 2009 because he needed a fast in-memory key-value store for his real-time web analytics startup. He designed it as a networked dictionary \u2014 a remote hash map you can query over TCP. The name captures exactly what it is: a dictionary server that lives on a remote machine.

> **Interview Rule:** Always use an ACID-compliant engine (RDBMS or NewSQL) for core ledgers. Use NoSQL only for write-heavy, eventually-consistent workloads like clickstreams, activity logs, or audit trail event streams.


## Database Sharding Strategies

When database size or write throughput exceeds the limits of a single master server, you must partition the database across multiple physical machines. This is called **Sharding**.

### Sharding Methodologies

1. **Range-Based Sharding:** Partitioning data based on ranges of an attribute (e.g., routing users with IDs 1–1,000,000 to Shard A, and 1,000,001–2,000,000 to Shard B).

   - **Trade-off:** Simple to implement but leads to severe write imbalances if activity is concentrated in a specific range.
2. **Hash-Based Sharding:** Applying a hash function to the partition key:
   
   ```
   Shard ID = hash(key) % N
   ```
   
   - **Trade-off:** Uniform data distribution. However, if the number of shards $N$ changes (re-sharding), almost all historical data must be migrated.
3. **Directory-Based Sharding:** Utilizing a centralized lookup service (lookup table) to track which shard stores a specific partition key.

![Database Sharding Strategies — Range, Hash, and Directory Based](editions/python/chapters/18-database-compliance/visuals/sharding_strategies.jpg){width=85%}

   - **Trade-off:** Flexible, but introduces a single point of failure and query latency bottleneck at the lookup layer.


## Indexing Deep-Dive & Performance Optimization

Database indexes are critical for search speed, but they carry a write cost. Every index added increases database write latency and storage requirements.

### Index Types

- **B-Tree Indexes (Default):** Balanced search trees. Optimized for exact matches, range queries, and sorted order retrieval.
- **Hash Indexes:** Use hash tables. Optimized *only* for exact matches (`=`). Do not support range queries or sorting.
- **Inverted Indexes (GIN/GiST):** Used for full-text search and complex document data structures (like JSONB fields in PostgreSQL).

### Index Design Guidelines

1. **Covering Indexes:** An index that contains all columns required for a specific query. If a query selects columns `A` and `B` from a table, creating a composite index on `(A, B)` allows the database engine to retrieve the values directly from the index tree, bypassing the primary table pages entirely.
2. **Composite Index Left-Prefix Rule:** A composite index on `(columnA, columnB)` can be used to optimize queries searching by `columnA`, or `columnA AND columnB`. However, it *cannot* optimize queries searching only by `columnB`. Order your composite index columns based on query frequency.
3. **Write Amplification:** Avoid indexing columns that are updated frequently. Doing so forces the database to rewrite index pages constantly, degrading overall write performance.


## PCI-DSS Compliance & Tokenization

The Payment Card Industry Data Security Standard (PCI-DSS) imposes strict requirements on the handling of Primary Account Numbers (PANs). Storing raw 16-digit card numbers in your main application database is a major security risk and forces your entire infrastructure to fall within the scope of costly annual PCI audits.

### The Tokenization Pattern
To minimize audit scope, you must implement **Tokenization**:

1.  **Card Vault:** A separate, highly secure, network-isolated database (the Vault) that maps a PAN to a randomly generated, non-reversible **Token** (e.g., `tok_9a8b7c`).
2.  **Encryption:** Inside the Vault, PAN data is encrypted using AES-256-GCM before storage.
3.  **Application Separation:** The main billing and ledger applications only store and reference the token. Since they never store, process, or transmit raw card data, they are kept outside the scope of PCI-DSS regulations.

![PCI-DSS Tokenization Vault Architecture](editions/python/chapters/18-database-compliance/visuals/tokenization_vault.png){width=85%}

The following utility demonstrates the encryption standard (AES-256 in Galois/Counter Mode) required for encrypting PANs or PII:

```python
import base64
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class TokenizationUtility:
    """
    Utility for AES-GCM 256-bit encryption/decryption of sensitive PII or PAN data,
    adhering to PCI-DSS requirements.
    """
    
    @staticmethod
    def encrypt(plaintext: str, key_bytes: bytes) -> str:
        if not plaintext or len(key_bytes) != 32:
            raise ValueError("Invalid plaintext or key size. Key must be 256-bit.")
            
        # 1. Generate a secure random Initialization Vector (IV)
        iv = os.urandom(12)
        
        # 2. Encrypt using AES-GCM
        aesgcm = AESGCM(key_bytes)
        ciphertext = aesgcm.encrypt(iv, plaintext.encode('utf-8'), None)
        
        # 3. Combine IV and Ciphertext and base64-encode
        payload = iv + ciphertext
        return base64.urlsafe_b64encode(payload).decode('utf-8').rstrip('=')

    @staticmethod
    def decrypt(base64_payload: str, key_bytes: bytes) -> str:
        if not base64_payload or len(key_bytes) != 32:
            raise ValueError("Invalid payload or key size. Key must be 256-bit.")
            
        # 1. Pad and decode the base64 string
        missing_padding = len(base64_payload) % 4
        if missing_padding:
            base64_payload += '=' * (4 - missing_padding)
        encrypted_payload = base64.urlsafe_b64decode(base64_payload.encode('utf-8'))
        
        if len(encrypted_payload) < 12:
            raise ValueError("Ciphertext payload is truncated or invalid.")
            
        # 2. Extract IV and Ciphertext
        iv = encrypted_payload[:12]
        ciphertext = encrypted_payload[12:]
        
        # 3. Decrypt using AES-GCM
        aesgcm = AESGCM(key_bytes)
        decrypted_bytes = aesgcm.decrypt(iv, ciphertext, None)
        return decrypted_bytes.decode('utf-8')
```


GCM (Galois/Counter Mode) is preferred over CBC (Cipher Block Chaining) because it provides both **confidentiality** and **integrity (authenticity)**. It appends an authentication tag that prevents attackers from modifying the ciphertext in transit.


## GDPR vs. Immutable Ledgers: Crypto-Shredding

A major conflict exists in modern database design between audit compliance (SOC2) and data privacy regulations (GDPR/CCPA):

- **SOC2 Requirement:** Maintain an immutable, append-only, cryptographically chained audit log that can never be modified or deleted.
- **GDPR Requirement:** The **Right to be Forgotten**. Users can request that all of their personal identifiable information (PII) be permanently deleted from your databases.

### The Solution: Cryptographic Erasure (Crypto-Shredding)
Because you cannot delete a user's record from an immutable ledger (as doing so would break the cryptographic chain), you must apply **Crypto-Shredding**:

1. When a user is created, generate a unique, user-specific encryption key (e.g., AES-256 key).
2. Store the user's key in a secure Key Management Service (KMS) or Vault database.
3. All PII data written to the immutable ledger is encrypted using that specific user's key.
4. When a user submits a GDPR deletion request, **destroy the user's specific key from the KMS**.
5. Once the key is destroyed, the encrypted PII in the immutable ledger becomes mathematical noise that can never be decrypted again. This is legally accepted as a permanent deletion under GDPR compliance while keeping the ledger chain intact.


## SOC2 Audit Trails & Immutable Ledgers

For compliance frameworks like SOC2, you must maintain a tamper-proof audit trail of all financial actions.

### Design of a Tamper-Proof Audit Log

1.  **Append-Only Tables:** Database permissions should restrict application users to `INSERT` queries on audit tables, preventing `UPDATE` or `DELETE` operations.
2.  **Cryptographic Chaining:** Each audit log row should contain a cryptographic hash of the current row and the previous row's hash (similar to a blockchain ledger). If an attacker modifies a historical row, the chain break is instantly detectable during audit validation.
3.  **Immutable Databases:** Utilize native ledger databases (like Amazon QLDB) or WORM (Write Once, Read Many) storage to mathematically guarantee data immutability.

![Cryptographic Audit Trail Chain](editions/python/chapters/18-database-compliance/visuals/audit_trail.png){width=85%}


### Mock Interview Transcript: PCI-DSS and GDPR Compliance

> **Interviewer:** Design a database schema for a financial system that must comply with PCI-DSS and GDPR. How do you approach the storage of sensitive data?
> **Candidate:** For PCI-DSS, the most critical step is reducing the audit scope. I would implement a tokenization vault. The main transaction ledger would only store a non-reversible token. The actual Primary Account Numbers (PANs) are stored in an isolated, highly secured vault database, encrypted at rest using AES-256-GCM.
> **Interviewer:** That handles PCI. What about GDPR and the Right to be Forgotten?
> **Candidate:** For GDPR, we need to guarantee deletion of PII. However, our financial ledgers must remain immutable for SOC2 compliance.
> **Interviewer:** Exactly. How do you handle a GDPR deletion request for data that's referenced in immutable audit logs?
> **Candidate:** Good question, I hadn't thought about the audit log specifically... Ah, we can use crypto-shredding. When a user is created, we generate a unique KMS encryption key for their PII. We encrypt their PII before writing it to the immutable ledger. When a GDPR deletion is requested, we permanently destroy their specific key in the KMS. The audit log remains cryptographically unbroken, but the PII becomes unrecoverable mathematical noise.
> **Interviewer:** How do you ensure the key itself isn't compromised?
> **Candidate:** We'd enforce strict IAM roles, ensuring only the encryption service can access the KMS, and we'd log every decryption request to a separate, append-only CloudTrail log. 
> **Interviewer:** Very solid. 

**Technical Summary:** The candidate successfully navigated conflicting compliance requirements by decoupling sensitive data via a tokenization vault (PCI-DSS) and employing crypto-shredding (GDPR) to satisfy deletion mandates without compromising the immutability of financial audit trails.


## Hardening the Data Tier & Audits

Securing financial databases requires separating database engine administration from data access. During audits, one of the most critical security principles you must demonstrate is **perimeter isolation and credential partitioning**.

### Connection Pool & Infrastructure Hardening

1.  **Network Isolation (VPC):** The database should never reside in a public subnet. Access must be restricted via security groups to specific application servers residing in private subnets.
2.  **IAM-Based Authentication:** Instead of hardcoding database credentials or using long-lived passwords, utilize temporary IAM credentials (like AWS IAM Database Authentication) or secure secret rotation services (like HashiCorp Vault) with a 30-day rotation policy.
3.  **Connection Saturation & Timeouts:** To prevent Denial of Service (DoS) attacks or query starvation, configure pool limits strictly. Enforce a connection timeout of 250ms and a max life time of 30 minutes to recycle leaked database connections.

### Mock Audit Scenario Drill

Here is a mock review dialog during a SOC2 compliance audit:

**Auditor:** *"How do you guarantee that a database administrator (DBA) or a developer with access to the raw database files cannot read credit card numbers or sensitive transactional data?"*

**Candidate:** *"All card numbers are stored inside a dedicated Card Vault. The PAN (Primary Account Number) is encrypted inside the vault using AES-256-GCM. The encryption key is stored in a dedicated Key Management Service (KMS) with access restricted via IAM roles that only the Vault service application user can assume. 

Even if a DBA has root access to the database tables or extracts a raw disk backup, they cannot decrypt the PAN fields because they do not have decryption permissions on the KMS key. Furthermore, every decryption call is logged in an append-only audit trail in CloudTrail, which triggers instant alerts on unauthorized access attempts."*



> ⭐ **STAR Moment: The Security-First Architecture**
> 
> In a system design interview, explain the concept of *"auditing and perimeter isolation."* Show how you can use a separate network zone (VPC) for your Card Vault, with separate encryption keys managed by an HSM (Hardware Security Module) or Key Management Service (KMS), and separate access control roles. Decoupling data in this way reduces security risk and simplifies compliance audits.


# Behavioral Leadership and Technical Communication

> *"At a senior or executive level, your value is no longer measured by the quantity of code you write, but by your ability to align teams, navigate architectural tradeoffs, and resolve production crises with composure."*


## Technical vs. Behavioral Alignment

When interviewing for a senior, staff, or engineering manager role, clearing the coding and system design rounds is only half the battle. In-person or Teams video calls inevitably culminate in a behavioral evaluation. 

At this level, the interviewer assumes you possess technical competence. The behavioral round is designed to evaluate your **leadership, system ownership, conflict resolution, execution speed, and architectural maturity**. If you respond to situational questions with generic answers (e.g., *"I am a team player who works hard"*), you fail to demonstrate the maturity required to lead engineering organizations.

In this chapter, we adapt the classic **STAR (Situation, Task, Action, Result)** model into a technical-leadership narrative framework, providing mock response transcripts for common senior scenarios.


## The Technical STAR Framework

To present your career achievements effectively, structure your behavioral narratives around technical metrics and architectural trade-offs:

![The Technical STAR Framework](editions/python/chapters/19-behavioral-leadership/visuals/technical_star.png){width=90%}

> **How to apply the framework:**
>
> *   **Situation (S):** Establish the business scale and constraints. What was the starting state (e.g., transaction volume, bottlenecks, legacy limitations)?
> *   **Task (T):** Define the architectural objectives, SLA requirements, and the technical scope of what you were responsible for delivering (e.g., migrate the ledger to a CP database while maintaining 99.99% availability).
> *   **Action & Trade-offs (A):** Describe the design options you evaluated, the trade-off decisions you made, and how you led the team through implementation.
> *   **Result & Impact (R):** Present quantitative, data-driven outcomes. Never say: *"We made the system faster."* Say: *"We reduced p99 write latency by 45%, eliminated database locks, and passed the SOC2 compliance audit with zero findings."*


## Mock Scenario A: Architectural Disagreement (Lead / Staff Perspective)

**Interviewer:** *"Tell me about a time you had a major disagreement with a peer or stakeholder about a technical design. How did you resolve it?"*

### The Strategy
A junior candidate focuses on the personal conflict or tries to prove they were "right." A senior candidate frames the resolution around data-driven trade-off analysis, prototype benchmarks, and collaborative consensus-building.

### The Response Transcript
> *"In my previous role at ZenithTrade, my team was tasked with scaling our matching engine to handle a 5x spike in transaction volume. A principal architect proposed rewriting our processing loops using a reactive programming model (Spring WebFlux). I had serious concerns about the operational overhead of reactive code, specifically debuggability, stack trace readability, and the steep learning curve for our support engineers.*
>
> *The first approach I proposed actually failed to gain traction because I didn't provide enough empirical data. Realizing this, I pivoted and suggested a 3-day time-boxed prototyping run. My senior engineer Sarah and I built two benchmark pipelines: one using the proposed reactive model, and another using Java 21's new Virtual Threads (Project Loom).*
>
> *The prototype metrics revealed that while both models handled the required 20,000 concurrent requests without thread exhaustion, the virtual threads implementation reduced CPU utilization by 15% and preserved our existing synchronous debugging tools.*
>
> *I presented these findings in an architecture review document. Leadership was skeptical until they saw the raw trace logs side-by-side. The principal architect agreed with the data, and we proceeded collaboratively with the Virtual Threads design. In hindsight, I would have prototyped sooner rather than debating theory. The system successfully launched, sustaining 5x load with zero stability incidents."*


## Mock Scenario B: Production Crisis Management (Engineering Manager Perspective)

**Interviewer:** *"Describe a major production outage you managed. How did you coordinate the response and prevent it from happening again?"*

### The Strategy
Focus on command composure, blameless post-mortem culture, and root-cause remediation rather than pointing fingers or downplaying the event.

### The Response Transcript
> *"During a high-volume retail promotion on AuraPay, our ledger database connection pool saturated, causing transaction failures for approximately 15% of our users. As the Engineering Manager, I immediately initiated our incident response protocol. We hit a wall when the initial metrics didn't point to any specific query, so the team collectively decided to split up: one engineer analyzing database metrics, one reviewing application logs, and a product manager handling external client communications.*
>
> *We eventually identified that our connection pool size was set to 200, which was starving the database CPU with constant thread context switching. I instructed the team to apply the HikariCP pool sizing formula, reducing the connection limit to 30. This immediately stabilized database CPU utilization from 98% down to 42%, restoring transaction flow.*
>
> *To prevent future occurrences, I led a blameless post-mortem. We discovered that a recent release had introduced a database query inside a parallel stream pipeline, starving the common ForkJoinPool. What I learned from that failure was the importance of strict code boundaries. We refactored the stream to execute asynchronously outside the transaction boundary. Since then, our system uptime has remained at 99.99% under peak promotional events."*


## Mock Scenario C: Balancing Technical Debt vs. Features (Director Perspective)

**Interviewer:** *"How do you balance business pressure for new features against the technical necessity of refactoring legacy code?"*

### The Strategy
Frame technical debt as a financial risk to the business. Show that you can speak the language of product managers and executives, translating code quality into operational velocity.

### The Response Transcript
> *"When I joined ChiramTrust, the identity consent module was built as an anemic domain model with scattered business logic. Product management wanted to launch three new OAuth integrations within two months, but our engineering velocity was bottlenecked because every minor change broke unrelated validation paths.*
>
> *I knew that pushing features without refactoring would increase our defect rate in production. I met with the VP of Product and translated our technical debt into business risk. The product manager pushed back because of the strict timeline, arguing we couldn't afford a pause.*
>
> *I proposed a compromise: we would dedicate 30% of our capacity in the next two sprints to refactor the consent model into an encapsulated aggregate root. The remaining 70% would be spent on the integration layouts. The team collectively decided this was the most pragmatic path forward.*
>
> *The team successfully executed the refactor, removing setters and enclosing the invariants inside the domain objects. In hindsight, I would have involved QA earlier in the refactor planning, but the outcome was still solid. This reduced our regression bug rate to less than 2% and actually accelerated the development of the final two integrations, allowing us to launch the features a week ahead of the original deadline."*


### Scenario 4: Managing Underperformance
**Interviewer:** Tell me about a time you had to manage an underperforming team member.

**Candidate:** Six months into my role as engineering lead, one of our senior developers—let's call him Alex—had missed three consecutive sprint commitments. Rather than jumping to a PIP, I scheduled a private 1:1 to understand the root cause. It turned out Alex was struggling with our migration from monolith to microservices and felt embarrassed to ask for help after 8 years at the company.

I paired him with our most patient architect for bi-weekly knowledge transfer sessions and adjusted his sprint load to 70% for six weeks. I was transparent with the team that Alex was ramping on the new architecture without singling him out. Within two months, Alex was not only back to full velocity but had become our go-to person for the data migration layer because he understood both the old and new systems intimately.

The lesson I took away: underperformance is usually a symptom, not a character flaw. Diagnosing the root cause before applying a remedy saved us from losing an incredibly valuable engineer.


### Scenario 5: Leading a Project Pivot
**Interviewer:** Describe a time when you had to pivot a project mid-execution.

**Candidate:** Our team had spent five weeks building a custom real-time analytics dashboard when our VP of Product shared early results from a customer advisory board: customers wanted pre-built compliance reports, not custom dashboards. My first reaction was frustration—we'd invested significant effort. But after sleeping on it, I realized the data pipeline we'd built was reusable.

I called a team retrospective and was honest: "The analytics engine we built is solid, but the UI layer needs to pivot to templated reports." One engineer pushed back hard, feeling her frontend work was wasted. I acknowledged that directly and proposed we salvage her component library for the new report designer.

We re-scoped to a 3-week sprint, reusing 60% of the backend. The compliance reports shipped on time and became our highest-adopted feature that quarter. What I learned: pivot announcements need to honor the work already done, not just dictate the new direction.


## Checklist for Video (Teams) & In-Person Technical Interviews

To project executive presence and clear technical rounds on live video calls or in-person sessions, adhere to these guidelines:

1. **The Virtual Whiteboard Technique:** On Teams calls, do not just talk. Utilize a digital whiteboard (like Miro or Excalidraw) to draw bounded contexts, Saga flows, and database sharding rings. Visual diagrams make your architecture concrete and easy for the interviewer to follow.
2. **The Clarification Pause:** When presented with a coding problem, do not write code immediately. Pause for 2-3 minutes to write down the pre-conditions, post-conditions, and input/output types as comments. This shows structural discipline and prevents off-by-one errors.
3. **The Trade-Off Verbalization:** Throughout the interview, constantly verbalize your architectural trade-offs (e.g., *"If we use Redis for rate limiting, we gain speed, but we must handle memory expiration and potential write consistency issues during partition events"*). Never present a design as "perfect."


> ⭐ **STAR Moment: Speak in Metrics**
> 
> When presenting your career accomplishments, translate every engineering activity into a business outcome. Never say: *"I rewrote the database queries."* Say: *"I optimized our query indexes, reducing database read latency by 60% and cutting our monthly database hosting cost by $12,000."* Executives and engineering leaders hire developers who understand the financial and operational impact of their code.


# Testing and CI/CD Strategies for High-Performance Systems

> *"The quality of your production system is a direct reflection of your automated validation boundaries. If you cannot test it in isolation, you cannot trust it at scale."*


## The Testing Paradigm in Senior Interviews

In technical interviews for lead, staff, or engineering manager roles, coding challenges do not end with a working algorithm. The interviewer will ask: *"How do you test this code? How do you ensure this does not break in production? What is your strategy for validating microservice API contracts?"*

Many candidates respond with simple unit tests. However, a senior candidate must present a structured **Testing Pyramid** strategy, showing how they balance unit tests with Testcontainers-based integration tests, API contract tests, and continuous delivery (CI/CD) verification.

![The Technical Testing Pyramid](editions/python/chapters/20-testing-cicd/visuals/testing_pyramid.png){width=80%}


## The Testing Pyramid

An effective testing strategy separates validation boundaries into three layers, balancing execution speed and operational cost against validation fidelity:

### Unit Testing with Abstractions
Unit tests are the foundation of the pyramid. They validate the internal logic of a single class in isolation, replacing all external infrastructure dependencies (such as databases and network gateways) with mock interfaces.

- **Velocity:** Execute in milliseconds.
- **Boundary:** Focuses purely on code correctness and invariant compliance.
- **Coverage:** High code coverage (90%+), testing all logical paths and edge cases.

The following code illustrates unit testing our decoupled `TransactionProcessor` by mocking its repository and notification interfaces:

```python
import unittest
from unittest.mock import Mock, ANY
from decimal import Decimal

class TestTransactionProcessor(unittest.TestCase):
    def test_successful_transfer_enforces_invariants(self):
        # Arrange Mock Dependencies
        mock_repo = Mock()
        mock_calculator = Mock()
        mock_sender = Mock()

        source = LedgerAccount("acc-source", Decimal("100.00"), "USD")
        destination = LedgerAccount("acc-dest", Decimal("50.00"), "USD")

        mock_repo.find_by_id.side_effect = lambda id: source if id == "acc-source" else destination
        mock_calculator.calculate_fee.return_value = Decimal("0.00")

        processor = TransactionProcessor(mock_repo, mock_calculator, mock_sender)

        # Act
        processor.process_transfer("acc-source", "acc-dest", Decimal("30.00"))

        # Assert state invariants updated
        self.assertEqual(Decimal("70.00"), source.get_balance())
        self.assertEqual(Decimal("80.00"), destination.get_balance())

        # Assert repository saved both
        mock_repo.save.assert_any_call(source)
        mock_repo.save.assert_any_call(destination)
        mock_sender.send_notification.assert_called_with(ANY)
```


By utilizing mock objects, we verify that the processor correctly coordinates the transfer, updates balance invariants, and calls the persistence layer, without requiring an active database connection.

> **Why is it called "Mockito"?** The popular Java mocking framework is named after the **Mojito** cocktail — a playful twist by its Polish creator Szczepan Faber. Just as a bartender mixes ingredients to create something refreshing, Mockito mixes stubs and verifications to create clean, readable tests. The name also echoes the Spanish suffix *"-ito"* (meaning "little"), suggesting lightweight mock objects.

### Integration Testing with Testcontainers
While unit tests verify logic correctness, they cannot detect SQL query syntax errors, schema constraint violations, or message serialization bugs. For this, you need **Integration Tests**.
In system design interviews, describe **Testcontainers**:

- **Mechanism:** During test execution, the testing framework utilizes Docker to spin up actual database instances (PostgreSQL, MySQL) or brokers (Kafka, RabbitMQ) in local containers.
- **Assertion:** The test runs against real database engines, validating database locks, unique constraint violations, and transaction rollbacks. Once execution completes, the container is destroyed.
- **Impact:** Eliminates the anti-pattern of testing against mock database structures (e.g., using H2 in-memory database for testing PostgreSQL code, which misses Postgres-specific syntax or transaction behaviors).

### API Contract Testing (Pact)
In a microservices mesh, service dependencies are the primary source of integration failures (e.g., the User Service changes an API field name, breaking the Billing Service). 
To prevent this without the latency of End-to-End (E2E) testing, implement **Consumer-Driven Contract Testing (Pact)**:

- **Contract File:** The consumer service defines its API requests and expected responses in a contract file.
- **Provider Verification:** The provider service runs automated tests against this contract file to verify compliance. If the provider modifies its API in a way that violates the contract, the build fails before deployment.

> **Why is it called "Pact"?** A pact is a formal agreement between two parties. In contract testing, the consumer and provider make a *pact* — a machine-readable agreement about the API's shape. If either side breaks the pact, the build fails. The name was chosen by the team at REA Group (Australia) to emphasize that API compatibility is a mutual commitment, not a one-sided assumption.


## Advanced Testing Techniques

### Mutation Testing
Code coverage alone is a misleading metric. A test suite can achieve 100% line coverage while asserting nothing meaningful. **Mutation Testing** validates the *quality* of your assertions:

1. The mutation framework (e.g., PIT for Java, Stryker for JavaScript/C#) systematically injects small bugs ("mutants") into your production code — flipping comparison operators, negating boolean returns, removing method calls.
2. Your test suite is then executed against each mutant.
3. If your tests **fail** (detecting the mutant), the mutant is "killed" — your tests are effective.
4. If your tests **pass** despite the mutation, the mutant "survives" — your tests are weak and missed a real defect scenario.

A **mutation score** above 85% indicates a robust test suite. In interviews, mentioning mutation testing immediately signals that you understand test quality beyond surface-level coverage metrics.

### Property-Based Testing
Traditional unit tests validate specific hand-picked input-output pairs. **Property-Based Testing** (e.g., jqwik for Java, Hypothesis for Python, FsCheck for C#) generates thousands of random inputs and verifies that invariants hold for all of them:

- **Example:** For a `Money.add()` method, assert that `a.add(b).equals(b.add(a))` (commutativity) and `a.add(Money.ZERO).equals(a)` (identity) for any randomly generated amounts and currencies.
- **Power:** Discovers edge cases that human-authored tests miss — overflow boundaries, empty collections, unicode strings, negative values.

### Chaos Engineering
At the staff/principal level, you are expected to design systems that survive infrastructure failures. **Chaos Engineering** proactively injects failures into production-like environments to validate resilience:

- **Network Partitions:** Simulate network splits between services to verify Circuit Breaker activation and graceful degradation.
- **Latency Injection:** Add artificial delays (e.g., 5-second response times from a database) to verify timeout configurations and bulkhead isolation.
- **Instance Termination:** Randomly kill service instances to test auto-scaling recovery and consumer group rebalancing.
- **Tools:** Netflix Chaos Monkey, AWS Fault Injection Simulator, Gremlin, LitmusChaos.

> **Interview Signal:** When asked *"How do you ensure reliability?"*, answering *"We run quarterly game days using chaos engineering to validate our circuit breakers and consumer group rebalancing under simulated Kafka broker failures"* demonstrates operational maturity far beyond unit testing.


## Feature Flags and Progressive Delivery

Modern release engineering decouples **deployment** (shipping code to production) from **release** (enabling features for users):

### Feature Flag Architecture

- **Implementation:** Wrap new features behind boolean flags stored in a centralized configuration service (LaunchDarkly, Unleash, or a custom Redis-backed service).
- **Granularity:** Flags can target individual users (beta testers), user segments (enterprise tier), geographic regions, or percentages of traffic.
- **Kill Switch:** If a new feature causes errors in production, disable the flag instantly without rolling back the entire deployment.
- **Technical Debt:** Feature flags must have an expiration policy. Stale flags left in the codebase for months create branching complexity and testing overhead. Enforce cleanup sprints.

### Trunk-Based Development
Feature flags enable **trunk-based development** — all engineers commit directly to the main branch. There are no long-lived feature branches:

- Every commit is deployed to production behind a flag.
- Integration conflicts are caught immediately instead of during painful merge events weeks later.
- Release cadence accelerates from weekly to multiple daily deployments.


## Continuous Integration (CI/CD) Compliance Pipeline

A senior engineering leader does not rely on developers remembering to run tests. Quality checks must be automated inside a **CI/CD Pipeline** before merging code to main branches:

### Automated Pipeline Checks

1. **Linting & Code Formatting:** Ensures consistent style guidelines across the team.
2. **Static Application Security Testing (SAST):** Scans source code for potential vulnerabilities (e.g., SQL injections, insecure cryptographic configurations, hardcoded API keys) using tools like SonarQube.
3. **Dependency Scanning:** Scans imports for known security vulnerabilities (CVEs) and license compliance issues.
4. **Automated Unit & Integration Execution:** Blocks pull request merges if any test fails or if coverage drops below the required threshold.
5. **Mutation Score Gate:** Block merges if the mutation score drops below 80%, ensuring new code has meaningful test coverage.
6. **Contract Verification:** Run Pact provider verification against all consumer contracts before deploying API changes.

### Pipeline as Code
Define your entire CI/CD pipeline in version-controlled configuration files (e.g., GitHub Actions YAML, Jenkinsfile, GitLab CI):

- **Reproducibility:** Any engineer can trace exactly what checks ran for any commit.
- **Auditability:** SOC2 compliance requires evidence that all production releases passed automated security and quality gates. Pipeline-as-code provides this audit trail automatically.


## Modern Deployment Strategies

Once the CI/CD pipeline validates code correctness, releasing the software to production requires strategies that minimize user impact during updates:

### Blue-Green Deployments
Maintain two identical physical production environments:

- **Blue Environment:** Actively hosts production traffic.
- **Green Environment:** Hosts the new code release.
- **Switch:** Once validation tests pass on the Green environment, the load balancer switches traffic from Blue to Green. If a rollback is needed, the switch routes traffic back immediately.

### Canary Deployments
Deploy the new release to a small subset of production instances (e.g., routing 2% of user traffic to the new version).

- **Monitoring:** Monitor error rates, latency metrics, and resource utilization on the canary instances.
- **Scale:** If metrics remain stable, gradually scale traffic routing to 10%, 50%, and finally 100% of instances, destroying the old version.
- **Automated Rollback:** Configure automated rollback triggers — if the canary's error rate exceeds 1% or p99 latency increases by more than 200ms, automatically shift all traffic back to the stable version.

### Rolling Deployments
Update instances one at a time (or in small batches) behind the load balancer:

- Each instance is drained of active connections, updated, health-checked, and re-registered.
- Slower than blue-green but requires no duplicate infrastructure.
- Best suited for stateless microservices with fast startup times.


> ⭐ **STAR Moment: The Mocking Boundary**
> 
> In a technical interview, emphasize that you know *when* to mock. Say: *"We mock network calls and database interfaces in our unit tests to keep feedback loops fast. But we never mock our domain aggregates or value objects. Testing our business rules against actual domain structures guarantees that our core invariants are always enforced. For integration boundaries, we use Testcontainers against real Postgres and Kafka instances, and we validate API contracts using Pact before every deployment."* This shows you understand domain boundary protection and production-grade testing strategy.

### Mock Interview Transcript: Microservices Testing Strategy

> **Interviewer:** How would you design a testing strategy for a microservices architecture with 30+ services?
> **Candidate:** I'd structure it around the testing pyramid. We'd have extensive unit tests for domain logic. For integration boundaries, we'd use Testcontainers to spin up real databases locally. To manage the 30+ services communicating, we'd rely heavily on consumer-driven contract testing using Pact to ensure API compatibility without spinning up the entire mesh.
> **Interviewer:** How do you test cross-service transactions, like a payment saga that hits five different services?
> **Candidate:** For complex sagas, relying only on contract tests isn't enough. We'd need a targeted End-to-End test environment, but to avoid flakiness, we'd test the saga orchestrator specifically by mocking the participant responses, and then rely on synthetic monitoring in production. 
> **Interviewer:** What if a deployment passes all tests but still causes issues in production? What's your rollback strategy?
> **Candidate:** Actually, let me reconsider the standard pipeline... Instead of just relying on rollbacks, we should use feature flags and progressive delivery. We deploy the new code hidden behind a flag. We turn it on for 1% of users—a canary deployment. If error rates spike, we just toggle the flag off. It's much faster and safer than a full infrastructure rollback.
> **Interviewer:** And how do you ensure the system is resilient to infrastructure failures?
> **Candidate:** We'd employ chaos engineering. During off-peak hours, we randomly terminate instances or inject network latency to ensure our circuit breakers and bulkheads work as designed.

**Technical Summary:** The candidate demonstrated a mature understanding of testing at scale by emphasizing contract testing over brittle E2E tests, utilizing feature flags for rapid canary rollbacks, and incorporating chaos engineering to proactively validate system resilience.

## Performance Testing & Load Validation

Performance testing is a critical step in CI/CD pipelines to ensure systems remain reliable under expected and unexpected traffic. Rather than waiting for production outages, modern engineering teams validate performance continuously using different load profiles.

### Types of Performance Tests

1. **Load Testing** — Validate system behavior under expected peak load

   - **Goal:** verify response times and throughput meet SLAs under normal-to-peak traffic
   - **Example:** simulate 10,000 concurrent users on an e-commerce checkout API
   - **Key metrics:** p50/p95/p99 latency, requests/second, error rate

2. **Stress Testing** — Find the breaking point

   - **Goal:** push beyond expected load to discover where the system degrades or fails
   - **Example:** gradually increase from 10K to 100K concurrent users until error rate exceeds 5%
   - **Key insight:** identify the bottleneck (CPU? memory? DB connections? network?)

3. **Soak Testing (Endurance Testing)** — Detect memory leaks and resource exhaustion

   - **Goal:** run sustained moderate load for 4-24 hours
   - **Catches:** memory leaks, connection pool exhaustion, log file disk filling, GC pressure

4. **Spike Testing** — Validate auto-scaling and recovery

   - **Goal:** suddenly surge traffic (e.g., 0 to 50K users in 30 seconds) and verify recovery
   - **Catches:** auto-scaling lag, cold-start penalties, circuit breaker activation

### Performance Testing Tools

When selecting a tool, consider the protocols supported and how well it integrates into CI/CD pipelines:

| Tool | Language | Protocol Support | Distributed | Best For |
|---|---|---|---|---|
| k6 (Grafana) | JavaScript | HTTP, gRPC, WebSocket | Yes (k6 Cloud) | Developer-friendly, CI/CD integration |
| JMeter | Java/XML | HTTP, JDBC, JMS, LDAP | Yes | Enterprise, complex protocols |
| Gatling | Scala/Java | HTTP, WebSocket | Yes | High-performance simulation |
| Locust | Python | HTTP | Yes | Python teams, custom load patterns |
| Artillery | JavaScript | HTTP, WebSocket, Socket.IO | Yes (Cloud) | Serverless, quick setup |

### k6 Load Test Example

The following is a concrete k6 script example written in JavaScript. It demonstrates how to define load stages and enforce SLAs through thresholds:

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Hold at 100 users  
    { duration: '2m', target: 500 },   // Ramp up to 500 users
    { duration: '5m', target: 500 },   // Hold at 500 users
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<250'],  // 95% of requests under 250ms
    http_req_failed: ['rate<0.01'],    // Error rate under 1%
  },
};

export default function () {
  const res = http.get('https://api.example.com/orders');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  sleep(1);
}
```

### SLA Validation in CI/CD Pipelines

Integrating performance tests into CI/CD ensures that latency and throughput regressions are caught before they reach production:

- Run load tests as a pipeline stage AFTER integration tests pass
- Define performance budgets as code (thresholds in k6/Gatling config)
- Fail the build if p95 latency exceeds SLA or error rate exceeds threshold
- Store results in a time-series database (InfluxDB/Prometheus) for trend analysis
- Alert on performance regressions compared to the previous release baseline

### Performance Anti-Patterns

When designing load tests, avoid these common mistakes:
1. **Testing in non-production environments** — hardware differences invalidate results
2. **Not warming up the JVM** — JIT compilation skews early measurements (add a warm-up stage)
3. **Ignoring connection pooling** — each virtual user should reuse connections like production clients
4. **Measuring averages instead of percentiles** — p99 matters more than mean (a few slow requests hide behind good averages)
5. **Not testing database under load** — the DB is usually the bottleneck, not the application server

### HikariCP Connection Pool Sizing Under Load

When load testing data-intensive applications, connection pool sizing is a common bottleneck. As discussed in earlier chapters, the optimal pool size formula is:

```
Pool Size = Tn * (Cm - 1) + 1
```

Where $T_n$ = number of threads, $C_m$ = maximum concurrent queries per thread.

Under-provisioning the pool causes thread starvation, and over-provisioning wastes database connections.


# Distributed Event Streaming and Message Brokers

> *"An event log is the ultimate source of historical truth. In a distributed architecture, message brokers serve as the central nervous system, routing states across service boundaries."*


## Event Streaming in System Design

In microservice architectures, services must communicate asynchronously. Interview candidates often default to saying: *"We will send a message via Kafka."* 

If you stop there, you miss the opportunity to demonstrate depth. A senior systems architect must explain how the message broker is structured, how partition keys guarantee message ordering under concurrency, and how to achieve **Exactly-Once Semantics (EOS)** across transactions.

In this chapter, we deep-dive into Apache Kafka's storage internals and partition routing mechanics, showing how AuraPay shards event streams to maintain ledger correctness.

![Apache Kafka Topic Partitions and Consumer Groups](editions/python/chapters/21-message-brokers/visuals/kafka_internals.png){width=90%}


## Apache Kafka Internals & Sharding

Apache Kafka is designed as a distributed, partitioned, commit log. Understanding its storage structure is critical for scaling system throughput:

> **Why is it called "Kafka"?** LinkedIn engineer Jay Kreps named it after **Franz Kafka**, the Czech novelist famous for writing about surreal, labyrinthine bureaucracies. Kreps chose the name because Kafka is *"a system optimized for writing"* — and Franz Kafka was a writer. The literary nod is fitting: just as Kafka's novels depict characters navigating complex, opaque systems, Apache Kafka routes millions of messages through complex distributed topologies. The name stuck, and today "Kafka" is synonymous with high-throughput event streaming.

### Core Concepts

1. **The Commit Log:** A Kafka partition is an append-only, ordered sequence of records. Each record consists of a key, a value, and a timestamp. Records are immutable and assigned a sequential ID called an **offset**.
2. **Partitions:** Topics are divided into multiple partitions distributed across Kafka brokers. Partitions are the unit of scalability in Kafka: while a single partition can only handle a throughput limited by its host broker, multiple partitions allow parallel writes and reads across the cluster.
3. **Consumer Groups:** A consumer group is a collection of consumers working together to read messages from a topic. Kafka guarantees that each partition is assigned to exactly *one* consumer instance within a consumer group. This prevents duplicate processing of messages.

![Kafka Partitions and Consumer Group Parallelism](editions/python/chapters/21-message-brokers/visuals/kafka_partitions.jpg){width=85%}

### Replication and Durability
Each partition is replicated across multiple brokers for fault tolerance:

- **Leader Replica:** Handles all read and write requests for the partition.
- **Follower Replicas:** Passively replicate data from the leader. If the leader broker fails, a follower is promoted to leader via the controller election process.
- **ISR (In-Sync Replicas):** The set of replicas that are fully caught up with the leader. The producer configuration `acks=all` ensures a write is only acknowledged after all ISR replicas have persisted it, preventing data loss during broker failures.
- **Minimum ISR:** Setting `min.insync.replicas=2` with `acks=all` ensures at least two replicas must acknowledge a write. If only one replica is available, the broker rejects the write rather than risking data loss.


## The Ordering Invariant & Partition Keys

A common failure mode in message broker design is **out-of-order message delivery**. 
For example, if a user performs two actions:

1. Deposit \$100 (Event A)
2. Withdraw \$80 (Event B)

If the consumer processes Event B before Event A due to network concurrency, the withdrawal may be rejected due to insufficient funds, violating the system's business invariants.

### Guaranteeing In-Order Delivery
To guarantee in-order delivery, Kafka enforces a strict rule: **messages written to the same partition are always read in the exact order they were written.**

- If you publish messages without a key (null key), Kafka distributes them across partitions using a round-robin algorithm, losing all ordering guarantees.
- **The Solution:** Publish messages with a **Partition Key** (e.g., `accountId`). Kafka hashes the key to determine the partition:

```
Partition ID = hash(accountId) % Number of Partitions
```

By sharding on `accountId`, all transaction events for a specific account are guaranteed to land in the same partition and be processed in exact chronological order by a single consumer thread.

The following code illustrates this partition-key routing implementation in a Kafka producer:

```python
from confluent_kafka import Producer

class TransactionEventProducer:
    def __init__(self, bootstrap_servers: str, topic: str):
        config = {
            'bootstrap.servers': bootstrap_servers,
            'enable.idempotence': True,
            'acks': 'all'
        }
        self.producer = Producer(config)
        self.topic = topic

    def publish_event(self, account_id: str, event_json: str):
        # Shard by account_id to guarantee partition ordering
        self.producer.produce(
            self.topic, 
            key=account_id.encode('utf-8'), 
            value=event_json.encode('utf-8'),
            callback=lambda err, msg: print(f"Published: {msg.key()}") if not err else print(f"Error: {err}")
        )
        self.producer.poll(0)
```


Setting `enable.idempotence = true` ensures that network retries by the producer do not result in duplicate messages landing in the partition log.


## Exactly-Once Semantics (EOS)

In system design interviews, clearing the **Exactly-Once** challenge is a major differentiator.
How do you guarantee that a transaction is processed exactly once, even if the network fails midway?

Kafka achieves Exactly-Once Semantics (EOS) using a combination of three mechanisms:

1. **Idempotent Producers:** The producer appends a unique sequence number to each message. If a broker receives a duplicate sequence number due to a network retry, it discards the duplicate write.
2. **Transactional Writes:** When a service must read a message from an input topic, process it, and write the output to another topic (the Read-Process-Write pattern), Kafka allows wrapping these steps in a transaction.
3. **Offset Commits in DB:** Alternatively, when writing to a database (like AuraPay's ledger), combine the database write and the event offset commit inside a single database transaction. This is the **Transactional Outbox Pattern** (discussed in the Enterprise Integration and Resiliency chapter), ensuring that either both the database update and the event publish succeed, or both roll back.


## Consumer Group Rebalancing and Failure Recovery

When a consumer instance crashes or a new instance joins the group, Kafka triggers a **rebalance** — redistributing partition assignments across the remaining consumers:

### The Rebalancing Problem
During a rebalance, all consumers in the group temporarily stop processing. This "stop-the-world" pause can cause latency spikes in real-time systems.

### Mitigation Strategies

1. **Sticky Assignor:** Use the `StickyAssignor` partition assignment strategy. Unlike the default `RangeAssignor`, it minimizes partition movement during rebalances — consumers keep their existing assignments, and only the partitions owned by the departing consumer are redistributed.
2. **Cooperative Rebalancing:** Kafka 2.4+ supports **incremental cooperative rebalancing**, where only the affected partitions are revoked and reassigned. Non-affected consumers continue processing without interruption.
3. **Static Group Membership:** Assign a fixed `group.instance.id` to each consumer. When a consumer restarts within the `session.timeout.ms` window, Kafka recognizes it as the same member and skips the rebalance entirely.

### Dead Letter Queues (DLQ)
When a consumer repeatedly fails to process a message (e.g., due to a malformed payload or a downstream service outage), it must not block the entire partition:

- After a configurable number of retry attempts (e.g., 3), route the failed message to a **Dead Letter Queue** — a separate Kafka topic (e.g., `transactions.dlq`).
- The main consumer continues processing subsequent messages.
- A separate monitoring service reads the DLQ, alerts the operations team, and supports manual inspection and replay.


### Mock Interview Transcript: Consumer Group Rebalancing

> **Interviewer:** Your Kafka consumer group is experiencing rebalancing storms. The consumers keep dropping and rejoining, causing massive processing delays. How do you diagnose and fix this?
> **Candidate:** A rebalance storm usually means consumers are failing to send heartbeats or taking too long to process batches. I'd first check the `session.timeout.ms` and `max.poll.interval.ms` metrics. If our message processing is database-heavy, the consumer might exceed the poll interval, causing Kafka to assume it's dead. I'd tune `max.poll.records` down so the consumer processes smaller batches and polls more frequently.
> **Interviewer:** That stabilizes the group. But what if one partition has 10x the traffic of the others because of a highly active user?
> **Candidate:** That's a hot partition problem. Our partition key is likely skewed. Good question, I hadn't thought about skewed keys in this context... We could append a random salt to the key for that specific heavy user to distribute their events across partitions, though that breaks strict global ordering for them. If order is required, we'd need to scale vertically by increasing the consumer's thread pool, or optimizing the database writes.
> **Interviewer:** Let's say the rebalancing was caused by a malformed message crashing the consumer. How do you handle poison pill messages?
> **Candidate:** We wrap the deserialization and processing logic in a `try-catch` block. If a message fails validation after a few retries, we acknowledge the offset and forward the payload to a Dead Letter Queue (DLQ).
> **Interviewer:** How can we minimize the impact when we legitimately need to restart consumers for a deployment?
> **Candidate:** We'd enable static group membership by setting `group.instance.id`, and use the cooperative sticky assignor so only the partitions belonging to the restarting node are temporarily paused.

**Technical Summary:** The candidate effectively diagnosed rebalancing storms by identifying poll interval exhaustion, proposed Dead Letter Queues for poison pill messages, and utilized static group membership with cooperative rebalancing to minimize deployment disruptions. They correctly identified the trade-offs of handling hot partitions.


## Event Schema Evolution

As your system evolves, the structure of event payloads will change. Adding new fields, renaming properties, or changing data types can break downstream consumers if not managed carefully:

### Schema Registry (Confluent)

- **Central Registry:** All event schemas are registered in a **Schema Registry** (e.g., Confluent Schema Registry) using Avro, Protobuf, or JSON Schema formats.
- **Compatibility Modes:**
  - **BACKWARD:** New schema can read data written with the old schema. Achieved by only adding optional fields with defaults.
  - **FORWARD:** Old schema can read data written with the new schema. Achieved by only removing optional fields.
  - **FULL:** Both backward and forward compatible — the safest option for production systems.
- **Enforcement:** Producers must validate their serialized payload against the registered schema before publishing. If the payload violates the compatibility rules, the write is rejected at the producer level, preventing corrupt data from entering the topic.


## Kafka vs. Event-Driven Alternatives

### When NOT to Use Kafka
Kafka excels at high-throughput, ordered event streaming. However, it is not always the right choice:

- **Simple Task Queues:** If you need to distribute work items across workers without ordering guarantees (e.g., image resizing, email sending), a simpler queue like **RabbitMQ** or **AWS SQS** reduces operational complexity.
- **Real-Time WebSocket Push:** Kafka is pull-based. For real-time push notifications to browsers or mobile clients, use **Redis Pub/Sub** or a dedicated WebSocket gateway.
- **Sub-Millisecond Latency:** Kafka's batching and replication introduce millisecond-range latency. For ultra-low-latency inter-process communication (e.g., inside a matching engine), use shared memory or in-process queues.

## Message Broker Comparison Matrix

Selecting the right broker technology depends on the architectural requirements:

| Dimension | Apache Kafka | RabbitMQ | AWS SQS / SNS |
|---|---|---|---|
| **Architecture** | Partitioned commit log (pull-based) | Smart broker, dumb consumer (push-based) | Cloud-managed queue (pull/push-based) |
| **Throughput Scale** | **Extreme** (10M+ messages/sec via sequential disc I/O) | High (Capped by broker memory and queue routing complexity) | High (Managed automatically by cloud scaling limits) |
| **Routing Capability** | Basic (consumer groups read whole topic partitions) | **Complex** (supports exchange routing keys, fanout, headers) | Basic (SNS topic fanout to SQS queues) |
| **Ordering Guarantees** | Strict order *per partition* via keys | Order guaranteed only for single consumers | Strict order only when utilizing FIFO queues (low throughput) |
| **Backpressure** | Managed by consumer (consumer pulls when ready) | Smart broker manages queue size (pushes back on producer) | Managed by consumer pooling configurations |
| **Retention** | Durable (retains messages on disk for days/weeks) | Transient (messages are deleted immediately after consumption) | Transient (messages deleted after poll commit, max 14 days) |
| **Schema Evolution** | Schema Registry (Avro/Protobuf) | No native schema support | No native schema support |


> ⭐ **STAR Moment: The Ordering Guarantee**
> 
> In a system design interview, explain: *"We will configure our payment topics with a partitioning key based on the ledger account ID. This guarantees that all transactions affecting a specific account are processed sequentially by a single thread in our consumer group, eliminating race conditions and balance corruption during high-frequency parallel events. We use the StickyAssignor with cooperative rebalancing to minimize processing pauses when consumers scale, and route poison messages to a Dead Letter Queue after three retry attempts to prevent partition blocking."* This shows deep understanding of partition routing, failure recovery, and operational maturity.


# AI/ML System Design and LLM Integration

> *"Integrating intelligence into production applications requires more than wrapping an API call. It requires designing pipelines that scale, secure prompts, and enforce data boundaries."*


## AI/ML in Technical Interviews

In modern technical interviews (especially at FAANG and high-growth startups), system design questions have evolved. In addition to traditional payment or social network systems, you are highly likely to be asked: *"Design a real-time recommendation system," "Design a semantic search pipeline,"* or *"Design a secure, rate-limited gateway for LLM agents."*

Junior candidates treat AI as magic, describing prompt calls without considering scale, caching, latency, or security. A senior system designer must articulate how to generate embeddings, perform vector similarity search, secure LLM prompts from injection, and manage data confidentiality.

In this chapter, we outline a structured approach to AI/ML system design, focusing on the ML system design framework, vector databases, RAG architecture pipelines, agentic tool-use patterns, and prompt gateway security.

![Retrieval-Augmented Generation (RAG) Architecture Pipeline](editions/python/chapters/22-aiml-llm/visuals/rag_architecture.png){width=90%}


## The AI/ML System Design Framework

When asked to design a machine learning system (e.g., real-time recommendation), partition your design into three distinct pipelines:

### The Offline Data Ingestion & Training Pipeline

- **Raw Data Ingestion:** Extract user activity logs, purchase history, or item metadata from primary databases.
- **Feature Store:** Store processed features (user demographics, interaction history) in a high-speed Feature Store (e.g., Feast) to ensure training and serving pipelines use consistent feature definitions.
- **Model Training:** Train recommendation models offline (e.g., Collaborative Filtering, Deep Learning models) and export model checkpoints to a Model Registry.
- **Training Infrastructure:** Use distributed training frameworks (PyTorch DDP, TensorFlow Distribution Strategy) to train across multiple GPUs. Export models in a portable format (ONNX, TorchScript, SavedModel).

### The Online Prediction Pipeline

- **Low Latency:** Online predictions must run in the sub-100ms range.
- **Feature Retrieval:** When a user requests recommendations, retrieve their online features from the Feature Store cache.
- **Candidate Retrieval (Recall):** Query a vector database to retrieve the top 100 candidate items (using fast approximate nearest neighbors).
- **Ranking:** Run a lighter model online to rank these 100 candidate items, returning the top 10 to the user.
- **Model Serving:** Deploy models behind low-latency serving infrastructure (TensorFlow Serving, Triton Inference Server, or custom gRPC endpoints).

![Model Serving Infrastructure and Real-time Inference](editions/python/chapters/22-aiml-llm/visuals/model_serving.jpg){width=85%}

### The Evaluation & Monitoring Pipeline
Machine learning models degrade over time as the real-world distribution shifts away from the training data:

- **Offline Metrics:** Evaluate models on held-out test data using precision, recall, F1-score, and AUC-ROC before promoting to production.
- **Online Metrics:** Track live A/B test metrics — click-through rate (CTR), conversion rate, revenue per session — to validate that the model improves business outcomes, not just accuracy scores.
- **Data Drift Detection:** Monitor input feature distributions in production. If the mean, variance, or categorical distribution of a feature shifts significantly from the training baseline, trigger a model retraining alert.
- **Shadow Mode Deployment:** Before replacing the incumbent model, deploy the new model in **shadow mode** — it receives real traffic but its predictions are logged and compared against the live model without being served to users. Only promote when shadow metrics are statistically superior.


## Model Evaluation Metrics Deep-Dive

In ML system design interviews, you must explain the right evaluation metric for the use case:

| Metric | Formula | Best For | Pitfall |
|:-----------------|:--------------------------------------------|:----------------------------------------|:----------------------------------------|
| **Precision** | `TP / (TP + FP)` | Fraud detection (minimize false alarms) | Misses real fraud if too conservative |
| **Recall** | `TP / (TP + FN)` | Medical diagnosis (catch all positives) | Too many false positives annoy users |
| **F1-Score** | `2 * (Precision * Recall) / (Precision + Recall)` | Balanced classification tasks | Hides class imbalance issues |
| **AUC-ROC** | Area under the ROC curve | Ranking quality across thresholds | Misleading on heavily imbalanced datasets |
| **NDCG** | Normalized Discounted Cumulative Gain | Recommendation/search ranking | Sensitive to the number of results evaluated |

*Note: In the formulas above, **TP** represents True Positives (actual positive items correctly classified), **FP** represents False Positives (actual negative items incorrectly classified as positive), and **FN** represents False Negatives (actual positive items incorrectly classified as negative).*

> **Interview Signal:** If asked *"How do you evaluate a fraud detection model?"*, respond: *"We optimize for recall first — missing a real fraud case is far more costly than a false alarm. We track precision-recall curves rather than accuracy, since our dataset is heavily imbalanced (99.9% non-fraud). We set our classification threshold to achieve 95% recall, accepting a lower precision, and route flagged transactions to a human review queue."*


## Vector Databases & Semantic Search

For applications utilizing natural language (such as customer support search or legal document retrieval), standard keyword-based database queries (`LIKE %query%`) are insufficient. They cannot capture semantic meaning.

### Embeddings and Vector Search

- **Embeddings:** An embedding model (e.g., OpenAI text-embedding, BERT, Sentence-BERT) transforms text into a high-dimensional vector (e.g., 1536 floating-point values) representing the semantic meaning of the words.
- **Vector Database:** Specialized databases (Pinecone, Milvus, Qdrant, Weaviate, or Postgres with pgvector extension) store these vectors.
- **Index Optimization:** To query millions of vectors under millisecond constraints, vector databases utilize approximate nearest neighbors (ANN) index algorithms:
  - **HNSW (Hierarchical Navigable Small World):** A multi-layer graph index that provides extremely fast query speeds but requires high memory footprints to store the graph. Best for datasets under 50 million vectors where RAM budget permits.
  - **IVF-Flat (Inverted File Index):** Groups vectors into clusters using k-means, limiting search scope to the nearest clusters. Uses less memory than HNSW but has slightly lower search recall. Best for cost-sensitive deployments with large datasets.
  - **PQ (Product Quantization):** Compresses vectors by splitting them into sub-vectors and quantizing each independently. Dramatically reduces memory usage at the cost of some accuracy. Best for billion-scale datasets.

### Chunking Strategies for RAG
The quality of vector search results depends heavily on how source documents are split into chunks before embedding:

- **Fixed-Size Chunking:** Split documents into chunks of 512 or 1024 tokens. Simple but can break sentences mid-thought.
- **Semantic Chunking:** Split on paragraph or section boundaries, preserving logical coherence. Higher retrieval quality but variable chunk sizes.
- **Overlapping Windows:** Use a sliding window with 20% overlap between chunks. Ensures that concepts spanning chunk boundaries are captured by at least one chunk.
- **Metadata Enrichment:** Attach document title, section heading, page number, and source URL to each chunk as metadata. This enables filtered searches (e.g., "search only in the compliance policy documents").


## LLM Integration Patterns

When incorporating Large Language Models (LLMs) into production-grade systems, architects must resolve latency bottlenecks, costs, and security risks.

### Retrieval-Augmented Generation (RAG)
RAG addresses LLM knowledge limits and hallucinations by injecting relevant business data into the model prompt:

1. The user submits a query.
2. The query is converted to a vector embedding.
3. The vector database performs a similarity search, returning matching business documents.
4. The document text is inserted into the LLM system prompt as context.
5. The LLM processes the context to generate an accurate, grounded response.

### RAG Quality Optimization

- **Re-Ranking:** After retrieving the top-K documents from the vector database, pass them through a **cross-encoder re-ranker** (e.g., Cohere Rerank, a fine-tuned BERT cross-encoder) that scores each document against the original query. This dramatically improves context relevance over raw vector similarity alone.
- **Hybrid Search:** Combine vector similarity search with traditional keyword search (BM25). This catches exact-match terms that semantic search may miss (e.g., product codes, invoice numbers, legal clause identifiers).
- **Context Window Management:** LLMs have finite context windows (4K–128K tokens). If your retrieved documents exceed the window, implement a context budget — rank documents by relevance score and truncate at the token limit rather than naively concatenating all results.

### Semantic Caching
LLM API calls are slow and expensive. To optimize latency:

- Implement a **Semantic Cache** (e.g., GPTCache using Redis).
- Instead of exact match string caching, convert incoming prompts to vectors and check similarity against cached prompts.
- If a query has a 95%+ vector similarity match to a cached entry, return the cached LLM response directly, avoiding downstream API latency.
- **Cache Invalidation:** Set TTLs on cached entries aligned with the freshness requirements of the underlying data. For static knowledge bases, TTLs of 24–72 hours are appropriate. For real-time data, bypass the cache entirely.

### Fine-Tuning vs. Prompt Engineering
When adapting LLMs to domain-specific tasks, choose the right approach:

| Approach | When to Use | Cost | Latency Impact |
|---|---|---|---|
| **Prompt Engineering** | General tasks, rapid iteration, small domain context | Low (no training cost) | Adds tokens to every request |
| **Few-Shot Prompting** | Tasks with clear input/output patterns | Low | Moderate token overhead |
| **RAG** | Large knowledge bases, frequently updated data | Medium (embedding + vector DB) | Adds retrieval latency (~50ms) |
| **Fine-Tuning** | Consistent style/format, specialized domain language | High (GPU training cost) | Reduces prompt size, faster inference |
| **Full Pretraining** | Entirely new domains with no base model coverage | Very High | N/A (creates new model) |

> **Rule of Thumb:** Start with prompt engineering. Move to RAG if the model needs access to private or frequently updated data. Fine-tune only when prompt engineering consistently fails to produce the required output format or domain accuracy.

### Multimodal AI: Beyond Text

Modern AI systems increasingly process multiple modalities — text, images, audio, and video — within unified architectures. Interview questions are beginning to reflect this shift.

**Architectural Patterns for Multimodal Systems:**

**1. Vision-Language Models (VLMs):** Systems like GPT-4o and Gemini accept both images and text as input. The architectural pattern involves a visual encoder (often a Vision Transformer) that produces embedding tokens, which are concatenated with text tokens before being processed by the language model. For AuraPay, this enables check deposit processing: the VLM reads the check image, extracts the amount and payee, and populates the transaction record — replacing a fragile OCR pipeline.

**2. Audio Processing Pipelines:** Real-time transcription (Whisper, Deepgram) feeds into LLM reasoning. The key architectural decision is streaming vs. batch: streaming transcription adds 200-500ms latency but enables real-time agent responses, while batch processing is simpler and more accurate. ZenithTrade uses streaming transcription for compliance monitoring of trader phone calls.

**3. Multimodal RAG:** Instead of retrieving only text chunks, multimodal RAG indexes images, diagrams, and tables alongside text. Document understanding models (like LayoutLM) preserve spatial relationships in scanned documents. This is critical for ChiramTrust's regulatory document processing, where table structures contain compliance data that pure text extraction would lose.

**Interview Tip:** When asked about an AI/ML system, always clarify which modalities the system needs to handle. A document processing pipeline that handles scanned PDFs requires fundamentally different architecture than one processing structured text.

### Agentic Tool-Use Patterns
LLMs can be orchestrated as **agents** that decide which tools to call based on user intent:

- **Function Calling:** The LLM receives a list of available tool definitions (name, description, parameters). Based on the user query, it selects the appropriate tool and generates structured JSON arguments.
- **Multi-Step Orchestration:** Complex tasks require chaining multiple tool calls. An agent might: (1) query a database for account details, (2) call a fraud scoring API, (3) generate a human-readable summary. Frameworks like LangChain, LlamaIndex, or custom orchestrators manage this loop.
- **Guardrails:** Constrain the agent's tool access. A customer-facing agent should never have access to database deletion tools. Implement a **tool whitelist** per agent role, and validate all generated tool arguments against input schemas before execution.


## Prompt Gateway Security

LLMs are vulnerable to **Prompt Injection Attacks** (where an attacker crafts inputs to bypass system rules or extract private instruction prompts).
To defend your platform, you must place a **Security Filter** in front of your LLM call:

### Defense Layers

1. **Input Sanitization:** Parse incoming prompts to detect known injection patterns — phrases like "ignore previous instructions," "system prompt:", or attempts to encode instructions in base64 or unicode.
2. **PII Scrubbing:** Before sending user data to an external LLM API, scrub all Personally Identifiable Information — names, credit card numbers, Social Security numbers, phone numbers — using regex patterns and Named Entity Recognition (NER) models.
3. **Output Validation:** After receiving the LLM response, validate it against expected output schemas. If the model returns data that violates format constraints (e.g., includes SQL queries, URLs to external sites, or content that bypasses content policies), block the response.
4. **Rate Limiting:** Apply per-user and per-session rate limits on LLM API calls to prevent abuse and cost runaway. Use the Redis sliding window pattern (discussed in the Enterprise Integration and Resiliency chapter).

The following code illustrates a prompt verification filter:

```python
import re

class LlmGatewaySecurityFilter:
    _INJECTION_PATTERN = re.compile(
        r"(ignore all previous instructions|system prompt|bypass validation|reveal key)",
        re.IGNORECASE
    )

    def validate_prompt(self, user_prompt: str) -> bool:
        if not user_prompt or not user_prompt.strip():
            return False
        # Fail-fast if malicious injection signature detected
        if self._INJECTION_PATTERN.search(user_prompt):
            raise PermissionError("Potential prompt injection attack blocked")
        return True
```


Any incoming prompt containing injection signatures is blocked immediately before execution, protecting the LLM boundary from security drift.


## Case Study Integration: ML in Practice

**AuraPay: Real-Time Fraud Detection Pipeline**
AuraPay processes 50,000 transactions per second. Its fraud detection pipeline combines rule-based filters (velocity checks, geo-anomaly flags) with a gradient-boosted ensemble model trained on 18 months of labeled transaction data. Feature engineering extracts 47 signals per transaction: merchant category deviation, time-of-day risk scores, device fingerprint similarity, and spending velocity z-scores. The model runs inference in < 5ms per transaction via ONNX Runtime, with a fallback to rule-only evaluation if the ML service is unavailable (graceful degradation, per Chapter 17's resiliency patterns).

**ZenithTrade: LLM-Powered Compliance Checker**
ZenithTrade's regulatory compliance team reviews 200+ SEC filings weekly. Their LLM pipeline uses Retrieval-Augmented Generation (RAG) to cross-reference new filings against the firm's internal compliance rulebook (12,000 rules). The system generates structured compliance reports highlighting potential violations, with confidence scores and source citations. Human compliance officers review flagged items — the LLM augments but never replaces human judgment on regulatory decisions.

### Mock Interview Transcript: RAG Pipeline Design

> **Interviewer:** Design a RAG pipeline for a customer support chatbot that handles 10,000 queries/hour. Walk me through the architecture.
> **Candidate:** First, we need to vectorize our support documentation. We'd use semantic chunking to keep logical sections together, pass them through an embedding model like text-embedding-3-small, and store the vectors in a specialized vector database like Pinecone or Weaviate. When a user queries, we embed the query, perform an approximate nearest neighbor search to retrieve the top 5 chunks, and inject them into the LLM prompt.
> **Interviewer:** What's your latency budget for this, and how do you meet it at 10,000 queries/hour?
> **Candidate:** 10,000 an hour is roughly 3 queries a second. Our biggest bottleneck is the LLM inference time. To reduce latency and API costs, I'd implement semantic caching using Redis. We convert the incoming query to a vector and check if we have a 95%+ similarity match with a previous query. If so, we return the cached response immediately.
> **Interviewer:** How do you handle hallucinations? If the bot gives wrong refund instructions, it's a huge liability.
> **Candidate:** Actually, let me reconsider the prompt structure... We must constrain the model. In the system prompt, we explicitly instruct it to answer *only* using the provided context. If the answer isn't in the chunks, it must reply "I don't know" and escalate to a human. We'd also run a cross-encoder re-ranker after retrieval to ensure only highly relevant context is passed to the LLM.
> **Interviewer:** How do you prevent users from jailbreaking the bot to ignore those instructions?
> **Candidate:** We'd place a security filter gateway in front of the LLM to scan for prompt injection signatures, and use input sanitization to strip out command-like phrasing before embedding.

**Technical Summary:** The candidate successfully designed a robust RAG pipeline, incorporating semantic chunking and vector search. They addressed scale and latency via semantic caching, mitigated hallucinations through strict prompt constraints and re-ranking, and prioritized security with a prompt injection gateway.

## Cost Optimization for LLM-Powered Systems

LLM inference costs scale directly with token volume. At enterprise scale, unoptimized architectures can generate six-figure monthly bills:

1. **Model Tiering:** Route simple queries (FAQ lookups, classification) to smaller, cheaper models (GPT-4o-mini, Claude Haiku). Reserve expensive frontier models (GPT-4o, Claude Opus) for complex reasoning tasks. Implement a **router model** that classifies query complexity before dispatching.
2. **Prompt Compression:** Use techniques like LLMLingua to compress long context windows by removing redundant tokens while preserving semantic meaning. Can reduce token counts by 50–70%.
3. **Batch Processing:** For non-real-time workloads (document summarization, report generation), batch requests and use discounted batch API pricing.
4. **Self-Hosted Models:** For high-volume, latency-tolerant workloads, deploy open-source models (Llama, Mistral) on owned GPU infrastructure. Higher upfront cost but dramatically lower per-token cost at scale.


> ⭐ **STAR Moment: The Full ML System Design**
> 
> In a system design interview, demonstrate the complete picture: *"For the recommendation engine, we separate our architecture into three pipelines. The offline pipeline trains our ranking model using user interaction features stored in Feast, with weekly retraining triggered by data drift detection. The online pipeline retrieves candidate items via HNSW vector search, then re-ranks with a lightweight cross-encoder model, targeting sub-100ms p99 latency. We deploy new models in shadow mode first, comparing CTR and conversion rates against the incumbent via A/B testing before promotion. For cost control, we route simple classification queries to GPT-4o-mini and reserve frontier models for complex reasoning."* This shows end-to-end ML engineering maturity.


# Appendix: Quick Reference Cards and Cheat Sheets

> *"In the heat of a technical evaluation, clarity is your greatest asset. Maintain a structured checklist to eliminate cognitive overhead and stay focused on design correctness."*


## Big-O Complexity Quick Reference

The following table summarizes the time and space complexity of common data structures and algorithmic operations. Senior candidates should have these values committed to memory to quickly justify trade-offs.

### Data Structure Complexities

| Data Structure | Average Access | Average Search | Average Insertion | Average Deletion | Space Complexity |
|---|---|---|---|---|---|
| **Array** | $O(1)$ | $O(N)$ | $O(N)$ | $O(N)$ | $O(N)$ |
| **Singly-Linked List** | $O(N)$ | $O(N)$ | $O(1)$ | $O(1)$ | $O(N)$ |
| **Doubly-Linked List** | $O(N)$ | $O(N)$ | $O(1)$ | $O(1)$ | $O(N)$ |
| **Stack / Queue** | $O(N)$ | $O(N)$ | $O(1)$ | $O(1)$ | $O(N)$ |
| **Hash Table (HashMap)** | $O(1)$ | $O(1)$ | $O(1)$ | $O(1)$ | $O(N)$ |
| **Binary Search Tree** | $O(\log N)$ | $O(\log N)$ | $O(\log N)$ | $O(\log N)$ | $O(N)$ |
| **Red-Black Tree (TreeMap)** | $O(\log N)$ | $O(\log N)$ | $O(\log N)$ | $O(\log N)$ | $O(N)$ |
| **Binary Heap (PriorityQueue)**| $O(N)$ | $O(N)$ | $O(\log N)$ | $O(\log N)$ | $O(N)$ |

### Algorithmic Complexities

| Algorithm Class | Time Complexity (Best) | Time Complexity (Avg) | Time Complexity (Worst) | Space Complexity (Worst) |
|---|---|---|---|---|
| **Quicksort** | $O(N \log N)$ | $O(N \log N)$ | $O(N^2)$ | $O(\log N)$ |
| **Mergesort** | $O(N \log N)$ | $O(N \log N)$ | $O(N \log N)$ | $O(N)$ |
| **Heapsort** | $O(N \log N)$ | $O(N \log N)$ | $O(N \log N)$ | $O(1)$ |
| **Binary Search** | $O(1)$ | $O(\log N)$ | $O(\log N)$ | $O(1)$ |
| **Graph BFS / DFS** | $O(V + E)$ | $O(V + E)$ | $O(V + E)$ | $O(V)$ |
| **Dijkstra's Algorithm** | $O(E \log V)$ | $O(E \log V)$ | $O(E \log V)$ | $O(V)$ |
| **Bellman-Ford Algorithm** | $O(V E)$ | $O(V E)$ | $O(V E)$ | $O(V)$ |

*Note: In graph algorithmic complexities, **V** represents the number of Vertices (nodes) in the graph, and **E** represents the number of Edges (connections).*


## The Edge-Case Checklist

When writing code in a timed assessment or live coding session, run through this edge-case checklist before declaring your solution complete:

### Numeric Inputs (Integers / Floats)

- **Zero & Negatives:** Does the algorithm handle `0` or negative values correctly? (e.g., in binary search, partition loops, or currency scale calculations).
- **Overflow Limits:** Are you vulnerable to integer overflow? (In Java, if adding two numbers can exceed `Integer.MAX_VALUE` [$2^{31}-1$], utilize `long` arithmetic or `Math.addExact()`).
- **Dividing by Zero:** Ensure no division operations can occur with a zero denominator.

### Collection Inputs (Arrays, Lists, Maps)

- **Null & Empty:** Always write a fail-fast check: `if (nums == null || nums.length == 0) return ...;`
- **Single Element:** Does your binary search, partition, or sliding window terminate correctly if the collection contains exactly one element?
- **Duplicates:** Does the algorithm behave correctly if the collection is filled with duplicate values? (e.g., finding the target in a rotated sorted array containing duplicates).
- **Extremes:** What happens if the input has $10^6$ elements? Does your space complexity remain within standard heap bounds?

### String Inputs

- **Null & Empty:** Handled correctly?
- **Whitespace:** Does the string contain leading, trailing, or multiple consecutive spaces?
- **Case Sensitivity:** Does your hash map or sorting logic treat `'a'` and `'A'` correctly based on the problem specification?
- **Character Set:** Are you assuming ASCII characters (128 values) while the inputs could be UTF-8/Unicode?

### Linked Lists

- **Cycle Detection:** Does the list contain a cycle? (Will your loop run infinitely?)
- **Empty / Head-Tail Manipulations:** Does the code crash on pointer references (e.g., `node.next.next`) when handling lists of length 1 or 2?

### Defensive Coding Checklist for Assessments

Before submitting any solution in a timed assessment, verify these guards:

**Input Validation**
- [ ] Null/None check on input arrays, strings, and objects
- [ ] Empty collection check (length == 0)
- [ ] Single-element edge case
- [ ] Negative number handling (if applicable)
- [ ] Integer overflow risk (use long for running sums)

**Boundary Conditions**
- [ ] First element and last element processed correctly
- [ ] Off-by-one errors in loop bounds (< vs <=)
- [ ] Window/pointer doesn't exceed array bounds
- [ ] Division by zero guarded
- [ ] Modulo with negative numbers: use ((x % k) + k) % k

**Data Structure Edge Cases**
- [ ] HashMap: handle missing keys (getOrDefault)
- [ ] Stack/Deque: check isEmpty() before peek/pop
- [ ] Priority Queue: verify comparator handles equal elements
- [ ] Graph: handle disconnected components
- [ ] Tree: handle null left/right children

**Output Verification**
- [ ] Return type matches specification exactly
- [ ] Empty result case handled (return empty list, not null)
- [ ] Results sorted if specification requires ordering
- [ ] No duplicate entries if specification requires unique values

> **Time Budget:** Spend the final 2 minutes of any timed problem running through this checklist mentally. It catches 80% of edge-case failures.


## Distributed Systems Cheat Sheet

In system design interviews, refer to these rules of thumb to justify your infrastructure capacity planning:

### System Availability (The "Nines")

| Availability % | Downtime per Year | Downtime per Day | Class / Tier |
|---|---|---|---|
| **99% (Two Nines)** | 3.65 days | 14.4 minutes | Basic website |
| **99.9% (Three Nines)** | 8.76 hours | 1.44 minutes | Standard Cloud Microservice |
| **99.99% (Four Nines)** | 52.6 minutes | 8.6 seconds | Banking-grade service (AuraPay) |
| **99.999% (Five Nines)** | 5.26 minutes | 0.86 seconds | Telecommunications / HFT Exchange |

### Latency Numbers Every Programmer Should Know

To make back-of-the-envelope calculations, memorize these rough access latency scales:

| Operation | Time | Time (Human Scale) |
|---|---|---|
| **L1 Cache reference** | 1 ns | 1 sec |
| **Branch mispredict** | 5 ns | 5 sec |
| **L2 Cache reference** | 4 ns | 4 sec |
| **Main Memory reference (DDR5)** | 50 ns | 50 sec |
| **Compress 1K bytes with Zippy** | 3,000 ns | 50 min |
| **Send 2K bytes over 1 Gbps network** | 20,000 ns | 5.5 hours |
| **NVMe SSD random read** | 10-20 μs | ~3-6 hours |
| **NVMe SSD sequential 1MB read** | 100-200 μs | ~1-2 days |
| **Round trip within same datacenter** | 250-500 μs | ~3-6 days |
| **HDD seek** | 2-5 ms | ~1-2 months |
| **Read 1MB sequentially from Disk** | 20,000,000 ns | 7.5 months |
| **Send packet CA to Netherlands to CA** | 150,000,000 ns | 4.7 years |

These numbers reflect 2024 NVMe Gen4/5 SSDs and DDR5 RAM. Original latency numbers by Jeff Dean (2012) have been updated. Cloud VM performance may vary based on instance type and IO throttling.


## Day of the Interview Checklist

Before entering a live call (Teams/Zoom) or in-person evaluation, ensure you have completed these checks:

- [ ] **Whiteboard Readiness:** If using a digital whiteboard, log in and verify that your shortcut keys and drawing shapes function correctly.
- [ ] **Code Editor Settings:** Turn off all autocomplete/AI copilot extensions inside your IDE or browser coding window. Interviewers expect you to write clean syntax without AI assistance.
- [ ] **Audio/Video Setup:** Clean background, clear microphone, and a stable internet connection.
- [ ] **The "Trade-off" Mindset:** Remember, there are no "perfect" architectures in system design. For every choice you make, write down the corresponding scale, cost, or complexity trade-off on your whiteboard.
- [ ] **Boundary Verification:** Write your pre-conditions, post-conditions, and invariants *first* before implementing any code. Protect the boundary.


# References {.unnumbered}

Abadi, D. J. (2012). Consistency tradeoffs in modern distributed database system design: CAP is only part of the story. *Computer*, 45(2), 37-42. https://doi.org/10.1109/mc.2012.33

Bloch, J. (2018). *Effective Java* (3rd ed.). Addison-Wesley.

Brooks, F. P. (1975). *The Mythical Man-Month*. Addison-Wesley.

Burns, B., Beda, J., Hightower, K., & Evenson, L. (2022). *Kubernetes: Up and Running* (3rd ed.). O'Reilly Media.

Codd, E. F. (1970). A relational model of data for large shared data banks. *Communications of the ACM*, 13(6), 377-387. https://doi.org/10.1145/362384.362685

Corbett, J. C., Dean, J., Epstein, M., Fikes, A., Frost, C., Furman, J. J., Ghemawat, S., Gubarev, A., Heiser, C., Hochschild, P., Hsieh, W., Kanthak, S., Kogan, E., Li, H., Lloyd, A., Melnik, S., Mwaura, D., Nagle, D., Seanquin, S., ... Woody, S. (2013). Spanner: Google's globally distributed database. *ACM Transactions on Computer Systems (TOCS)*, 31(3), 1-22. https://doi.org/10.1145/2491245

Dean, J. (2012). *Latency numbers every programmer should know* [Presentation]. Stanford University. https://brenocon.com/dean_perf.html

Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified data processing on large clusters. *Communications of the ACM*, 51(1), 107-113. https://doi.org/10.1145/1327452.1327492

DeCandia, G., Hastorun, D., Jampani, M., Kakulapati, G., Lakshman, A., Pilchin, A., Sivasubramanian, S., Vosshall, W., & Vogels, W. (2007). Dynamo: Amazon's highly available key-value store. *ACM SIGOPS Operating Systems Review*, 41(6), 205-220. https://doi.org/10.1145/1294261.1294281

Dijkstra, E. W. (1968). Letters to the editor: Go to statement considered harmful. *Communications of the ACM*, 11(3), 147-148. https://doi.org/10.1145/362929.362947

Elhemaly, M., Gallagher, N., Tang, B., Gordon, N., Huang, H., Chen, H., Idziorek, J., Katz, M., Kosaian, J., Muthukkaruppan, K., Ramesh, S., Sowell, B., Veeramachaneni, S., Xiang, W., & Zhong, X. (2022). Amazon DynamoDB: A scalable, predictably performant, and fully managed NoSQL database service. *Proceedings of USENIX ATC '22*, 1037-1048.

Evans, E. (2003). *Domain-driven design: Tackling complexity in the heart of software*. Addison-Wesley.

Forsgren, N., Humble, J., & Kim, G. (2018). *Accelerate: The science of lean software and DevOps*. IT Revolution.

Fowler, M. (2002). *Patterns of enterprise application architecture*. Addison-Wesley.

Fowler, M. (2022). *Python concurrency with asyncio*. Manning Publications.

Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design patterns: Elements of reusable object-oriented software*. Addison-Wesley.

Garcia-Molina, H., & Salem, K. (1987). Sagas. *Proceedings of the ACM SIGMOD International Conference on Management of Data*, 249-259. https://doi.org/10.1145/38713.38742

Gilbert, S., & Lynch, N. (2002). Brewer's conjecture and the feasibility of consistent, available, partition-tolerant web services. *ACM SIGACT News*, 33(2), 51-59. https://doi.org/10.1145/564585.564601

Goetz, B., Peierls, T., Bloch, J., Bowbeer, J., Holmes, D., & Lea, D. (2006). *Java concurrency in practice*. Addison-Wesley.

Hoare, C. A. R. (1969). An axiomatic basis for computer programming. *Communications of the ACM*, 12(10), 576-580. https://doi.org/10.1145/363235.363259

Hohpe, G., & Woolf, B. (2003). *Enterprise integration patterns: Designing, building, and deploying messaging solutions*. Addison-Wesley.

Hunt, P., Konar, M., Junqueira, F. P., & Reed, B. (2010). ZooKeeper: Wait-free coordination for internet-scale systems. *USENIX Annual Technical Conference*, 2(9), 12-25. https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf

Kleppmann, M. (2017). *Designing data-intensive applications: The big ideas behind reliable, scalable, and maintainable systems*. O'Reilly Media.

Knuth, D. E. (1997). *The art of computer programming* (Vols. 1-3). Addison-Wesley.

Kreps, J., Narkhede, N., & Rao, J. (2011). Kafka: A distributed messaging system for log processing. *Proceedings of the NetDB*, 1-7. https://jkreps.files.wordpress.com/2011/09/kafka_netdb11.pdf

Lakshman, A., & Malik, P. (2010). Cassandra: A decentralized structured storage system. *ACM SIGOPS Operating Systems Review*, 44(2), 35-40. https://doi.org/10.1145/1773912.1773952

Lamport, L. (1978). Time, clocks, and the ordering of events in a distributed system. *Communications of the ACM*, 21(7), 558-565. https://doi.org/10.1145/359545.359563

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474. https://arxiv.org/abs/2005.11401

Liskov, B. H., & Wing, J. M. (1994). A behavioral notion of subtyping. *ACM Transactions on Programming Languages and Systems (TOPLAS)*, 16(6), 1811-1841. https://doi.org/10.1145/197320.197383

Martin, R. C. (2018). *Clean architecture: A craftsman's guide to software structure and design*. Prentice Hall.

National Institute of Standards and Technology. (2002). *The economic impacts of inadequate infrastructure for software testing* (Planning Report 02-3). U.S. Department of Commerce.

Newman, S. (2021). *Building microservices* (2nd ed.). O'Reilly Media.

Nygard, M. T. (2018). *Release it! Design and deploy production-ready software* (2nd ed.). Pragmatic Bookshelf.

Ongaro, D., & Ousterhout, J. (2014). In search of an understandable consensus algorithm. *USENIX Annual Technical Conference*, 305-319. https://www.usenix.org/system/files/conference/atc14/atc14-paper-ongaro.pdf

Ousterhout, J. (2021). *A philosophy of software design* (2nd ed.). Yaknyam Press.

PCI Security Standards Council. (2024). *Payment Card Industry Data Security Standard (PCI-DSS) v4.0.1*. PCI SSC.

Shvachko, K., Kuang, H., Radia, S., & Chansler, R. (2010). The Hadoop distributed file system. *IEEE MSST*, 1-10. https://doi.org/10.1109/msst.2010.5496972

Skeet, J. (2019). *C# in Depth* (4th ed.). Manning Publications.

Tanenbaum, A. S., & Van Steen, M. (2023). *Distributed systems* (4th ed.). Maarten van Steen.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008. https://arxiv.org/abs/1706.03762

W3C. (2022). *Decentralized Identifiers (DIDs) v1.0*. World Wide Web Consortium. https://www.w3.org/TR/did-core/
