# Visual Assets — Image Generation Prompts

> **Purpose:** Master catalog of all image prompts for *Spec-Driven Coding Interviews*.
> Each entry tracks the chapter target, filename, and exact prompt used for Gemini image generation.
> **Design Rule:** All figures use **white/light backgrounds** for print readability.

---

## 1. Book Cover
- **File:** `cover_light.png`
- **Target:** Root `visuals/cover.png`
- **Chapter:** Front matter
- **Prompt:**
```
Professional technical book cover design for "Spec-Driven Coding Interviews: The Advanced Reference Manual for Software Professionals" by Harinath Mallepally. Clean white background with geometric circuit board patterns and flowing data streams in deep navy (#1E3A8A) and gold (#B45309) accents. Clean, modern typography with the title in dark navy. The design should convey authority, precision, and senior engineering expertise. Minimalist, premium aesthetic like O'Reilly or Manning book covers. Include subtle code syntax elements integrated into the geometric patterns. No device frames.
```

---

## 2. AuraPay System Architecture
- **File:** `aurapay_arch_light.png`
- **Target:** `chapters/02-case-studies/visuals/aurapay_architecture.png`
- **Chapter:** Ch 2 — The Three System-Scale Case Studies
- **Prompt:**
```
Professional software architecture diagram for "AuraPay" — a banking-grade distributed payment ledger system. Clean white background with subtle light gray grid. Colored service boxes: (1) Client/Mobile app on the left, (2) API Gateway with rate limiting, (3) Three microservices in separate bounded boxes: "AuraPay Ledger Service" (navy blue #1E3A8A), "ZenithTrade Exchange" (teal #0D9488), "ChiramTrust Identity" (amber #B45309), (4) PostgreSQL database cylinder under Ledger, (5) Kafka message broker connecting services, (6) Redis cache layer, (7) Card Vault (HSM secured) as isolated zone with dashed border. Use clean arrows showing data flow. Professional infographic style, no hand-drawn look. Include labels for each component. Similar to AWS architecture diagrams.
```

---

## 3. Circuit Breaker State Machine
- **File:** `circuit_breaker_light.png`
- **Target:** `chapters/10-resiliency/visuals/circuit_breaker.png`
- **Chapter:** Ch 10 — Enterprise Integration and Resiliency
- **Prompt:**
```
Professional state machine diagram showing the Circuit Breaker pattern for microservice resiliency. Clean white background. Three large circular state nodes: "CLOSED" (green #22C55E, normal operation), "OPEN" (red #EF4444, failures detected, requests blocked), and "HALF-OPEN" (amber #F59E0B, probe requests allowed). Show labeled transition arrows between states: CLOSED to OPEN labeled "Failure threshold exceeded", OPEN to HALF-OPEN labeled "Timeout expires", HALF-OPEN to CLOSED labeled "Probe succeeds", HALF-OPEN to OPEN labeled "Probe fails". Inside each state node, include brief behavior description. Clean, modern infographic style. No hand-drawn look. Include a small legend box.
```

---

## 4. Sliding Window Algorithm Step-by-Step
- **File:** `sliding_window_light.png`
- **Target:** `chapters/08-algorithms-assessment/visuals/sliding_window.png`
- **Chapter:** Ch 8 — Core Algorithms & Assessment Tactical Guide
- **Prompt:**
```
Professional step-by-step visualization of the Sliding Window Maximum algorithm. Clean white background. Show an array of 8 integers [1, 3, -1, -3, 5, 3, 6, 7] with window size K=3. Display 4 sequential steps vertically: Step 1 shows window over indices 0-2 with max=3 highlighted in gold, Step 2 shows window shifted to indices 1-3 with max=3, Step 3 shows window at indices 2-4 with max=5, Step 4 shows window at indices 3-5 with max=5. Each step shows the array with the current window highlighted in a teal (#0D9488) box, the deque contents below in navy (#1E3A8A) boxes, and the result array building on the right in gold (#B45309). Clean arrows showing element additions and removals from deque. Modern technical diagram style.
```

---

## 5. DDD Bounded Context Map
- **File:** `ddd_contexts_light.png`
- **Target:** `chapters/09-system-architecture/visuals/ddd_contexts.png`
- **Chapter:** Ch 9 — System Architecture and Design Fundamentals
- **Prompt:**
```
Professional Domain-Driven Design (DDD) bounded context map diagram. Clean white background with light gray grid lines. Three large rounded rectangles representing bounded contexts: "Exchange Context — ZenithTrade" (teal #0D9488 border), "Ledger Context — AuraPay" (navy #1E3A8A border), "Identity Context — ChiramTrust" (amber #B45309 border). Inside each context, show key domain objects as smaller boxes: Exchange has "Order", "OrderBook", "MatchResult"; Ledger has "LedgerAccount", "TransactionRecord", "Money (VO)"; Identity has "UserCredential", "ConsentPolicy", "Guardian". Between contexts, show integration arrows labeled: "Anti-Corruption Layer", "REST API", "Domain Events via Kafka". Include a legend explaining Entity vs Value Object vs Aggregate Root. Clean, modern technical diagram.
```

---

## 6. Saga Orchestration vs Choreography
- **File:** `saga_compare_light.png`
- **Target:** `chapters/10-resiliency/visuals/saga_comparison.png`
- **Chapter:** Ch 10 — Enterprise Integration and Resiliency
- **Prompt:**
```
Professional side-by-side comparison diagram of Saga patterns in distributed systems. Clean white background. LEFT SIDE labeled "Orchestration-Based Saga": Central "Saga Orchestrator" box (gold #B45309) with numbered arrows going to three service boxes: "1. Exchange Service", "2. Ledger Service", "3. Custody Service", with dotted red arrows showing compensating transactions on failure. RIGHT SIDE labeled "Choreography-Based Saga": Same three services connected by event arrows through a "Kafka Event Bus" bar, each emitting events that trigger the next, with dotted red compensating event arrows. Include pros and cons beneath each side. Clean modern infographic style with colored service boxes on white.
```

---

## 7. Transactional Outbox Pattern Flow
- **File:** `outbox_pattern_light.png`
- **Target:** `chapters/10-resiliency/visuals/outbox_pattern.png`
- **Chapter:** Ch 10 — Enterprise Integration and Resiliency
- **Prompt:**
```
Professional data flow diagram of the Transactional Outbox Pattern for reliable messaging. Clean white background. Horizontal flow: (1) "Application Service" box (navy #1E3A8A) with arrow labeled "Single DB Transaction" to (2) a database cylinder with two tables: "Business Table" and "Outbox Table" inside a transaction boundary (dashed green border), then (3) "CDC / Poller" worker box (teal #0D9488) reading from Outbox, then (4) arrow to "Kafka / Message Broker" (amber #B45309), then (5) arrows to multiple "Consumer Services". Show a red X crossing out a direct "Application to Kafka" arrow labeled "ANTI-PATTERN: Dual Write". Clean, modern technical diagram.
```

---

## 8. Virtual Threads vs Platform Threads
- **File:** `threads_comparison.png`
- **Target:** `chapters/07-concurrency-performance/visuals/virtual_threads.png`
- **Chapter:** Ch 7 — Designing for Performance and Concurrency
- **Prompt:**
```
Clean white background infographic. Two sections separated by a horizontal line. Top: "Platform Threads" showing four large colored blocks labeled "1MB" each, connected to operating system icons. Text: "4000 max concurrent, 4GB memory". Bottom: "Virtual Threads Java 21" showing many tiny teal squares sitting on four blue carrier blocks. Text: "1 million concurrent, 200MB memory". Arrows show virtual threads mounting and unmounting from carriers during IO operations. Professional flat design with navy and teal colors.
```

---

## 9. B-Tree vs LSM-Tree Comparison
- **File:** `btree_lsm_light.png`
- **Target:** `chapters/11-database-compliance/visuals/btree_vs_lsm.png`
- **Chapter:** Ch 11 — Database Design, Compliance, and Security
- **Prompt:**
```
Professional side-by-side comparison of B-Tree vs LSM-Tree database storage engines. Clean white background. LEFT labeled "B-Tree (PostgreSQL, MySQL)": balanced tree with root branching to internal nodes and leaf nodes with sorted keys. Labels: "Reads: O(log N) Fast", "Writes: In-place updates". RIGHT labeled "LSM-Tree (Cassandra, RocksDB)": write path showing MemTable in memory, flush arrow down to SSTable levels L0, L1, L2 on disk with compaction arrows. Labels: "Writes: O(1) Sequential appends", "Reads: Check multiple levels". Bottom comparison table with Use Case, Read Speed, Write Speed columns. B-Tree in navy #1E3A8A, LSM-Tree in teal #0D9488. Modern technical diagram.
```

---

## 10. PCI-DSS Tokenization Vault Architecture
- **File:** `tokenization_vault_light.png`
- **Target:** `chapters/11-database-compliance/visuals/tokenization_vault.png`
- **Chapter:** Ch 11 — Database Design, Compliance, and Security
- **Prompt:**
```
Professional security architecture diagram for PCI-DSS card tokenization. Clean white background. Two network zones separated by a firewall icon: LEFT ZONE labeled "Application Zone (Outside PCI Scope)" with light blue background containing "Billing Service", "Ledger Service", and "App Database" storing tokens. RIGHT ZONE labeled "PCI-DSS Secured Zone" with light red border containing "Card Vault Service", "Encrypted Vault DB", and "HSM / KMS". Flow: User enters card number, app sends PAN to Vault, Vault encrypts with AES-256-GCM, returns token to app. Lock icons and shield symbols for security. Clean infographic style.
```

---

## 11. Pattern Recognition Decision Flowchart
- **File:** `pattern_flow_light.png`
- **Target:** `chapters/08-algorithms-assessment/visuals/pattern_flowchart.png`
- **Chapter:** Ch 8 — Core Algorithms & Assessment Tactical Guide
- **Prompt:**
```
Professional decision flowchart for recognizing algorithmic patterns in coding interviews. Clean white background. Start diamond "Read the Problem" at top center. Decision branches flowing downward: "Sub-arrays or windows?" YES to teal box "Sliding Window / Monotonic Deque". "Sorted pairs or linked lists?" YES to navy box "Two Pointer / Fast-Slow". "Max/min intervals?" YES to amber box "Greedy / Interval Scheduling". "Exploring all paths?" YES to sub-diamond "Shortest path?" YES to green "BFS", NO to green "DFS / Backtracking". "Overlapping subproblems?" YES to purple "Dynamic Programming". "Next greater/smaller element?" YES to red "Monotonic Stack". Clean flowchart with rounded rectangles, diamond nodes, clear arrows. Modern infographic.
```

---

## 12. Anemic vs Rich Domain Model
- **File:** `anemic_vs_rich_light.png`
- **Target:** `chapters/03-oop-principles/visuals/anemic_vs_rich.png`
- **Chapter:** Ch 3 — Principles of Object-Oriented Design
- **Prompt:**
```
Professional side-by-side comparison of Anemic vs Rich Domain Model patterns. Clean white background. LEFT labeled "ANEMIC MODEL (Anti-Pattern)" with red accent border: "LedgerAccount" class box with only fields (id, balance, currency) and getters/setters. Arrow to large "LedgerService" containing debit(), credit(), transfer(), validate(). Red X icon. Caption: "Logic in service layer. Domain objects are dumb data bags." RIGHT labeled "RICH DOMAIN MODEL (Recommended)" with green accent border: "LedgerAccount" class box with fields AND methods debit(), credit(), transferTo(), validateCurrency(). Thin coordinator service. Green checkmark. Caption: "Logic inside domain objects. Objects protect their own invariants." UML class diagram style, clean modern colors.
```

---

## 13. Observer Pattern Class Diagram
- **File:** `observer_pattern_light.png`
- **Target:** `chapters/06-design-patterns/visuals/observer_pattern.png`
- **Chapter:** Ch 6 — Design Patterns in Enterprise Frameworks
- **Prompt:**
```
Professional UML class diagram showing the Observer Pattern in AuraPay. Clean white background. Interface "TransactionObserver" (navy #1E3A8A header) with methods onTransactionSuccess() and onTransactionFailed(). Three concrete observers with dashed implementation arrows: "AuditTrailObserver" (teal), "FraudDetectionObserver" (amber), "SmsNotificationObserver" (green). "TransactionEventPublisher" class (gold header) with methods registerObserver(), deRegisterObserver(), notifySuccess(), notifyFailure() and field List of observers. Composition arrow from Publisher to Observer interface. Clean modern UML style with shadows and rounded corners.
```

---

## 14. SOLID Dependency Inversion Graph
- **File:** `solid_dip_light.png`
- **Target:** `chapters/04-solid-boundaries/visuals/solid_dip.png`
- **Chapter:** Ch 4 — SOLID Principles: Enforcing Boundaries
- **Prompt:**
```
Professional diagram showing Dependency Inversion Principle before and after. Clean white background. TOP section "WITHOUT DIP (Tightly Coupled)" with light red background: "TransactionProcessor" box with dependency arrows pointing DOWN to concrete classes "PostgresRepository", "SmtpEmailSender", "StripeGateway". Red X. BOTTOM section "WITH DIP (Inverted)" with light green background: "TransactionProcessor" depending on interface boxes "LedgerRepository", "NotificationSender", "PaymentGateway" at same level. Concrete implementations point UP to interfaces. Green checkmark. Callout: "Both high-level and low-level depend on abstractions." Clear arrows, modern diagram with color coding.
```

---

## 15. Stream Pipeline Visualization
- **File:** `stream_pipeline_light.png`
- **Target:** `chapters/05-functional-streams/visuals/stream_pipeline.png`
- **Chapter:** Ch 5 — Modern Functional Programming and Stream APIs
- **Prompt:**
```
Professional data pipeline visualization of functional stream processing. Clean white background. Horizontal conveyor belt style pipeline. LEFT: 6 transaction boxes with amounts ($50, $200, $15, $500, $80, $1000). Stage 1 "FILTER" (teal #0D9488 header): items below $100 drop off with red arrows ($50, $15, $80 removed). Stage 2 "MAP" (navy #1E3A8A header): remaining items transformed, extracting merchantId and amount. Stage 3 "COLLECT / GROUP BY" (amber #B45309 header): groups by merchant, sums into final Map output. Items flow left to right through stages. Labels: "Lazy evaluation" and "Stateless, side-effect free". Clean modern infographic style.
```

---

## 16. Spec-Driven Path vs Syntax Trap (Prologue)
- **File:** `spec_vs_syntax_path.png`
- **Target:** `chapters/00-prologue/visuals/spec_vs_syntax.png`
- **Chapter:** Prologue
- **Prompt:**
```
Professional comparison infographic for a technical book. Clean white background. Two horizontal pathways shown one above the other. TOP PATH labeled "THE SPEC-DRIVEN PATH" with green accent: Three connected boxes flowing left to right: "1. Define Invariants" (with checkmark icon) → "2. Establish Types" (with shield icon) → "3. Implement" (with code icon). Each box has a subtitle: "Guaranteed Correct", "Boundary Enforcement", "Simple Syntax". BOTTOM PATH labeled "THE SYNTAX TRAP" with red accent: Three connected boxes: "1. Start Typing" (with keyboard icon) → "2. Hack Edge Cases" (with bandaid icon) → "3. Debug Loop" (with circular arrow icon). Subtitles: "Unbounded Chaos", "Fragile Patching", "Failed Tests". A large green checkmark on the top path and a large red X on the bottom. Clean modern infographic style.
```

---

## 17. The Invariant Wall Layers (Ch 1)
- **File:** `invariant_wall_layers.png`
- **Target:** `chapters/01-invariant-first/visuals/invariant_wall.png`
- **Chapter:** Ch 1 — The Invariant-First Strategy
- **Prompt:**
```
Professional layered diagram showing "The Invariant Wall" concept for coding interviews. Clean white background. Three horizontal layers stacked like a brick wall, viewed from the side. Bottom layer (navy #1E3A8A): "Pre-conditions — Input constraints that must be true before execution". Middle layer (teal #0D9488): "Class/Data Invariants — State rules that always hold true for domain objects". Top layer (amber #B45309): "Post-conditions — Guarantees satisfied after successful execution". On the left side, a shield icon labeled "Boundary Protection". On the right, a code bracket icon labeled "Safe Implementation Zone" inside the wall. Below the wall, wavy red lines labeled "Invalid States Blocked". Clean, modern infographic with subtle shadows on each layer.
```

---

## 18. GCA 70-Minute Time Allocation Blueprint (Ch 8)
- **File:** `gca_time_allocation.png`
- **Target:** `chapters/08-algorithms-assessment/visuals/gca_timeline.png`
- **Chapter:** Ch 8 — Algorithms & Data Structures
- **Prompt:**
```
Recursion tree for climbStairs(5) illustrating Dynamic Programming overlapping subproblems. Clean white background, textbook style. Show nodes branching down from "climbStairs(5)". Nodes contain only the function names (e.g. climbStairs(4), climbStairs(3)). No text underneath the nodes. Recalculated duplicate nodes (climbStairs(3), climbStairs(2), climbStairs(1)) are colored soft orange with a dotted border. First-time nodes are white with solid borders. At the bottom, include a neat Legend: soft orange box = Recalculated Overlapping Subproblem, white box = Unique Calculation. Clean typography, no single quotes.
```

- **Chapter:** Ch 8 — Algorithms & Data Structures
- **Prompt:**
```
Clear technical diagram showing the 1D Dynamic Programming table (array) for climbStairs(5). Clean white background, modern textbook infographic style. Show a horizontal array with 6 cells representing indices 0, 1, 2, 3, 4, 5. Inside each cell, show the value: Index 0: 1 (Base Case), Index 1: 1 (Base Case), Index 2: 2, Index 3: 3, Index 4: 5, Index 5: 8. Above the array, show indices 0, 1, 2, 3, 4, 5 labeled clearly. Draw two neat, curved connection arrows: One arrow from Index 3 (value 3) to Index 5 (value 8), and one arrow from Index 4 (value 5) to Index 5 (value 8). Above the arrows, write the formula: "dp[5] = dp[4] + dp[3] = 5 + 3 = 8". Highly readable, crisp typography, clean technical infographic style. No quotation marks or single quotes around any text.
```
- **Prompt:**
```
Professional timeline diagram showing GCA 70-Minute Time Allocation Blueprint. Clean white background, modern textbook infographic style. Show a very thin horizontal progress bar (sleek strip) representing 70 minutes, divided into 5 colored segments: Segment 1: Green (8 min, "Q1"), Segment 2: Teal (12 min, "Q2"), Segment 3: Brown/Orange (20 min, "Q3"), Segment 4: Navy Blue (25 min, "Q4"), Segment 5: Gray (5 min, "Buffer"). The bar itself should be thin and elegant (about 10% of the total image height). Below the timeline bar, show neat, clean text labels and time markers: 0m, 12m, 20m, 40m, 60m, 70m marking the transition points. Under each segment, list the question level, duration, and topic (Q1: Easy, 8m, Array/String; Q2: Medium, 12m, Matrix/Simulation; Q3: Medium-Hard, 20m, HashMap/Grouping; Q4: Hard, 25m, Monotonic Stack/Binary Search; Buffer, 5m, Review & Final Check). No hexadecimal color code text labels inside the colored segments. Highly readable, crisp typography, premium technical textbook style.
```

---

## 19. DP Table Memoization (Ch 8)
- **File:** `dp_memoization_table.png`
- **Target:** `chapters/08-algorithms-assessment/visuals/dp_table.png`
- **Chapter:** Ch 8 — Core Algorithms & Assessment Tactical Guide
- **Prompt:**
```
Professional visualization of Dynamic Programming memoization for the 0/1 Knapsack problem. Clean white background. Show a 2D grid/table labeled "DP Table" with rows representing items (i=0 to 4) and columns representing weight capacity (w=0 to 7). Cells are filled with computed values, with the optimal path highlighted in gold (#B45309). Arrows show how each cell value comes from either the cell directly above (skip item) or from a previous cell plus item value (take item). Include the recurrence relation formula at the top: "DP[i][w] = max(DP[i-1][w], DP[i-1][w-wi] + vi)". Color code: computed cells in light teal, optimal path in gold, current computation in navy. Clean, modern technical diagram suitable for a textbook.
```

---

## 20. Optimistic vs Pessimistic Concurrency (Ch 7)
- **File:** `optimistic_vs_pessimistic.png`
- **Target:** `chapters/07-concurrency-performance/visuals/occ_vs_pcc.png`
- **Chapter:** Ch 7 — Designing for Performance and Concurrency
- **Prompt:**
```
Professional side-by-side comparison diagram of Optimistic vs Pessimistic Concurrency Control in databases. Clean white background. LEFT SIDE labeled "Pessimistic Locking (PCC)": Show two transaction threads (Thread A navy, Thread B teal) trying to access the same database row. Thread A acquires a lock (lock icon), Thread B is BLOCKED with a red wait indicator. After Thread A commits, Thread B proceeds. Caption: "SELECT ... FOR UPDATE". Pros: "Guaranteed safety". Cons: "High contention, deadlock risk". RIGHT SIDE labeled "Optimistic Locking (OCC)": Both threads read the same row simultaneously (no blocking). At commit time, Thread A succeeds (version check passes), Thread B fails with "OptimisticLockException" (version mismatch). Thread B retries. Caption: "WHERE version = ?". Pros: "High throughput". Cons: "Retry on conflict". Clean modern infographic.
```

---

## 21. HikariCP Pool Sizing Formula (Ch 7)
- **File:** `hikaricp_pool_formula.png`
- **Target:** `chapters/07-concurrency-performance/visuals/hikaricp_formula.png`
- **Chapter:** Ch 7 — Designing for Performance and Concurrency
- **Prompt:**
```
Professional infographic showing database connection pool sizing formula. Clean white background. Center: Large formula "Pool Size = (CPU Cores x 2) + Spindle Count" in navy blue. Below: A worked example with an 8-core server and SSD (spindle=1) showing calculation "(8 x 2) + 1 = 17 connections". Two comparison bars: Bar 1 (red) "Common Mistake: 500 connections" showing CPU overwhelmed with context switching. Bar 2 (green) "Optimal: 17 connections" showing efficient throughput. Small chart showing throughput on Y axis vs pool size on X axis, with peak at approximately 17 and declining after. Caption "Based on PostgreSQL benchmark testing". Clean modern infographic style with navy and teal colors. No logos, no placeholder images.
```

---

## 22. Architectural Styles Comparison (Ch 9)
- **File:** `monolith_micro_event.png`
- **Target:** `chapters/09-system-architecture/visuals/arch_styles.png`
- **Chapter:** Ch 9 — System Architecture and Design Fundamentals
- **Prompt:**
```
Professional three-column comparison diagram of architectural styles. Clean white background. Three columns side by side: Column 1 "MONOLITHIC" (navy #1E3A8A): Single large box containing all components (Matching, Ledger, Users) sharing one database. Pros: "Ultra-low latency, Simple transactions". Cons: "Hard to scale, Single deployment". Column 2 "MICROSERVICES" (teal #0D9488): Three separate service boxes connected by REST/gRPC arrows, each with own database. Pros: "Independent deployment, Team autonomy". Cons: "Network latency, Distributed transactions". Column 3 "EVENT-DRIVEN" (amber #B45309): Three service boxes connected through a central Kafka event bus bar, each with own database. Pros: "High decoupling, Resilient". Cons: "Eventual consistency, Complex debugging". Below all three, a decision guide: "Use Monolith for core engine, Microservices for scaling, Events for async". Clean modern infographic.
```

---

## 23. Composition vs Inheritance (Ch 3)
- **File:** `composition_over_inheritance.png`
- **Target:** `chapters/03-oop-principles/visuals/composition_vs_inheritance.png`
- **Chapter:** Ch 3 — Principles of Object-Oriented Design
- **Prompt:**
```
Professional comparison diagram of Composition vs Inheritance in object-oriented design. Clean white background. LEFT SIDE labeled "INHERITANCE (Fragile)" with red accent: Show a deep class hierarchy tree: BaseSettlementService at top, with AchService, FedWireService, VisaService inheriting from it, and further subclasses below. Red arrows show tight coupling. Label: "Changes to parent break all children". RIGHT SIDE labeled "COMPOSITION (Flexible)" with green accent: Show a SettlementProcessor box containing a list of SettlementRoute interfaces. Three separate implementation boxes (AchRoute, FedWireRoute, VisaRoute) plug into the interface independently. Green arrows show loose coupling. Label: "New routes added without modifying existing code". Clean UML-inspired diagram with modern styling.
```

---

## 24. SOLID Principles Reference Card (Ch 4)
- **File:** `solid_five_principles.png`
- **Target:** `chapters/04-solid-boundaries/visuals/solid_summary.png`
- **Chapter:** Ch 4 — SOLID Principles: Enforcing Boundaries
- **Prompt:**
```
Professional summary card of all 5 SOLID principles for software design. Clean white background. Five horizontal rows, each with a large letter on the left (S, O, L, I, D), the principle name in bold, a one-line description, and a small icon. "S" (navy): "Single Responsibility — One module, one reason to change" with a target icon. "O" (teal): "Open/Closed — Open for extension, closed for modification" with a plug icon. "L" (amber): "Liskov Substitution — Subtypes must be substitutable for base types" with a swap icon. "I" (green): "Interface Segregation — Clients should not depend on unused interfaces" with a scissors icon. "D" (purple): "Dependency Inversion — Depend on abstractions, not concretions" with an arrow-flip icon. Clean, modern reference card style suitable for a book page.
```

---

## 25. Cryptographic Audit Trail Chain (Ch 11)
- **File:** `audit_trail_chain.png`
- **Target:** `chapters/11-database-compliance/visuals/audit_trail.png`
- **Chapter:** Ch 11 — Database Design, Compliance, and Security
- **Prompt:**
```
Professional diagram of a tamper-proof cryptographic audit trail for SOC2 compliance. Clean white background. Show a horizontal chain of 5 audit log blocks connected by hash arrows, similar to a blockchain. Each block contains: "Row ID", "Timestamp", "Action (DEBIT/CREDIT)", "Amount", "Actor", and "Hash(prev + current)". The hash of each block feeds into the next block via a blue arrow. Show one block in the middle highlighted in red with a "TAMPERED" label, and the chain breaking (hash mismatch detected) on the next block. Include labels: "Append-Only Table", "WORM Storage", "Immutable Ledger". Clean modern infographic style with navy and teal colors.
```

---

## 26. ZenithTrade Order Lifecycle Sequence (Ch 9)
- **File:** `order_lifecycle_sequence_light.png`
- **Target:** `chapters/09-system-architecture/visuals/order_lifecycle.png`
- **Chapter:** Ch 9 — System Architecture and Design Fundamentals
- **Prompt:**
```
Professional sequence diagram showing ZenithTrade Order Lifecycle Sequence. Clean white background. Vertical lifelines: "Client/Trader" (green), "API Gateway" (teal), "Order Validator" (navy), "Matching Engine" (gold), "AuraPay Ledger" (purple), "Notification Service" (blue). Show sequential numbered horizontal interaction arrows between lifelines: 1. Submit Limit Order, 2. Validate Order, 3. Match Order, 4. Double-Entry Settlement, 5. Publish Event, 6. WebSockets/SMS Confirmation. Clean technical diagram style with clear boxes and labels. Navy, teal, and gold colors.
```

---

## 27. Redis Sliding Window Rate Limiting (Ch 10)
- **File:** `rate_limiter_light.png`
- **Target:** `chapters/10-resiliency/visuals/rate_limiter.png`
- **Chapter:** Ch 10 — Enterprise Integration and Resiliency
- **Prompt:**
```
Professional diagram showing Redis Sliding Window Rate Limiting. Clean white background. A timeline showing request timestamps represented as dots. A sliding window box (teal #0D9488) of width 1 minute moving over the timeline. Below, show Redis Sorted Set commands: 1. ZADD key timestamp UUID (to add current request), 2. ZREMRANGEBYSCORE key -inf (timestamp - 1 min) (to remove old requests), 3. ZCARD key (to count remaining requests), 4. EXPIRE key 60 (to set TTL). Labeled comparison arrows: ZCARD <= limit (allow request, green arrow), ZCARD > limit (reject request, red arrow). Clean, modern technical diagram.
```

---

## 28. Automated Testing Pyramid (Ch 13)
- **File:** `testing_pyramid_light.png`
- **Target:** `chapters/13-testing-cicd/visuals/testing_pyramid.png`
- **Chapter:** Ch 13 — Testing and CI/CD Strategies for High-Performance Systems
- **Prompt:**
```
Professional software testing pyramid diagram. Clean white background. A large pyramid divided into 3 horizontal sections: Top section (small, gold #B45309) labeled "E2E & Contract Tests (Pact)" with a browser icon. Middle section (medium, teal #0D9488) labeled "Integration Tests (Testcontainers)" with a database cylinder icon and docker whale icon. Bottom section (large, navy #1E3A8A) labeled "Unit Tests (JUnit/Mockito)" with a code file and puzzle piece icon. On the left side of the pyramid, show a vertical arrow pointing UP labeled "Execution Time / Cost (High)". On the right side, show a vertical arrow pointing DOWN labeled "Isolation / Feedback Velocity (Fast)". Clean modern technical infographic.
```

---

## 29. Apache Kafka Topic Partitions (Ch 14)
- **File:** `kafka_internals_light.png`
- **Target:** `chapters/14-message-brokers/visuals/kafka_internals.png`
- **Chapter:** Ch 14 — Distributed Event Streaming and Message Brokers
- **Prompt:**
```
Professional architecture diagram showing Apache Kafka Topic Partitions and Consumer Groups. Clean white background. In the center, a box representing "Kafka Topic: transactions" containing three parallel partition queues: "Partition 0", "Partition 1", and "Partition 2" filled with message blocks. Arrows show messages sharded by partition key "accountId". On the right, a dashed box labeled "Consumer Group: ledger-service" containing three consumer instances: "Consumer A", "Consumer B", and "Consumer C". Show arrows mapping Partition 0 to Consumer A, Partition 1 to Consumer B, and Partition 2 to Consumer C. Show message blocks with partition offsets (0, 1, 2...). Clean modern technical diagram with navy (#1E3A8A) and teal (#0D9488) accents.
```

---

## 30. LLM RAG Pipeline (Ch 15)
- **File:** `rag_architecture_light.png`
- **Target:** `chapters/15-aiml-llm/visuals/rag_architecture.png`
- **Chapter:** Ch 15 — AI/ML System Design and LLM Integration
- **Prompt:**
```
Professional architecture diagram for Retrieval-Augmented Generation (RAG) pipeline. Clean white background with light gray grid. Flow showing: 1. User Query (user icon) enters. 2. Query sent to "Embedding Model" (neural network icon) to generate query vector. 3. Vector query sent to "Vector Database" (pgvector/Pinecone cube icon). 4. Vector DB returns top-K relevant document contexts. 5. Context and User Query combined inside "Prompt Template" box. 6. Final prompt sent to "Large Language Model (LLM)" (brain icon). 7. LLM returns natural language response to user. Use clean arrows showing step numbers 1 to 7. Navy blue (#1E3A8A) and gold (#B45309) accents. Modern infographic.
```

---

## File Mapping — Generated Image → Book Visuals

| # | Generated File | Chapter | Copy To |
|---|---|---|---|
| 1 | `cover_light.png` | Cover | `visuals/cover.png` |
| 2 | `aurapay_arch_light.png` | Ch 2 | `chapters/02-case-studies/visuals/aurapay_architecture.png` |
| 3 | `circuit_breaker_light.png` | Ch 10 | `chapters/10-resiliency/visuals/circuit_breaker.png` |
| 4 | `sliding_window_light.png` | Ch 8 | `chapters/08-algorithms-assessment/visuals/sliding_window.png` |
| 5 | `ddd_contexts_light.png` | Ch 9 | `chapters/09-system-architecture/visuals/ddd_contexts.png` |
| 6 | `saga_compare_light.png` | Ch 10 | `chapters/10-resiliency/visuals/saga_comparison.png` |
| 7 | `outbox_pattern_light.png` | Ch 10 | `chapters/10-resiliency/visuals/outbox_pattern.png` |
| 8 | `threads_comparison.png` | Ch 7 | `chapters/07-concurrency-performance/visuals/virtual_threads.png` |
| 9 | `btree_lsm_light.png` | Ch 11 | `chapters/11-database-compliance/visuals/btree_vs_lsm.png` |
| 10 | `tokenization_vault_light.png` | Ch 11 | `chapters/11-database-compliance/visuals/tokenization_vault.png` |
| 11 | `pattern_flow_light.png` | Ch 8 | `chapters/08-algorithms-assessment/visuals/pattern_flowchart.png` |
| 12 | `anemic_vs_rich_light.png` | Ch 3 | `chapters/03-oop-principles/visuals/anemic_vs_rich.png` |
| 13 | `observer_pattern_light.png` | Ch 6 | `chapters/06-design-patterns/visuals/observer_pattern.png` |
| 14 | `solid_dip_light.png` | Ch 4 | `chapters/04-solid-boundaries/visuals/solid_dip.png` |
| 15 | `stream_pipeline_light.png` | Ch 5 | `chapters/05-functional-streams/visuals/stream_pipeline.png` |
| 16 | `spec_vs_syntax_path.png` | Prologue | `chapters/00-prologue/visuals/spec_vs_syntax.png` |
| 17 | `invariant_wall_layers.png` | Ch 1 | `chapters/01-invariant-first/visuals/invariant_wall.png` |
| 18 | `gca_time_allocation.png` | Ch 8 | `chapters/08-algorithms-assessment/visuals/gca_timeline.png` |
| 19 | `dp_memoization_table.png` | Ch 8 | `chapters/08-algorithms-assessment/visuals/dp_table.png` |
| 20 | `optimistic_vs_pessimistic.png` | Ch 7 | `chapters/07-concurrency-performance/visuals/occ_vs_pcc.png` |
| 21 | `hikaricp_pool_formula.png` | Ch 7 | `chapters/07-concurrency-performance/visuals/hikaricp_formula.png` |
| 22 | `monolith_micro_event.png` | Ch 9 | `chapters/09-system-architecture/visuals/arch_styles.png` |
| 23 | `composition_over_inheritance.png` | Ch 3 | `chapters/03-oop-principles/visuals/composition_vs_inheritance.png` |
| 24 | `solid_five_principles.png` | Ch 4 | `chapters/04-solid-boundaries/visuals/solid_summary.png` |
| 25 | `audit_trail_chain.png` | Ch 11 | `chapters/11-database-compliance/visuals/audit_trail.png` |
| 26 | `order_lifecycle_sequence_light.png` | Ch 9 | `chapters/09-system-architecture/visuals/order_lifecycle.png` |
| 27 | `rate_limiter_light.png` | Ch 10 | `chapters/10-resiliency/visuals/rate_limiter.png` |
| 28 | `testing_pyramid_light.png` | Ch 13 | `chapters/13-testing-cicd/visuals/testing_pyramid.png` |
| 29 | `kafka_internals_light.png` | Ch 14 | `chapters/14-message-brokers/visuals/kafka_internals.png` |
| 30 | `rag_architecture_light.png` | Ch 15 | `chapters/15-aiml-llm/visuals/rag_architecture.png` |
| 31 | `sliding_window_trace.png` | Ch 8 | `chapters/08-algorithms-assessment/visuals/sliding_window_trace.png` |
| 32 | `two_pointer_trace.png` | Ch 8 | `chapters/08-algorithms-assessment/visuals/two_pointer_trace.png` |
| 33 | `greedy_interval_trace.png` | Ch 8 | `chapters/08-algorithms-assessment/visuals/greedy_interval_trace.png` |
| 34 | `bfs_dfs_trace.png` | Ch 8 | `chapters/08-algorithms-assessment/visuals/bfs_dfs_trace.png` |
| 35 | `dp_lcs_trace.png` | Ch 8 | `chapters/08-algorithms-assessment/visuals/dp_lcs_trace.png` |

---

*Last updated: 2026-07-19 — Prompt catalog fully synchronized (35 visuals total)*

---

## 31. Sliding Window Maximum — Execution Trace (Ch 8)
- **File:** `sliding_window_trace.png`
- **Target:** `chapters/08-algorithms-assessment/visuals/sliding_window_trace.png`
- **Chapter:** Ch 8 — Core Algorithms & Assessment Tactical Guide
- **Prompt:**
```
Create a clean, professional algorithm tracing diagram on a pure white background showing a "Sliding Window Maximum" step-by-step execution trace.

Array: [1, 3, -1, -3, 5, 3, 6, 7], window size k=3

Show 6 rows/steps, each row showing:
- Step number (Step 1 through Step 6)
- The full array with the current window highlighted in blue/teal
- The deque state shown as a horizontal queue showing indices and their values
- The output array building up progressively

Step 1: Window [1,3,-1], Deque: [1(3)], Max=3, Output: [3]
Step 2: Window [3,-1,-3], Deque: [1(3)], Max=3, Output: [3,3]
Step 3: Window [-1,-3,5], Deque: [4(5)], Max=5, Output: [3,3,5]
Step 4: Window [-3,5,3], Deque: [4(5),5(3)], Max=5, Output: [3,3,5,5]
Step 5: Window [5,3,6], Deque: [6(6)], Max=6, Output: [3,3,5,5,6]
Step 6: Window [3,6,7], Deque: [7(7)], Max=7, Output: [3,3,5,5,6,7]

Use clean arrows showing window sliding right. Color scheme: white background, teal for active window, gray for inactive elements, navy for text. Professional textbook quality diagram.
```

---

## 32. Two Pointer — Container With Most Water Trace (Ch 8)
- **File:** `two_pointer_trace.png`
- **Target:** `chapters/08-algorithms-assessment/visuals/two_pointer_trace.png`
- **Chapter:** Ch 8 — Core Algorithms & Assessment Tactical Guide
- **Prompt:**
```
Create a clean, professional algorithm tracing diagram on a pure white background showing the "Two Pointer - Container With Most Water" step-by-step execution.

Title: "Two Pointer Pattern: Container With Most Water"

Array heights: [1, 8, 6, 2, 5, 4, 8, 3, 7]

Show the array as vertical bars (bar chart style), with two pointers (L and R) shown as arrows below.

Show 4 key steps:
Step 1: L=0(h=1), R=8(h=7), Area = min(1,7) × 8 = 8. Move L (shorter side). Best=8
Step 2: L=1(h=8), R=8(h=7), Area = min(8,7) × 7 = 49. Move R. Best=49
Step 3: L=1(h=8), R=7(h=3), Area = min(8,3) × 6 = 18. Move R. Best=49
Step 4: L=1(h=8), R=6(h=8), Area = min(8,8) × 5 = 40. Move R. Best=49

Show the water area shaded in light blue between the two pointer bars.
Below each step, show: "Move the shorter pointer inward"

Color scheme: white background, teal bars, light blue water fill, red arrows for pointers, navy text. Professional textbook quality.
```

---

## 33. Greedy Interval Scheduling — Execution Trace (Ch 8)
- **File:** `greedy_interval_trace.png`
- **Target:** `chapters/08-algorithms-assessment/visuals/greedy_interval_trace.png`
- **Chapter:** Ch 8 — Core Algorithms & Assessment Tactical Guide
- **Prompt:**
```
Create a clean, professional algorithm tracing diagram on a pure white background showing the "Greedy Interval Scheduling" step-by-step execution.

Title: "Greedy Pattern: Interval Scheduling (Earliest Deadline First)"

Show a timeline (horizontal axis 0 to 14) with intervals as horizontal colored bars:
Input intervals (unsorted): [1,4], [2,3], [3,5], [6,7], [5,9], [8,10], [11,13]

Step 1: Sort by end time → [2,3], [1,4], [3,5], [6,7], [5,9], [8,10], [11,13]
Step 2: Select [2,3] ✓ (first interval). lastEnd = 3
Step 3: Skip [1,4] ✗ (start 1 < lastEnd 3, overlaps)
Step 4: Select [3,5] ✓ (start 3 >= lastEnd 3). lastEnd = 5
Step 5: Select [6,7] ✓ (start 6 >= lastEnd 5). lastEnd = 7
Step 6: Skip [5,9] ✗ (start 5 < lastEnd 7, overlaps)
Step 7: Select [8,10] ✓ (start 8 >= lastEnd 7). lastEnd = 10
Step 8: Select [11,13] ✓ (start 11 >= lastEnd 10). lastEnd = 13

Show selected intervals in green, skipped in red/gray with strikethrough.
Result: 5 non-overlapping intervals selected.

Color scheme: white background, green for selected, red strikethrough for skipped, navy text. Professional textbook quality.
```

---

## 34. BFS vs DFS — Graph Traversal Comparison (Ch 8)
- **File:** `bfs_dfs_trace.png`
- **Target:** `chapters/08-algorithms-assessment/visuals/bfs_dfs_trace.png`
- **Chapter:** Ch 8 — Core Algorithms & Assessment Tactical Guide
- **Prompt:**
```
Create a clean, professional algorithm tracing diagram on a pure white background showing BFS vs DFS traversal side by side.

Title: "BFS vs DFS: Graph Traversal Comparison"

Show a simple graph with 7 nodes (labeled 0-6) arranged in a tree-like shape:
   0
  / \
 1   2
/ \   \
3  4   5
   |
   6

LEFT SIDE - BFS (Queue):
Show step-by-step with queue state:
Step 1: Visit 0, Queue: [1, 2]
Step 2: Visit 1, Queue: [2, 3, 4]
Step 3: Visit 2, Queue: [3, 4, 5]
Step 4: Visit 3, Queue: [4, 5]
Step 5: Visit 4, Queue: [5, 6]
Step 6: Visit 5, Queue: [6]
Step 7: Visit 6, Queue: []
Order: 0 → 1 → 2 → 3 → 4 → 5 → 6 (Level by level)

RIGHT SIDE - DFS (Stack/Recursion):
Step 1: Visit 0 → Visit 1 → Visit 3 (go deep)
Step 2: Backtrack to 1 → Visit 4 → Visit 6 (go deep)
Step 3: Backtrack to 0 → Visit 2 → Visit 5
Order: 0 → 1 → 3 → 4 → 6 → 2 → 5 (Branch by branch)

Show arrows with numbered order on each edge. BFS in blue, DFS in orange.
Color scheme: white background, blue for BFS, orange for DFS, navy text. Professional textbook quality.
```

---

## 35. Dynamic Programming — LCS Table Fill Trace (Ch 8)
- **File:** `dp_lcs_trace.png`
- **Target:** `chapters/08-algorithms-assessment/visuals/dp_lcs_trace.png`
- **Chapter:** Ch 8 — Core Algorithms & Assessment Tactical Guide
- **Prompt:**
```
Create a clean, professional algorithm tracing diagram on a pure white background showing the Dynamic Programming "Longest Common Subsequence" step-by-step table fill.

Title: "Dynamic Programming: Longest Common Subsequence (LCS)"

Text1: "ABCDE"
Text2: "ACE"

Show the DP table being filled step by step:
- Rows labeled with "" (empty), A, C, E (text2)
- Columns labeled with "" (empty), A, B, C, D, E (text1)

Fill values:
     ""  A  B  C  D  E
""  [ 0, 0, 0, 0, 0, 0]
A   [ 0, 1, 1, 1, 1, 1]
C   [ 0, 1, 1, 2, 2, 2]
E   [ 0, 1, 1, 2, 2, 3]

Highlight the diagonal matches (where characters match) in green:
- (A,A) → diagonal +1 = 1
- (C,C) → diagonal +1 = 2
- (E,E) → diagonal +1 = 3

Show arrows in the table:
- Diagonal arrow (green) when characters match
- Right/down arrow (gray) when taking max of neighbors

At bottom: "LCS = ACE, Length = 3"
Show the backtrack path highlighted with a colored trail through the table.

Color scheme: white background, green for matches, gray for non-matches, teal for the backtrack path, navy text. Professional textbook quality.
```

---

## 30. Technical STAR Framework (Ch 12)
- **File:** `technical_star.png`
- **Target:** `chapters/12-behavioral-leadership/visuals/technical_star.png`
- **Chapter:** Ch 12 — Behavioral Interviews & Leadership
- **Prompt:**
```
Professional sequence diagram showing the Technical STAR Framework for senior engineering interviews. Clean white background, modern textbook infographic style. At the top of the diagram, show the title "Technical STAR Framework for Senior Engineering Interviews" in a clean, bold, modern font with absolutely NO single quotes or quotation marks around any part of the title. Show 4 horizontal steps connected by neat arrows: Step 1: 'Situation (S)' (Sub-bullets: Business Context, Current Scale, Legacy Bottlenecks). Colored a soft blue. Step 2: 'Task (T)' (Sub-bullets: Architectural Goals, Target SLAs/SLOs, Constraints). Colored a soft purple. Step 3: 'Action & Trade-offs (A)' (Sub-bullets: Design Options Evaluated, Decision Rationale, Leadership & Execution). Colored a soft green/teal. Step 4: 'Result & Impact (R)' (Sub-bullets: p99 Latency Improvements, Cost Reductions, Compliance & SOC2). Colored a soft amber/orange. Draw crisp arrows connecting the steps. Highly readable, crisp typography, clean technical infographic style. No hex code labels inside the blocks.
```
