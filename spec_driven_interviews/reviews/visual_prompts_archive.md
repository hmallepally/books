# Visual Generation Prompts Archive

> All prompts used to generate the 8 CRITICAL visuals for the Spec-Driven Technical Interviews book.
> These can be reused to regenerate visuals at higher resolution or with different styles.

---

## V1. Problem Decomposition Tree (Ch 02)
**File:** `chapters/02-problem-decomposition/visuals/decomposition_tree.jpg`

**Prompt:**
> A clean, professional technical diagram showing a "Problem Decomposition Tree" for software engineering interviews. At the top is a complex problem box labeled "Trapping Rain Water". Below it branches into 3 sub-problems: "1. Find maxLeft for each index", "2. Find maxRight for each index", "3. Calculate water at each index = min(maxLeft, maxRight) - height". Each sub-problem has an arrow pointing to its solution approach: "Prefix Array", "Suffix Array", "Single Pass". Use a dark theme with blue accent colors, clean sans-serif font, and minimal design. No background decorations.

---

## V2. Big-O Comparison Chart (Ch 09)
**File:** `chapters/09-algorithms-assessment/visuals/big_o_comparison.jpg`

**Prompt:**
> A clean, professional line graph showing Big-O complexity comparison for software engineering interviews. X-axis labeled "Input Size (N)" from 0 to 100. Y-axis labeled "Operations". Six colored curves: O(1) as a flat green line at the bottom, O(log N) as a barely rising blue curve, O(N) as a diagonal cyan line, O(N log N) as a slightly steeper yellow curve, O(N squared) as a sharply rising orange parabola, and O(2^N) as an extremely steep red exponential curve shooting off the chart. Each curve is clearly labeled with its complexity class. Dark theme, grid lines visible, clean sans-serif font. Professional data visualization style.

---

## V3. CAP Theorem Triangle (Ch 16)
**File:** `chapters/16-system-architecture/visuals/cap_theorem.jpg`

**Prompt:**
> A clean, professional CAP Theorem diagram for system design interviews. A triangle with three vertices: "Consistency" (top), "Availability" (bottom-left), "Partition Tolerance" (bottom-right). Between each pair of vertices, show the trade-off: CA systems (PostgreSQL, MySQL) between C and A, CP systems (MongoDB, HBase, Redis Cluster) between C and P, AP systems (Cassandra, DynamoDB, CouchDB) between A and P. Center of triangle says "Pick 2 of 3 (in practice: choose C or A when P occurs)". Dark theme with blue/purple accent colors, clean sans-serif font, professional technical documentation style.

---

## V4. System Evolution Diagram (Ch 16)
**File:** `chapters/16-system-architecture/visuals/system_evolution.jpg`

**Prompt:**
> A clean, professional 3-stage system architecture evolution diagram for system design interviews. Stage 1 (left): "Monolith" - single box containing Web Server, Business Logic, and Database all together, labeled "1-1K users". Stage 2 (middle): "Microservices" - separate boxes for API Gateway, User Service, Order Service, Payment Service, each with own DB, connected by arrows, labeled "1K-1M users". Stage 3 (right): "Event-Driven" - microservices connected through a central Event Bus (Kafka), with CQRS read/write separation, CDN, and Cache layer, labeled "1M+ users". Arrows showing evolution between stages. Dark theme with gradient blue colors, clean sans-serif font.

---

## V5. Kafka Partitions / Consumer Groups (Ch 21)
**File:** `chapters/21-message-brokers/visuals/kafka_partitions.jpg`

**Prompt:**
> A clean, professional Kafka partition and consumer group architecture diagram. Left side: 3 Producer boxes sending messages. Center: a Topic split into 4 Partitions (P0, P1, P2, P3), each showing message offsets (0,1,2,3...). Right side: Consumer Group A with 2 consumers (C1 reads P0+P1, C2 reads P2+P3) and Consumer Group B with 4 consumers (one per partition). Arrows show message flow from producers to partitions to consumers. Show consumer offsets being tracked. Dark theme with blue/teal accent colors, clean sans-serif font, professional technical documentation style.

---

## V6. Problem Analysis Canvas (Ch 14)
**File:** `chapters/14-mastering-decomposition/visuals/problem_analysis_canvas.jpg`

**Prompt:**
> A clean, professional Problem Analysis Canvas template for coding interviews. A 2x2 grid layout with 4 quadrants: Top-left: "INPUTS" (What data do I receive? What are the types? Can inputs be empty/null?), Top-right: "OUTPUTS" (What must I return? What format? What if no valid answer?), Bottom-left: "CONSTRAINTS" (N range? Time limit? Memory limit? Sorted? Unique values?), Bottom-right: "EDGE CASES" (Empty input? Single element? All same values? Maximum size? Negative numbers?). Center circle: "INVARIANT: What must ALWAYS be true?". Clean professional layout with subtle grid lines, dark theme, blue accent colors, sans-serif font.

---

## V7. Deadlock Diagram (Ch 08)
**File:** `chapters/08-concurrency-performance/visuals/deadlock_diagram.jpg`

**Prompt:**
> A clean, professional deadlock diagram for software engineering. Shows 2 threads (Thread A and Thread B) as blue rectangles and 2 resources (Lock X and Lock Y) as orange circles. Thread A holds Lock X (solid arrow from Lock X to Thread A labeled "held by") and waits for Lock Y (dashed arrow from Thread A to Lock Y labeled "waiting for"). Thread B holds Lock Y (solid arrow from Lock Y to Thread B labeled "held by") and waits for Lock X (dashed arrow from Thread B to Lock X labeled "waiting for"). The circular dependency is highlighted with a red "DEADLOCK" warning in the center. Dark theme, clean sans-serif font, professional technical documentation style.

---

## V8. Lazy Evaluation Pipeline (Ch 06)
**File:** `chapters/06-functional-streams/visuals/lazy_evaluation.jpg`

**Prompt:**
> A clean, professional lazy evaluation pipeline diagram for functional programming. Shows a horizontal data pipeline with 5 stages. Source: an array [1, 5, 3, 8, 2, 7, 4, 6]. Stage 1: filter(x > 3) - only elements 5, 8, 7, 6 pass through. Stage 2: map(x * 2) - transforms to 10, 16, 14, 12. Stage 3: findFirst() - terminal operation, takes only 10 and STOPS. Key insight shown: elements 8, 7, 6 are NEVER processed because findFirst() short-circuits after the first match. Gray out the unprocessed elements. Label: "Lazy: only 2 elements processed instead of 8". Dark theme with green pipeline arrows, red X marks on skipped elements, clean sans-serif font.

---

## Still Needed (IMPORTANT priority, not yet generated)

| # | Visual | Chapter | Type |
|---|--------|---------|------|
| V9 | Before/After state diagrams for loop invariants | Ch 01 | State Diagram |
| V10 | Constraint Analysis Flowchart | Ch 02 | Flowchart |
| V11 | ZenithTrade architecture | Ch 03 | Architecture |
| V12 | ChiramTrust architecture | Ch 03 | Architecture |
| V13 | OOP violation detector class diagrams | Ch 04 | UML |
| V14 | Thread lifecycle state machine | Ch 08 | State Machine |
| V15 | Decomposition decision tree | Ch 14 | Decision Tree |
| V16 | Timer/pacing strategy timeline | Ch 15 | Timeline |
| V17 | Consistent hashing ring | Ch 16 | Diagram |
| V18 | Sharding strategies | Ch 18 | Data Flow |
| V19 | Model serving architecture | Ch 22 | Architecture |
