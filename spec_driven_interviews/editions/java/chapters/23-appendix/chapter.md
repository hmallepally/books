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

| Operation | Time (ns) | Time (Human Scale) |
|---|---|---|
| **L1 Cache reference** | 0.5 ns | 0.5 sec |
| **Branch mispredict** | 5 ns | 5 sec |
| **L2 Cache reference** | 7 ns | 7 sec |
| **Main Memory reference (RAM)** | 100 ns | 1.6 min |
| **Compress 1K bytes with Zippy** | 3,000 ns | 50 min |
| **Send 2K bytes over 1 Gbps network** | 20,000 ns | 5.5 hours |
| **Read 1MB sequentially from SSD** | 1,000,000 ns | 11.5 days |
| **Round trip within same datacenter** | 500,000 ns | 5.7 days |
| **Read 1MB sequentially from Disk** | 20,000,000 ns | 7.5 months |
| **Send packet CA to Netherlands to CA** | 150,000,000 ns | 4.7 years |


## Day of the Interview Checklist

Before entering a live call (Teams/Zoom) or in-person evaluation, ensure you have completed these checks:

- [ ] **Whiteboard Readiness:** If using a digital whiteboard, log in and verify that your shortcut keys and drawing shapes function correctly.
- [ ] **Code Editor Settings:** Turn off all autocomplete/AI copilot extensions inside your IDE or browser coding window. Interviewers expect you to write clean syntax without AI assistance.
- [ ] **Audio/Video Setup:** Clean background, clear microphone, and a stable internet connection.
- [ ] **The "Trade-off" Mindset:** Remember, there are no "perfect" architectures in system design. For every choice you make, write down the corresponding scale, cost, or complexity trade-off on your whiteboard.
- [ ] **Boundary Verification:** Write your pre-conditions, post-conditions, and invariants *first* before implementing any code. Protect the boundary.
