# Core Algorithms & Assessment Tactical Guide

> *"Algorithms are not trivia; they are the baseline vocabulary of computational efficiency under resource constraints."*

## From Theory to Tactical Execution

The Prologue established *why* pattern-based problem solving beats memorization. This chapter provides the *how*: a complete tactical guide to the 25 Canonical Programming Patterns, the 70-minute speed-run blueprint, and the six difficulty modules that map directly to Chapters 10–13.

1. **Learn the 25 Canonical Programming Patterns** — the core mathematical invariants and code skeletons that govern all algorithmic problems.
2. **Analyze the problem structure** to map requirements directly to a pattern ID (`[PAT-01]` through `[PAT-25]`).
3. **Practice by doing.** Implement 2–3 problems for each pattern independently until the code skeleton becomes pure muscle memory.

When you master the 25 patterns below, you no longer need to memorize hundreds of solutions. You simply recognize the pattern, apply the appropriate code skeleton, and derive the solution cleanly on demand.

## General Coding Assessment (GCA) Tactics

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

| Big-O Time Complexity Comparison Graph |
|---|
| ![Big-O Time Complexity Comparison Graph](visuals/big_o_comparison.jpg){width=85%} |

**The Constraint-to-Complexity Rule:** Read the problem constraints FIRST. If N ≤ 10^4, O(N²) is acceptable. If N ≤ 10^5, you need O(N log N) or better. If N ≤ 10^6, you need O(N). This single rule eliminates 50% of wrong algorithm choices before you write a line of code.

![Constraint-to-Complexity Flowchart](visuals/constraint_flowchart.jpg){width=85%}

# The 25 Canonical Programming Patterns

The following catalog defines the 25 fundamental patterns of computational problem-solving. Each pattern represents a proven, invariant structure for solving a specific class of problems.

Every pattern is structured around a **5-Part Pedagogical Blueprint**:

1. **Formal Neutral Invariant:** A mathematically precise, language-neutral, domain-agnostic statement of the pattern's core property.
2. **Intuitive Mental Model:** A conceptual operational metaphor explaining the mechanism.
3. **Concrete Tracing Exemplar:** A canonical problem used to demonstrate the pattern step-by-step.
4. **Visual Architecture / Data-Flow Diagram:** A structural diagram illustrating data structures, pointer movements, and state transformations.
5. **Step-by-Step State Trace Table:** A detailed execution trace tracking iteration steps, pointer positions, data structure states, and variables.


## Module 1: Array & String Mechanics

### [PAT-01] Direct Indexing & Frequency Buckets

- **Invariant (Neutral):** Given a bounded discrete key space $K \in [0, U-1]$ of size $U$, a direct-mapped array $A$ of size $U$ performs element insertion, lookup, and frequency counting in $O(1)$ time and $O(1)$ space without hashing overhead or collision handling.
- **Mental Model:** A labeled key rack where every key slides directly into a pre-assigned numerical slot corresponding to its value.
- **Concrete Tracing Exemplar:** First Non-Repeating Character in a String (e.g., `s = "leetcode"`).
- **Visual Architecture / Data-Flow Diagram:**

```
Input String: "leetcode"
Character:    'l' (108)  'e' (101)  'e' (101)  't' (116)  ...
                 │          │          │          │
Offset Map:  (c - 'a')  (c - 'a')  (c - 'a')  (c - 'a')
                 │          │          │          │
Index:          [11]       [4]        [4]        [19]
Frequency Array: ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
                 │ 0 │...│ 2 │...│ 1 │...│ 1 │...│ 0 │
                 └───┴───┴───┴───┴───┴───┴───┴───┴───┘
                 idx: 0     4         11        19    25
```

- **Canonical Code Skeleton:**
```csharp
public int FirstUniqueChar(string s)
{
    int[] counts = new int[256];
    foreach (char c in s) counts[c]++;
    for (int i = 0; i < s.Length; i++)
    {
        if (counts[s[i]] == 1) return i;
    }
    return -1;
}
```


- **Step-by-Step State Trace (Input: `s = "leetcode"`):**

| Step | Char | Index (`c - 'a'`) | Array State at Index | Action |
|:---:|:---:|:---:|:---:|:---|
| 1 | `'l'` | 11 | `freq[11] = 1` | Increment count |
| 2 | `'e'` | 4 | `freq[4] = 1` | Increment count |
| 3 | `'e'` | 4 | `freq[4] = 2` | Increment count |
| 4 | `'t'` | 19 | `freq[19] = 1` | Increment count |
| 5 | `'c'` | 2 | `freq[2] = 1` | Increment count |
| 6 | `'o'` | 14 | `freq[14] = 1` | Increment count |
| 7 | `'d'` | 3 | `freq[3] = 1` | Increment count |
| 8 | `'e'` | 4 | `freq[4] = 3` | Increment count |

- **Diagnostic Triggers:** "First non-repeating character", "Anagram check", "Fixed alphabet character frequency".
- **Boundary Conditions:** Verify key space bounds ($U=26$ for lowercase English, $U=128$ for ASCII, $U=256$ for Extended ASCII).
- **Real-World Application:** High-speed network packet header inspection, audit log byte-frequency analysis.

### [PAT-02] In-Place Mutation & Two-Pointer Compaction

- **Invariant (Neutral):** Subarray $A[0..w-1]$ maintains all elements satisfying predicate $P(x)$ in their original relative order, while read pointer $r$ scans elements $0..N-1$. The write pointer $w$ advances if and only if $P(A[r]) = \text{true}$, achieving $O(N)$ time and $O(1)$ auxiliary space.
- **Mental Model:** A filter funnel where valid items are compacted behind a moving boundary while invalid items are overwritten or pushed outside the valid range.
- **Concrete Tracing Exemplar:** Move Zeros to End (e.g., `nums = [0, 1, 0, 3, 12]`).
- **Visual Architecture / Data-Flow Diagram:**

```
Initial:  [ 0 , 1 , 0 , 3 , 12 ]
            ▲   ▲
            w   r  (P(0) is false: r moves, w stays)

Step 1:   [ 1 , 1 , 0 , 3 , 12 ]
                ▲   ▲
                w   r  (P(1) is true: write 1 at w, increment w & r)

Step 2:   [ 1 , 3 , 0 , 3 , 12 ]
                    ▲           ▲
                    w           r  (P(3) & P(12) true: copy to w)

Final Fill: [ 1 , 3 , 12 , 0 , 0 ]  (fill w..N-1 with 0)
                           ▲
                           w
```

- **Canonical Code Skeleton:**
```csharp
public int RemoveDuplicates(int[] nums)
{
    if (nums.Length == 0) return 0;
    int write = 1;
    for (int read = 1; read < nums.Length; read++)
    {
        if (nums[read] != nums[read - 1])
        {
            nums[write++] = nums[read];
        }
    }
    return write;
}
```


- **Step-by-Step State Trace (Input: `nums = [0, 1, 0, 3, 12]`):**

| Step | `r` | `nums[r]` | $P(\text{val}) \neq 0$ | Action | `w` | Array State $nums[0..4]$ |
|:---:|:---:|:---:|:---:|:---|:---:|:---|
| Init | 0 | 0 | False | Skip | 0 | `[0, 1, 0, 3, 12]` |
| 1 | 1 | 1 | True | `nums[w] = 1; w++` | 1 | `[1, 1, 0, 3, 12]` |
| 2 | 2 | 0 | False | Skip | 1 | `[1, 1, 0, 3, 12]` |
| 3 | 3 | 3 | True | `nums[w] = 3; w++` | 2 | `[1, 3, 0, 3, 12]` |
| 4 | 4 | 12 | True | `nums[w] = 12; w++`| 3 | `[1, 3, 12, 3, 12]` |
| Fill | - | - | - | Zero fill `w..N-1` | 3 | `[1, 3, 12, 0, 0]` |

- **Diagnostic Triggers:** "In-place array compaction", "Remove element without extra memory", "Move specific elements to end".
- **Boundary Conditions:** Handle empty array or array containing all valid/all invalid elements upfront.
- **Real-World Application:** In-memory garbage collection compaction, log stream filtering.

### [PAT-03] Prefix Sums & Range Query Invariants

- **Invariant (Neutral):** For array $A$ of length $N$, a precomputed cumulative array $P$ where $P[k] = \sum_{m=0}^{k-1} A[m]$ allows any contiguous subarray sum $\sum_{m=i}^{j} A[m]$ to be calculated in $O(1)$ time via the difference $P[j+1] - P[i]$.
- **Mental Model:** Odometer distance subtraction—computing trip distance between two milestones by subtracting initial odometer reading from final reading.
- **Concrete Tracing Exemplar:** Subarray Sum Equals K (e.g., `nums = [1, 1, 1], k = 2`).
- **Visual Architecture / Data-Flow Diagram:**

```
Array A:        [  1  ,  1  ,  1  ]
Indices:           0      1      2
Prefix Sum P: [ 0 , 1  ,  2  ,  3  ]
Indices:        0   1      2      3

Range Sum A[1..2] = P[3] - P[1] = 3 - 1 = 2
```

- **Canonical Code Skeleton:**
```csharp
public int SubarraySumEqualsK(int[] nums, int k)
{
    var prefCounts = new Dictionary<int, int>();
    prefCounts[0] = 1;
    int currentSum = 0, count = 0;

    foreach (int num in nums)
    {
        currentSum += num;
        if (prefCounts.TryGetValue(currentSum - k, out int val))
        {
            count += val;
        }
        prefCounts[currentSum] = prefCounts.GetValueOrDefault(currentSum, 0) + 1;
    }
    return count;
}
```


- **Step-by-Step State Trace (Input: `nums = [1, 1, 1], k = 2`, `map={0:1}`):**

| Step | `i` | `nums[i]` | `prefSum` | Complement (`prefSum - k`) | Found in Map? | Count | Map State |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Init | - | - | 0 | - | - | 0 | `{0: 1}` |
| 1 | 0 | 1 | 1 | $1 - 2 = -1$ | No | 0 | `{0: 1, 1: 1}` |
| 2 | 1 | 1 | 2 | $2 - 2 = 0$ | Yes (`map[0]=1`)| 1 | `{0: 1, 1: 1, 2: 1}` |
| 3 | 2 | 1 | 3 | $3 - 2 = 1$ | Yes (`map[1]=1`)| 2 | `{0: 1, 1: 1, 2: 1, 3: 1}` |

- **Diagnostic Triggers:** "Subarray sum equals K", "Range sum queries with $O(1)$ lookup", "Equal number of 0s and 1s".
- **Boundary Conditions:** Always seed frequency map with `map.put(0, 1)` to handle subarrays starting at index 0.
- **Real-World Application:** Financial ledger balance auditing, cumulative network bandwidth calculation.


## Module 2: Windowing & Pointer Navigation

### [PAT-04] Dynamic Sliding Window (Variable Size)

- **Invariant (Neutral):** Contiguous window $A[L..R]$ satisfies monotonic constraint predicate $V$. Incrementing $R$ expands window state; if $V$ is violated, incrementing $L$ contracts window state until $V$ is restored, evaluating all optimal subsegment candidates in $O(N)$ amortized time.
- **Mental Model:** An adjustable measuring tape expanding to capture maximal elements until a threshold breaks, then tightening from the tail to restore compliance.
- **Concrete Tracing Exemplar:** Longest Substring Without Repeating Characters (e.g., `s = "abcabcbb"`).
- **Visual Architecture / Data-Flow Diagram:**

```
Expand R:   [ a  b  c ] a  b  c  b  b   (Window valid: "abc", len=3)
              L        R
Violation:  [ a  b  c  a ] b  c  b  b   ('a' repeated! Invalid)
              L           R
Shrink L:     a [ b  c  a ] b  c  b  b   (Increment L: "bca", valid again)
                 L        R
```

- **Canonical Code Skeleton:**
```csharp
public int LongestSubarray(int[] nums, int k)
{
    int left = 0, result = 0, zeroCount = 0;

    for (int right = 0; right < nums.Length; right++)
    {
        if (nums[right] == 0) zeroCount++;

        while (zeroCount > k)
        {
            if (nums[left] == 0) zeroCount--;
            left++; // Always advance left during shrink
        }

        result = Math.Max(result, right - left + 1);
    }
    return result;
}
```


- **Step-by-Step State Trace (Input: `s = "abcabcbb"`):**

| Step | `R` | `s[R]` | Window State (Map/Set) | Valid? | Action | `L` | Max Length |
|:---:|:---:|:---:|:---|:---:|:---|:---:|:---:|
| 1 | 0 | `'a'` | `{'a':1}` | Yes | `maxLen = max(0, 0-0+1)` | 0 | 1 |
| 2 | 1 | `'b'` | `{'a':1, 'b':1}` | Yes | `maxLen = max(1, 1-0+1)` | 0 | 2 |
| 3 | 2 | `'c'` | `{'a':1, 'b':1, 'c':1}` | Yes | `maxLen = max(2, 2-0+1)` | 0 | 3 |
| 4 | 3 | `'a'` | `{'a':2, 'b':1, 'c':1}` | No | Shrink $L$ until `'a'` count == 1 | 1 | 3 |
| 5 | 4 | `'b'` | `{'b':2, 'c':1, 'a':1}` | No | Shrink $L$ until `'b'` count == 1 | 2 | 3 |

- **Diagnostic Triggers:** "Longest/shortest contiguous subarray satisfying condition", "At most K distinct elements".
- **Boundary Conditions:** Set-based windows shrink BEFORE expanding; HashMap/Sum-based windows expand FIRST then shrink.
- **Real-World Application:** Sliding-window network rate limiters, memory consumption stream monitoring.

### [PAT-05] Fixed-Size Monotonic Deque Window

- **Invariant (Neutral):** A double-ended queue maintains element indices in strictly monotonic order of their values for a sliding window of fixed width $K$. The front of the deque holds the index of the optimal (maximum/minimum) element for window $[i-K+1..i]$ in $O(N)$ time.
- **Mental Model:** A line of candidates where any newly arriving candidate evicts all older, weaker candidates from the back, while expired candidates fall off the front.
- **Concrete Tracing Exemplar:** Sliding Window Maximum (e.g., `nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3`).
- **Visual Architecture / Data-Flow Diagram:**

```
Window [1, 3, -1]:
Arrival '3' evicts '1' (3 > 1). Arrival '-1' appended.
Deque (Indices): [1, 2]  -> Values: [3, -1]
Front Index 1 (Value 3) is Maximum for Window 0..2.

Window slides to [3, -1, -3]:
Arrival '-3' appended. Deque: [1, 2, 3] -> Values: [3, -1, -3]
Front Index 1 (Value 3) is Maximum.
```

- **Canonical Code Skeleton:**
```csharp
public int[] MaxSlidingWindow(int[] nums, int k)
{
    var deque = new LinkedList<int>();
    var res = new int[nums.Length - k + 1];
    int idx = 0;

    for (int i = 0; i < nums.Length; i++)
    {
        while (deque.Count > 0 && deque.First.Value < i - k + 1) deque.RemoveFirst(); // Expire
        while (deque.Count > 0 && nums[deque.Last.Value] < nums[i]) deque.RemoveLast(); // Kill weaker
        deque.AddLast(i);
        if (i >= k - 1) res[idx++] = nums[deque.First.Value];
    }
    return res;
}
```


- **Step-by-Step State Trace (Input: `nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3`):**

| Step `i` | `nums[i]` | Evict Back (Smaller) | Evict Front (Out of Window) | Deque State (Indices) | Window Full? | Output Max |
|:---:|:---:|:---|:---|:---|:---:|:---:|
| 0 | 1 | None | None | `[0]` (val:1) | No | - |
| 1 | 3 | Evict `0` (1 < 3) | None | `[1]` (val:3) | No | - |
| 2 | -1 | None | None | `[1, 2]` (vals:3,-1) | Yes (`i>=2`)| `nums[1]` = 3 |
| 3 | -3 | None | None | `[1, 2, 3]` | Yes | `nums[1]` = 3 |
| 4 | 5 | Evict `3,2,1` (5>all) | Evict `1` ($1 < 4-3+1$) | `[4]` (val:5) | Yes | `nums[4]` = 5 |

- **Diagnostic Triggers:** "Maximum/minimum element in every sliding window of size K".
- **Boundary Conditions:** Deque MUST store indices to evaluate window expiration (`deque.peekFirst() <= i - k`).
- **Real-World Application:** High-frequency financial tick peak detection, SLA rolling latency maximums.

### [PAT-06] Converging Two-Pointers

- **Invariant (Neutral):** Two pointers starting at opposite boundaries ($L=0, R=N-1$) define a shrinking candidate search interval. Evaluating condition $f(L, R)$ deterministically eliminates either candidate $L$ or candidate $R$, reducing search space in $O(N)$ time.
- **Mental Model:** Hydraulic vise squeezing an interval inward from both boundaries.
- **Concrete Tracing Exemplar:** Container With Most Water (e.g., `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]`).
- **Visual Architecture / Data-Flow Diagram:**

```
Pointers:  L=0 (val:1)                                R=8 (val:7)
Array:    [ 1 ,  8 ,  6 ,  2 ,  5 ,  4 ,  8 ,  3 ,  7 ]
Width:     8, Height: min(1,7)=1 -> Area = 8
Decision:  height[L] < height[R] (1 < 7) -> L moves right (L=1)
```

- **Canonical Code Skeleton:**
```csharp
public int[] TwoSumSorted(int[] nums, int target)
{
    int left = 0, right = nums.Length - 1;
    while (left < right)
    {
        int sum = nums[left] + nums[right];
        if (sum == target) return new int[] { left, right };
        else if (sum < target) left++;
        else right--;
    }
    return new int[0];
}
```


- **Step-by-Step State Trace (Input: `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]`):**

| Step | `L` | `R` | `h[L]` | `h[R]` | Width | Area | Max Area | Action |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 0 | 8 | 1 | 7 | 8 | $8 \times 1 = 8$ | 8 | $h[L] < h[R] \implies L++$ |
| 2 | 1 | 8 | 8 | 7 | 7 | $7 \times 7 = 49$ | 49 | $h[L] \ge h[R] \implies R--$ |
| 3 | 1 | 7 | 8 | 3 | 6 | $6 \times 3 = 18$ | 49 | $h[R] < h[L] \implies R--$ |
| 4 | 1 | 6 | 8 | 8 | 5 | $5 \times 8 = 40$ | 49 | $h[L] \ge h[R] \implies R--$ |

- **Diagnostic Triggers:** "Find pair in sorted array", "Container with most water", "Symmetric string palindrome validation".
- **Boundary Conditions:** Array MUST be sorted for target pair search. Termination condition is `L < R`.
- **Real-World Application:** Order-matching engine pairing, bid-ask spread reconciliation.

### [PAT-07] Fast & Slow Pointers (Floyd's Cycle Detection)

- **Invariant (Neutral):** In a sequence with non-cyclic prefix length $F$ and cycle length $C$, pointers advancing at rates $v$ and $2v$ will meet inside the cycle at step $k \cdot C$. Resetting one pointer to the origin and advancing both at rate $v$ causes them to meet at the cycle entrance after exactly $F$ steps.
- **Mental Model:** Two runners on a track with a non-circular entry path.
- **Concrete Tracing Exemplar:** Linked List Cycle II (Find Cycle Start).
- **Visual Architecture / Data-Flow Diagram:**

```
Head ───► [ 1 ] ───► [ 2 ] (Entrance) ◄───┐
                       │                  │
                      [ 3 ] ───► [ 4 ] ───┘ (Meeting Point)
Non-cyclic Tail (F=1): Node 1 -> Node 2
Cycle (C=3): Nodes 2 -> 3 -> 4 -> 2

Phase 1: Slow & Fast meet at Node 4.
Phase 2: Reset Slow to Head (Node 1). Move both 1 step -> Meet at Node 2 (Entrance).
```

- **Canonical Code Skeleton:**
```csharp
public bool HasCycle(ListNode head)
{
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null)
    {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }
    return false;
}
```


- **Step-by-Step State Trace:**

| Phase | Step | `slow` Node | `fast` Node | Notes / Action |
|:---:|:---:|:---:|:---:|:---|
| 1 | 0 | 1 | 1 | Start |
| 1 | 1 | 2 | 3 | `slow` +1, `fast` +2 |
| 1 | 2 | 3 | 2 | `slow` +1, `fast` +2 |
| 1 | 3 | 4 | 4 | Intersection detected! (Phase 1 Complete) |
| 2 | 0 | 1 (Reset) | 4 | `slow` reset to head |
| 2 | 1 | 2 | 2 | Both move +1 -> Intersect at Node 2 (Cycle Entrance) |

- **Diagnostic Triggers:** "Detect cycle in linked list", "Find duplicate number in array $1..N$", "Happy number cycle detection".
- **Boundary Conditions:** Null-guard `fast != null && fast.next != null`.
- **Real-World Application:** Infinite loop detection in graph workflows, deadlocked transaction cycle recovery.


## Module 3: Stacks, Queues & Monotonic Structures

### [PAT-08] LIFO Matching & Expression Parsing

- **Invariant (Neutral):** A LIFO stack maintains open structural context elements; arriving closing elements must match the top element of the stack, enforcing balanced hierarchical nesting.
- **Mental Model:** Stack of nested plates representing open scopes.
- **Concrete Tracing Exemplar:** Valid Parentheses (e.g., `s = "{[()]}"`).
- **Visual Architecture / Data-Flow Diagram:**

```
Input: "{ [ ( ) ] }"
Char '{': Push '}'  -> Stack: [ '}' ]
Char '[': Push ']'  -> Stack: [ '}', ']' ]
Char '(': Push ')'  -> Stack: [ '}', ']', ')' ]
Char ')': Pop & Match ')' == ')' -> Stack: [ '}', ']' ]
Char ']': Pop & Match ']' == ']' -> Stack: [ '}' ]
Char '}': Pop & Match '}' == '}' -> Stack: [ ] (Valid!)
```

- **Canonical Code Skeleton:**
```csharp
public bool IsValidParentheses(string s)
{
    var stack = new Stack<char>();
    foreach (char c in s)
    {
        if (c == '(') stack.Push(')');
        else if (c == '{') stack.Push('}');
        else if (c == '[') stack.Push(']');
        else if (stack.Count == 0 || stack.Pop() != c) return false;
    }
    return stack.Count == 0;
}
```


- **Step-by-Step State Trace (Input: `s = "{[()]}"`):**

| Step | Char | Stack State (Top at Right) | Action | Result |
|:---:|:---:|:---|:---|:---:|
| 1 | `'{'` | `['}']` | Push expected matching delimiter | Valid |
| 2 | `'['` | `['}', ']']` | Push expected matching delimiter | Valid |
| 3 | `'('` | `['}', ']', ')']` | Push expected matching delimiter | Valid |
| 4 | `')'` | `['}', ']']` | Pop top and check equality | Match (`')' == ')'`) |
| 5 | `']'` | `['}']` | Pop top and check equality | Match (`']' == ']'`) |
| 6 | `'}'` | `[]` | Pop top and check equality | Match (`'}' == '}'`) |

- **Diagnostic Triggers:** "Valid parentheses", "Evaluate arithmetic expression", "Simplify file paths".
- **Boundary Conditions:** Verify `stack.isEmpty()` before popping; stack must be empty upon traversal completion.
- **Real-World Application:** AST compiler parsers, JSON syntax validators, undo/redo buffers.

### [PAT-09] Monotonic Stack ("The Waiting Room")

- **Invariant (Neutral):** A stack maintains element indices in strictly monotonic order of their values. Arriving element $x$ pops all top elements violating monotonicity, resolving the next-greater/smaller relationship for each popped index in $O(N)$ amortized time.
- **Mental Model:** A queue of pending elements waiting for a boundary-breaking value to resolve their state.
- **Concrete Tracing Exemplar:** Next Greater Element / Daily Temperatures (e.g., `temperatures = [73, 74, 75, 71, 69, 72, 76]`).
- **Visual Architecture / Data-Flow Diagram:**

```
Stack holds indices of strictly decreasing values:
Idx 2 (75), Idx 3 (71), Idx 4 (69)  <- Stack top

Arrival of Idx 5 (Val 72):
72 > 69 -> Pop Idx 4. Next greater for Idx 4 is Idx 5 (Dist: 5-4 = 1)
72 > 71 -> Pop Idx 3. Next greater for Idx 3 is Idx 5 (Dist: 5-3 = 2)
72 < 75 -> Stop popping. Push Idx 5.

New Stack: Idx 2 (75), Idx 5 (72)
```

- **Canonical Code Skeleton:**
```csharp
public int[] DailyTemperatures(int[] temps)
{
    var ans = new int[temps.Length];
    var stack = new Stack<int>(); // Stores INDICES

    for (int i = 0; i < temps.Length; i++)
    {
        while (stack.Count > 0 && temps[stack.Peek()] < temps[i])
        {
            int prevIdx = stack.Pop();
            ans[prevIdx] = i - prevIdx;
        }
        stack.Push(i);
    }
    return ans;
}
```


- **Step-by-Step State Trace (Input: `[73, 74, 75, 71, 69, 72, 76]`):**

| Step `i` | Val | Stack (Indices) | Stack (Values) | Popped Indices | Resolved Next Greater Index |
|:---:|:---:|:---|:---|:---|:---|
| 0 | 73 | `[0]` | `[73]` | None | - |
| 1 | 74 | `[1]` | `[74]` | `0` | `ans[0] = 1 - 0 = 1` |
| 2 | 75 | `[2]` | `[75]` | `1` | `ans[1] = 2 - 1 = 1` |
| 3 | 71 | `[2, 3]` | `[75, 71]` | None | - |
| 4 | 69 | `[2, 3, 4]` | `[75, 71, 69]` | None | - |
| 5 | 72 | `[2, 5]` | `[75, 72]` | `4, 3` | `ans[4]=1, ans[3]=2` |
| 6 | 76 | `[6]` | `[76]` | `5, 2` | `ans[5]=1, ans[2]=4` |

- **Diagnostic Triggers:** "Next greater/smaller element", "Daily temperatures", "Largest rectangle in histogram".
- **Boundary Conditions:** Store INDICES on stack, not values. Unresolved elements remain default `-1` or `0`.
- **Real-World Application:** Stock drop notification engines, automated threshold breach monitoring.


## Module 4: Search Space & Decision Trees

### [PAT-10] Monotonic Partition Binary Search

- **Invariant (Neutral):** In a partitioned search space $[L..R]$, at least one half $[L..M]$ or $[M..R]$ preserves strict monotonicity, allowing deterministic boundary verification and half-space elimination in $O(\log N)$ time.
- **Mental Model:** Testing which side of a fractured slope is contiguous to eliminate the other side.
- **Concrete Tracing Exemplar:** Search in Rotated Sorted Array (e.g., `nums = [4, 5, 6, 7, 0, 1, 2], target = 0`).
- **Visual Architecture / Data-Flow Diagram:**

```
Array: [ 4 , 5 , 6 , 7 , 0 , 1 , 2 ]
         L           M           R
Left Half [4..7] is Strictly Sorted (nums[L] <= nums[M]: 4 <= 7).
Target 0 lies OUTSIDE left half [4..7] -> Eliminate Left Half!
Set L = M + 1 (L = 4, searching [0, 1, 2]).
```

- **Canonical Code Skeleton:**
```csharp
public int SearchRotated(int[] nums, int target)
{
    int left = 0, right = nums.Length - 1;
    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;

        if (nums[left] <= nums[mid]) // Left half sorted (MUST use <=)
        {
            if (nums[left] <= target && target < nums[mid]) right = mid - 1;
            else left = mid + 1;
        }
        else // Right half sorted
        {
            if (nums[mid] < target && target <= nums[right]) left = mid + 1;
            else right = mid - 1;
        }
    }
    return -1;
}
```


- **Step-by-Step State Trace (Input: `nums = [4,5,6,7,0,1,2], target = 0`):**

| Step | `L` | `R` | `M` | `nums[M]` | Sorted Half? | Target in Sorted Range? | Next Action |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| 1 | 0 | 6 | 3 | 7 | Left (`4 <= 7`) | $0 \notin [4, 7]$ | $L = M + 1 = 4$ |
| 2 | 4 | 6 | 5 | 1 | Left (`0 <= 1`) | $0 \in [0, 1]$ | $R = M - 1 = 4$ |
| 3 | 4 | 4 | 4 | 0 | Target Found! | Yes | Return Index 4 |

- **Diagnostic Triggers:** "Search in rotated sorted array", "Find pivot in shifted monotonic sequence".
- **Boundary Conditions:** Use `nums[L] <= nums[M]` (with `<=`) to handle 1-element partitions correctly.
- **Real-World Application:** Distributed log partition lookups, sharded database range routing.

### [PAT-11] Binary Search on Solution Range

- **Invariant (Neutral):** A predicate decision function $P(x) \in \{\text{false}, \text{true}\}$ is monotonic over bounded integer interval $[lo..hi]$. Binary search identifies the minimal $x$ where $P(x) = \text{true}$ in $O(\log(hi - lo) \cdot \text{Cost}(P))$ time.
- **Mental Model:** Flipping a monotonic multi-switch to find the exact threshold point where state changes from False to True.
- **Concrete Tracing Exemplar:** Capacity To Ship Packages Within D Days (e.g., `weights = [1,2,3,4,5,6,7,8,9,10], D = 5`).
- **Visual Architecture / Data-Flow Diagram:**

```
Capacity Space:  [ 10 ... 14  |  15 ... 55 ]
Predicate P(x):  [ F  ... F   |   T ...  T ]
                              ▲
                       Minimal Capacity = 15
```

- **Canonical Code Skeleton:**
```csharp
public int ShipWithinDays(int[] weights, int days)
{
    int lo = 0, hi = 0;
    foreach (int w in weights)
    {
        lo = Math.Max(lo, w);
        hi += w;
    }

    while (lo < hi)
    {
        int mid = lo + (hi - lo) / 2;
        if (CanShip(weights, days, mid)) hi = mid; // Try smaller capacity
        else lo = mid + 1;                         // Must increase capacity
    }
    return lo;
}

private bool CanShip(int[] weights, int days, int capacity)
{
    int dayCount = 1, currentLoad = 0;
    foreach (int w in weights)
    {
        if (currentLoad + w > capacity)
        {
            dayCount++;
            currentLoad = 0;
        }
        currentLoad += w;
    }
    return dayCount <= days;
}
```


- **Step-by-Step State Trace (Input: `weights=[1..10], D=5`, Range: `[10..55]`):**

| Step | `lo` | `hi` | `mid` | $P(\text{mid})$ (Days Needed $\le 5$) | Action |
|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 10 | 55 | 32 | True (Needs 2 Days $\le 5$) | `hi = 32` |
| 2 | 10 | 32 | 21 | True (Needs 3 Days $\le 5$) | `hi = 21` |
| 3 | 10 | 21 | 15 | True (Needs 5 Days $\le 5$) | `hi = 15` |
| 4 | 10 | 15 | 12 | False (Needs 6 Days $> 5$) | `lo = 13` |
| 5 | 13 | 15 | 14 | False (Needs 6 Days $> 5$) | `lo = 15` |
| End | 15 | 15 | - | Terminate: `lo == hi` | Minimum Capacity = 15 |

- **Diagnostic Triggers:** "Find minimum capacity/speed satisfying constraint", "Koko eating bananas".
- **Boundary Conditions:** Define correct initial bounds (`lo = max(weights)`, `hi = sum(weights)`).
- **Real-World Application:** Cloud resource scaling optimization, thread pool sizing limit search.

### [PAT-12] Backtracking & State-Space Pruning

- **Invariant (Neutral):** Explores an implicit state-space tree depth-first. State mutation $S' = S \cup \{c\}$ is applied before entering a child branch and strictly reverted $S = S' \setminus \{c\}$ upon returning, evaluating all valid configuration paths while pruning invalid branches.
- **Mental Model:** Walking a decision tree while unrolling state changes upon hitting dead ends.
- **Concrete Tracing Exemplar:** Generate All Permutations (e.g., `nums = [1, 2]`).
- **Visual Architecture / Data-Flow Diagram:**

```
                     []
            ┌────────┴────────┐
           [1]               [2]
            │                 │
          [1,2]             [2,1]
         (Backtrack)       (Backtrack)
```

- **Canonical Code Skeleton:**
```csharp
public void Backtrack(List<IList<int>> res, List<int> path, int[] nums, bool[] used)
{
    if (path.Count == nums.Length)
    {
        res.Add(new List<int>(path));
        return;
    }
    for (int i = 0; i < nums.Length; i++)
    {
        if (used[i]) continue;
        used[i] = true;
        path.Add(nums[i]);
        Backtrack(res, path, nums, used); // Recurse
        path.RemoveAt(path.Count - 1);    // Undo (backtrack)
        used[i] = false;
    }
}
```


- **Step-by-Step State Trace (Input: `nums = [1, 2]`):**

| Step | Depth | Active Path | Choice | Constraint Met? | Action | Output List |
|:---:|:---:|:---|:---:|:---:|:---|:---|
| 1 | 0 | `[]` | 1 | Yes | Add 1 -> Recurse | `[]` |
| 2 | 1 | `[1]` | 2 | Yes | Add 2 -> Recurse | `[]` |
| 3 | 2 | `[1, 2]` | Base | Full | Deep Copy Path | `[[1, 2]]` |
| 4 | 1 | `[1]` | Backtrack| Undo 2 | Remove 2 | `[[1, 2]]` |
| 5 | 0 | `[]` | Backtrack| Undo 1 | Remove 1 | `[[1, 2]]` |
| 6 | 1 | `[2]` | 1 | Yes | Add 1 -> Recurse | `[[1, 2]]` |
| 7 | 2 | `[2, 1]` | Base | Full | Deep Copy Path | `[[1, 2], [2, 1]]` |

- **Diagnostic Triggers:** "Generate all permutations/combinations/subsets", "Sudoku solver".
- **Boundary Conditions:** Always store a deep copy (`new ArrayList<>(path)`) when appending to result list.
- **Real-World Application:** Security path authorization traversal, automated constraint solving.


## Module 5: Graph & Grid Traversals

### [PAT-13] Level-by-Level BFS Wavefront

- **Invariant (Neutral):** A FIFO queue maintains nodes at uniform distance $d$ from origin. Processing all snapshot elements of level $d$ before enqueuing level $d+1$ guarantees the first arrival at target is an unweighted shortest path in $O(V+E)$ time.
- **Mental Model:** Concentric water ripples expanding outward 1 unit per timestep.
- **Concrete Tracing Exemplar:** Shortest Path in Unweighted Grid (e.g., $3 \times 3$ grid).
- **Visual Architecture / Data-Flow Diagram:**

```
Level 0: (0,0)
Level 1: (0,1), (1,0)
Level 2: (0,2), (1,1), (2,0)
Level 3: (1,2), (2,1)
Level 4: (2,2) [Target Reached!]
```

- **Canonical Code Skeleton:**
```csharp
public int ShortestPath(char[][] grid, int startR, int startC)
{
    int rows = grid.Length, cols = grid[0].Length;
    var queue = new Queue<int[]>();
    bool[][] visited = new bool[rows][];
    for (int i = 0; i < rows; i++) visited[i] = new bool[cols];

    queue.Enqueue(new int[] { startR, startC });
    visited[startR][startC] = true; // Mark visited ON PUSH
    int steps = 0;
    int[][] DIRS = new int[][] {
        new int[] { 1, 0 }, new int[] { -1, 0 },
        new int[] { 0, 1 }, new int[] { 0, -1 }
    };

    while (queue.Count > 0)
    {
        int size = queue.Count;
        for (int i = 0; i < size; i++)
        {
            int[] curr = queue.Dequeue();
            if (grid[curr[0]][curr[1]] == 'E') return steps;

            foreach (int[] d in DIRS)
            {
                int nr = curr[0] + d[0], nc = curr[1] + d[1];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols
                    && !visited[nr][nc] && grid[nr][nc] != 'X')
                {
                    visited[nr][nc] = true; // MARK ON PUSH!
                    queue.Enqueue(new int[] { nr, nc });
                }
            }
        }
        steps++;
    }
    return -1;
}
```


- **Step-by-Step State Trace:**

| Level `d` | Queue Snapshot at Start | Nodes Processed | Enqueued Next Level | Visited Set Updates |
|:---:|:---|:---|:---|:---|
| 0 | `[(0,0)]` | `(0,0)` | `(0,1), (1,0)` | `{(0,0), (0,1), (1,0)}` |
| 1 | `[(0,1), (1,0)]` | `(0,1), (1,0)` | `(0,2), (1,1), (2,0)` | `+(0,2),(1,1)` |
| 2 | `[(0,2), (1,1), (2,0)]` | `(0,2), (1,1), (2,0)` | `(1,2), (2,1)` | `+{(1,2),(2,1)}` |
| 3 | `[(1,2), (2,1)]` | `(1,2)` | `(2,2)` [Target!] | Return Distance = 4 |

- **Diagnostic Triggers:** "Shortest path in unweighted graph/grid", "Minimum steps to reach goal".
- **Boundary Conditions:** ALWAYS mark nodes visited *upon enqueue*, NOT upon dequeue.
- **Real-World Application:** Social network degree-of-separation lookup, network packet broadcast routing.

### [PAT-14] Multi-Source BFS Parallel Spreading

- **Invariant (Neutral):** Initializing a FIFO queue with all $K$ origin sources at time $t=0$ executes parallel BFS traversal wavefronts, computing minimum distance from *any* source to all reachable vertices in $O(V+E)$ time.
- **Mental Model:** Multiple simultaneous drop points spreading ripples across a surface.
- **Concrete Tracing Exemplar:** Rotting Oranges / Multi-Source Spreading.
- **Visual Architecture / Data-Flow Diagram:**

```
t=0:  [ S1 ]  .   .   [ S2 ]
t=1:   S1   [1]  [1]   S2
t=2:   S1    1    2    S2
```

- **Canonical Code Skeleton:**
```csharp
public int OrangesRotting(int[][] grid)
{
    int rows = grid.Length, cols = grid[0].Length;
    var queue = new Queue<int[]>();
    int freshCount = 0;

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if (grid[r][c] == 2) queue.Enqueue(new int[] { r, c }); // Push ALL sources
            else if (grid[r][c] == 1) freshCount++;
        }
    }
    if (freshCount == 0) return 0;
    
    int minutes = 0;
    int[][] DIRS = new int[][] {
        new int[] { 1, 0 }, new int[] { -1, 0 },
        new int[] { 0, 1 }, new int[] { 0, -1 }
    };

    while (queue.Count > 0 && freshCount > 0)
    {
        int size = queue.Count;
        minutes++;
        for (int i = 0; i < size; i++)
        {
            int[] curr = queue.Dequeue();
            foreach (int[] d in DIRS)
            {
                int nr = curr[0] + d[0], nc = curr[1] + d[1];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1)
                {
                    grid[nr][nc] = 2; // Mutate grid as visited
                    freshCount--;
                    queue.Enqueue(new int[] { nr, nc });
                }
            }
        }
    }
    return freshCount == 0 ? minutes : -1;
}
```


- **Step-by-Step State Trace:**

| Time `t` | Queue State (Level Snapshot) | Fresh Target Count | Action |
|:---:|:---|:---:|:---|
| 0 | `[S1(0,0), S2(0,3)]` | 4 | Pop sources, enqueue adjacent targets at $t=1$ |
| 1 | `[(0,1), (0,2)]` | 2 | Fresh target count drops to 2 |
| 2 | `[(1,1)]` | 0 | Fresh targets empty -> Return Elapsed Time $t=2$ |

- **Diagnostic Triggers:** "Rotting oranges", "Distance to nearest 0 in matrix", "Multi-point outbreak propagation".
- **Boundary Conditions:** Track target count upfront to avoid extraneous time increments.
- **Real-World Application:** Multi-datacenter cache invalidation, multi-source resource allocation.

### [PAT-15] DFS Component Sinking & Flood Fill

- **Invariant (Neutral):** Recursive depth-first traversal visits all connected component vertices. In-place state mutation marks visited vertices, isolating distinct components without extra memory overhead.
- **Mental Model:** Consuming a connected landmass while walking over it so it is never revisited.
- **Concrete Tracing Exemplar:** Number of Islands.
- **Visual Architecture / Data-Flow Diagram:**

```
Grid Scan Finds '1' at (0,0) -> Increments Island Count to 1.
Sink Component via DFS:
(0,0) '1' -> '0'
  ├──> (0,1) '1' -> '0'
  └──> (1,0) '1' -> '0'
Component fully submerged. Grid scan continues.
```

- **Canonical Code Skeleton:**
```csharp
public int NumIslands(char[][] grid)
{
    int count = 0;
    for (int r = 0; r < grid.Length; r++)
    {
        for (int c = 0; c < grid[0].Length; c++)
        {
            if (grid[r][c] == '1')
            {
                count++;
                DfsSink(grid, r, c);
            }
        }
    }
    return count;
}

private void DfsSink(char[][] grid, int r, int c)
{
    if (r < 0 || r >= grid.Length || c < 0 || c >= grid[0].Length || grid[r][c] == '0') return;
    grid[r][c] = '0'; // Sink cell
    DfsSink(grid, r + 1, c);
    DfsSink(grid, r - 1, c);
    DfsSink(grid, r, c + 1);
    DfsSink(grid, r, c - 1);
}
```


- **Step-by-Step State Trace:**

| Cell `(r,c)` | Value | Scan Action | DFS Recursive Action | Island Count |
|:---:|:---:|:---|:---|:---:|
| (0,0) | `'1'` | Trigger DFS | Mutate `grid[0][0]='0'`, Recurse Neighbors | 1 |
| (0,1) | `'1'` | Inside DFS | Mutate `grid[0][1]='0'`, Recurse Neighbors | 1 |
| (1,0) | `'1'` | Inside DFS | Mutate `grid[1][0]='0'`, Recurse Neighbors | 1 |
| (0,2) | `'0'` | Skip | None | 1 |

- **Diagnostic Triggers:** "Number of islands", "Flood fill region", "Surrounded regions".
- **Boundary Conditions:** Base case must validate row/col boundary limits *before* cell value lookup.
- **Real-World Application:** Image segmentation, GIS terrain landmass classification.

### [PAT-16] Topological Sort (Kahn's & DFS)

- **Invariant (Neutral):** In a Directed Acyclic Graph (DAG), vertices with in-degree 0 have zero pending dependencies. Processing in-degree 0 nodes and decrementing neighbor in-degrees constructs a valid linear ordering; if total processed vertices $< V$, a cycle exists.
- **Mental Model:** Task resolution queue where tasks become unblocked as their prerequisites complete.
- **Concrete Tracing Exemplar:** Course Schedule II (Task Scheduling).
- **Visual Architecture / Data-Flow Diagram:**

```
DAG Edges: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
In-Degree Array: [0: 0, 1: 1, 2: 1, 3: 2]

1. Queue: [0] -> Process 0 -> Decr 1 & 2 -> In-Degrees: [1:0, 2:0, 3:2]
2. Queue: [1, 2] -> Process 1 & 2 -> Decr 3 twice -> In-Degree 3: 0
3. Queue: [3] -> Process 3
Order: [0, 1, 2, 3]
```

- **Canonical Code Skeleton:**
```csharp
public int[] FindOrder(int numCourses, int[][] prerequisites)
{
    var inDegree = new int[numCourses];
    var adj = new List<List<int>>();
    for (int i = 0; i < numCourses; i++) adj.Add(new List<int>());
    foreach (int[] p in prerequisites)
    {
        adj[p[1]].Add(p[0]);
        inDegree[p[0]]++;
    }

    var queue = new Queue<int>();
    for (int i = 0; i < numCourses; i++) if (inDegree[i] == 0) queue.Enqueue(i);

    int[] order = new int[numCourses];
    int idx = 0;
    while (queue.Count > 0)
    {
        int curr = queue.Dequeue();
        order[idx++] = curr;
        foreach (int neighbor in adj[curr])
        {
            if (--inDegree[neighbor] == 0) queue.Enqueue(neighbor);
        }
    }
    return idx == numCourses ? order : new int[0];
}
```


- **Step-by-Step State Trace:**

| Step | Queue (In-Degree 0) | Node Processed | Neighbors Decremented | Neighbor In-Degrees | Output List |
|:---:|:---|:---:|:---|:---|:---|
| 1 | `[0]` | 0 | 1, 2 | `inDegree[1]=0, inDegree[2]=0` | `[0]` |
| 2 | `[1, 2]` | 1 | 3 | `inDegree[3]=1` | `[0, 1]` |
| 3 | `[2]` | 2 | 3 | `inDegree[3]=0` | `[0, 1, 2]` |
| 4 | `[3]` | 3 | None | - | `[0, 1, 2, 3]` |

- **Diagnostic Triggers:** "Course schedule", "Task dependency ordering", "Build compilation sequence".
- **Boundary Conditions:** If output list length $< V$, return empty array (cycle detected).
- **Real-World Application:** Build dependency resolution (Maven/Gradle), CI/CD pipeline stage ordering.

### [PAT-17] Disjoint Set Union (Union-Find)

- **Invariant (Neutral):** Manages a partition of $N$ elements into disjoint equivalence sets. Path compression flattens tree depth during `find`, achieving near $O(1)$ amortized ($O(\alpha(N))$) operations for set union and connectivity queries.
- **Mental Model:** Forest of trees where elements point to canonical root set leaders.
- **Concrete Tracing Exemplar:** Number of Connected Components in Undirected Graph.
- **Visual Architecture / Data-Flow Diagram:**

```
Before Path Compression:          After Path Compression find(4):
         1                                      1
        /                                     / | \
       2                                     2  3  4
      /
     3
    /
   4
```

- **Canonical Code Skeleton:**
```csharp
public class UnionFind
{
    private int[] parent;
    private int[] rank;

    public UnionFind(int n)
    {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    public int Find(int i)
    {
        if (parent[i] == i) return i;
        return parent[i] = Find(parent[i]); // Path compression
    }

    public bool Union(int i, int j)
    {
        int rootI = Find(i), rootJ = Find(j);
        if (rootI != rootJ)
        {
            if (rank[rootI] < rank[rootJ]) parent[rootI] = rootJ;
            else if (rank[rootI] > rank[rootJ]) parent[rootJ] = rootI;
            else { parent[rootJ] = rootI; rank[rootI]++; }
            return true;
        }
        return false; // Already connected!
    }
}
```


- **Step-by-Step State Trace:**

| Op | Union Pair | Root X | Root Y | Action | Parent Array State | Component Count |
|:---:|:---:|:---:|:---:|:---|:---|:---:|
| Init | - | - | - | Init `parent[i] = i` | `[0, 1, 2, 3]` | 4 |
| 1 | `(0, 1)` | 0 | 1 | `parent[1] = 0` | `[0, 0, 2, 3]` | 3 |
| 2 | `(2, 3)` | 2 | 3 | `parent[3] = 2` | `[0, 0, 2, 2]` | 2 |
| 3 | `(1, 3)` | 0 | 2 | `parent[2] = 0` | `[0, 0, 0, 2]` | 1 |

- **Diagnostic Triggers:** "Redundant connection", "Number of connected components", "Dynamic connectivity".
- **Boundary Conditions:** Path compression `parent[i] = find(parent[i])` is essential for linearithmic performance.
- **Real-World Application:** Network topology clustering, distributed consensus group membership.

### [PAT-18] Weighted Shortest Path (Dijkstra / Min-Heap)

- **Invariant (Neutral):** For non-negative edge weights $w(u,v) \ge 0$, greedily extracting the unvisited vertex $u$ with minimum tentative distance $d[u]$ guarantees $d[u]$ is optimal, relaxing neighbor distances in $O((V+E) \log V)$ time.
- **Mental Model:** Expanding shortest path frontiers ordered by accumulated cost.
- **Concrete Tracing Exemplar:** Network Delay Time.
- **Visual Architecture / Data-Flow Diagram:**

```
Min-Heap: [(Dist:0, Node:1)]
Pop (0, Node 1) -> Finalize Dist[1]=0.
Relax Neighbors:
  Edge 1->2 (w=1): Dist[2] = 0+1 = 1 -> Push (1, Node 2)
  Edge 1->3 (w=4): Dist[3] = 0+4 = 4 -> Push (4, Node 3)
Min-Heap: [(1, Node 2), (4, Node 3)]
```

- **Canonical Code Skeleton:**
```csharp
public int NetworkDelayTime(int[][] times, int n, int k)
{
    var adj = new Dictionary<int, List<int[]>>();
    foreach (int[] t in times)
    {
        if (!adj.ContainsKey(t[0])) adj[t[0]] = new List<int[]>();
        adj[t[0]].Add(new int[] { t[1], t[2] });
    }

    var pq = new PriorityQueue<int, int>(); // [node, dist] ordered by dist
    pq.Enqueue(k, 0);
    var dist = new Dictionary<int, int>();

    while (pq.Count > 0)
    {
        pq.TryDequeue(out int node, out int d);
        
        if (dist.ContainsKey(node)) continue;
        dist[node] = d;

        if (adj.ContainsKey(node))
        {
            foreach (int[] edge in adj[node])
            {
                if (!dist.ContainsKey(edge[0]))
                {
                    pq.Enqueue(edge[0], d + edge[1]);
                }
            }
        }
    }
    return dist.Count == n ? dist.Values.Max() : -1;
}
```


- **Step-by-Step State Trace:**

| Step | Min-Heap State | Popped Node `u` | Popped Dist `d` | Skip? (`d > dist[u]`) | Relax Neighbor `v` | Dist Array State |
|:---:|:---|:---:|:---:|:---:|:---|:---|
| Init | `[(0, 1)]` | - | - | - | - | `[1:0, 2:∞, 3:∞]` |
| 1 | `[(0, 1)]` | 1 | 0 | No | `dist[2]=1, dist[3]=4` | `[1:0, 2:1, 3:4]` |
| 2 | `[(1, 2), (4, 3)]`| 2 | 1 | No | Edge 2->3 (w=1): `dist[3]=2`| `[1:0, 2:1, 3:2]` |
| 3 | `[(2, 3), (4, 3)]`| 3 | 2 | No | None | `[1:0, 2:1, 3:2]` |
| 4 | `[(4, 3)]` | 3 | 4 | Yes (`4 > 2`) | Skip stale heap entry | `[1:0, 2:1, 3:2]` |

- **Diagnostic Triggers:** "Network delay time", "Cheapest path with non-negative edge weights".
- **Boundary Conditions:** Must include stale node check `if (d > dist[u]) continue` to ignore outdated heap entries.
- **Real-World Application:** Latency-based API routing engines, map routing algorithms.


## Module 6: Dynamic Programming & Optimization

### [PAT-19] 1D Choice Optimization (O(1) Space DP)

- **Invariant (Neutral):** Optimal state $DP[i]$ depends only on a bounded history horizon $\{DP[i-1], \dots, DP[i-k]\}$. Maintaining rolling scalar variables reduces space complexity from $O(N)$ to $O(k)$ while preserving $O(N)$ time.
- **Mental Model:** A sliding window of state memory variables propagating optimal choices forward.
- **Concrete Tracing Exemplar:** House Robber (e.g., `nums = [2, 7, 9, 3, 1]`).
- **Visual Architecture / Data-Flow Diagram:**

```
State Recurrence: DP[i] = max(DP[i-1], DP[i-2] + nums[i])
Variable Rolling:
prev2  prev1  ->  curr  (New prev2 = old prev1, New prev1 = curr)
```

- **Canonical Code Skeleton:**
```csharp
public int Rob(int[] nums)
{
    if (nums == null || nums.Length == 0) return 0;
    int prev2 = 0, prev1 = 0;

    foreach (int num in nums)
    {
        int curr = Math.Max(prev1, prev2 + num); // Skip vs Take
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```


- **Step-by-Step State Trace (Input: `nums = [2, 7, 9, 3, 1]`):**

| Step `i` | `nums[i]` | Choice 1 (`prev1`) | Choice 2 (`prev2 + nums[i]`) | `curr` | `prev2` Next | `prev1` Next |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Init | - | - | - | - | 0 | 0 |
| 0 | 2 | 0 | $0 + 2 = 2$ | 2 | 0 | 2 |
| 1 | 7 | 2 | $0 + 7 = 7$ | 7 | 2 | 7 |
| 2 | 9 | 7 | $2 + 9 = 11$ | 11 | 7 | 11 |
| 3 | 3 | 11 | $7 + 3 = 10$ | 11 | 11 | 11 |
| 4 | 1 | 11 | $11 + 1 = 12$| 12 | 11 | 12 |

- **Diagnostic Triggers:** "House robber", "Climbing stairs", "Min cost climbing stairs".
- **Boundary Conditions:** Handle single-element input upfront.
- **Real-World Application:** Capacity allocation, CPU time-slot scheduling.

### [PAT-20] 0/1 & Unbounded Knapsack DP

- **Invariant (Neutral):** State $DP[w]$ tracks optimal score for resource capacity $w$. Iterating capacity backward ($W..w$) ensures each item is used at most once (0/1), whereas iterating forward ($w..W$) allows unbounded item reuse.
- **Mental Model:** A capacity table updated by integrating discrete resource choices.
- **Concrete Tracing Exemplar:** Coin Change (Unbounded) (e.g., `coins = [1, 2, 5], amount = 11`).
- **Visual Architecture / Data-Flow Diagram:**

```
0/1 Knapsack (Backward Iteration):
Capacity:  W ◄───────── w   (Prevents overwriting DP state used in same pass)

Unbounded Knapsack (Forward Iteration):
Capacity:  w ─────────► W   (Allows current pass updates to chain reuse)
```

- **Canonical Code Skeleton:**
```csharp
public int CoinChange(int[] coins, int amount)
{
    int[] dp = new int[amount + 1];
    Array.Fill(dp, amount + 1);
    dp[0] = 0;

    for (int i = 1; i <= amount; i++)
    {
        foreach (int coin in coins)
        {
            if (i - coin >= 0)
            {
                dp[i] = Math.Min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
```


- **Step-by-Step State Trace (Input: `coins = [1, 2, 5], amount = 5`):**

| Coin | Capacity `w` | Forward Update Equation: $DP[w] = \min(DP[w], DP[w - \text{coin}] + 1)$ | Array State $DP[0..5]$ |
|:---:|:---:|:---|:---|
| Init | - | Seed $DP[0]=0$, all others $\infty$ | `[0, ∞, ∞, ∞, ∞, ∞]` |
| 1 | 1..5 | $DP[1]=1, DP[2]=2, DP[3]=3, DP[4]=4, DP[5]=5$ | `[0, 1, 2, 3, 4, 5]` |
| 2 | 2..5 | $DP[2]=\min(2, 0+1)=1, DP[3]=2, DP[4]=2, DP[5]=3$ | `[0, 1, 1, 2, 2, 3]` |
| 5 | 5 | $DP[5]=\min(3, DP[0]+1)=1$ | `[0, 1, 1, 2, 2, 1]` |

- **Diagnostic Triggers:** "Coin change", "Partition equal subset sum", "Knapsack capacity".
- **Boundary Conditions:** Fill array with sentinel value (`amount + 1`) representing infinity.
- **Real-World Application:** Resource packing in cloud instances, currency change calculators.

### [PAT-21] 2D Grid Path Optimization

- **Invariant (Neutral):** State $DP[r][c]$ holds optimal path value to grid cell $(r,c)$, derived from valid predecessor states $\min/\max(DP[r-1][c], DP[r][c-1])$. Grid structure provides topological evaluation order in $O(R \cdot C)$ time.
- **Mental Model:** Accumulating optimal path costs along a grid matrix.
- **Concrete Tracing Exemplar:** Minimum Path Sum (e.g., $3 \times 3$ grid `[[1,3,1],[1,5,1],[4,2,1]]`).
- **Visual Architecture / Data-Flow Diagram:**

```
Grid:
[ 1 , 3 , 1 ]
[ 1 , 5 , 1 ]
[ 4 , 2 , 1 ]

DP Table:
[ 1 , 4 , 5 ]
[ 2 , 7 , 6 ]
[ 6 , 8 , 7 ]  <- Minimum Path Sum = 7
```

- **Canonical Code Skeleton:**
```csharp
public int MinPathSum(int[][] grid)
{
    int rows = grid.Length, cols = grid[0].Length;
    int[][] dp = new int[rows][];
    for (int i = 0; i < rows; i++) dp[i] = new int[cols];

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if (r == 0 && c == 0) dp[r][c] = grid[r][c];
            else if (r == 0) dp[r][c] = dp[r][c - 1] + grid[r][c];
            else if (c == 0) dp[r][c] = dp[r - 1][c] + grid[r][c];
            else dp[r][c] = Math.Min(dp[r - 1][c], dp[r][c - 1]) + grid[r][c];
        }
    }
    return dp[rows - 1][cols - 1];
}
```


- **Step-by-Step State Trace:**

| Cell `(r,c)` | Grid Value | Predecessor Min (`top`, `left`) | $DP[r][c]$ Calculation |
|:---:|:---:|:---|:---|
| (0,0) | 1 | Base | 1 |
| (0,1) | 3 | Left: 1 | $1 + 3 = 4$ |
| (0,2) | 1 | Left: 4 | $4 + 1 = 5$ |
| (1,0) | 1 | Top: 1 | $1 + 1 = 2$ |
| (1,1) | 5 | Top: 4, Left: 2 -> Min: 2 | $2 + 5 = 7$ |
| (2,2) | 1 | Top: 6, Left: 8 -> Min: 6 | $6 + 1 = 7$ |

- **Diagnostic Triggers:** "Minimum path sum", "Unique paths in grid", "Dungeon game".
- **Boundary Conditions:** Initialize first row and first column carefully.
- **Real-World Application:** Cost-effective data routing across grid-structured networks.

### [PAT-22] String Alignment & Sequence DP

- **Invariant (Neutral):** Entry $DP[i][j]$ holds optimal alignment metric for prefixes $S_1[0..i-1]$ and $S_2[0..j-1]$. Character match $S_1[i-1] == S_2[j-1]$ transitions diagonally ($DP[i-1][j-1] + 1$), while mismatch branches on insertion/deletion transitions.
- **Mental Model:** Grid comparison matching two strings character-by-character.
- **Concrete Tracing Exemplar:** Longest Common Subsequence (e.g., `s1 = "abcde", s2 = "ace"`).
- **Visual Architecture / Data-Flow Diagram:**

```
       Ø   a   c   e
   Ø [ 0 , 0 , 0 , 0 ]
   a [ 0 , 1 , 1 , 1 ]  (Match 'a' -> Diagonal + 1)
   b [ 0 , 1 , 1 , 1 ]
   c [ 0 , 1 , 2 , 2 ]  (Match 'c' -> Diagonal + 1)
   d [ 0 , 1 , 2 , 2 ]
   e [ 0 , 1 , 2 , 3 ]  (Match 'e' -> Diagonal + 1)
```

- **Canonical Code Skeleton:**
```csharp
public int LongestCommonSubsequence(string text1, string text2)
{
    int m = text1.Length, n = text2.Length;
    int[][] dp = new int[m + 1][];
    for (int i = 0; i <= m; i++) dp[i] = new int[n + 1];

    for (int i = 1; i <= m; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            if (text1[i - 1] == text2[j - 1])
            {
                dp[i][j] = 1 + dp[i - 1][j - 1];
            }
            else
            {
                dp[i][j] = Math.Max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}
```


- **Step-by-Step State Trace:**

| `i` (`s1`) | `j` (`s2`) | `s1[i-1]` | `s2[j-1]` | Match? | Transition Equation | $DP[i][j]$ |
|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| 1 | 1 | `'a'` | `'a'` | Yes | $DP[0][0] + 1 = 0 + 1$ | 1 |
| 1 | 2 | `'a'` | `'c'` | No | $\max(DP[0][2], DP[1][1]) = \max(0, 1)$ | 1 |
| 3 | 2 | `'c'` | `'c'` | Yes | $DP[2][1] + 1 = 1 + 1$ | 2 |
| 5 | 3 | `'e'` | `'e'` | Yes | $DP[4][2] + 1 = 2 + 1$ | 3 |

- **Diagnostic Triggers:** "Longest common subsequence", "Edit distance", "Wildcard matching".
- **Boundary Conditions:** Matrix dimensions are `(m + 1) x (n + 1)`. Access chars using `i - 1` and `j - 1`.
- **Real-World Application:** Git diff algorithms, DNA sequence alignment, text similarity search.

### [PAT-23] Sweep-Line & Interval Scheduling

- **Invariant (Neutral):** Sorting $N$ intervals by start coordinate transforms 2D temporal overlap detection into 1D sequential scan, maintaining active boundary state in $O(N \log N)$ time.
- **Mental Model:** A vertical timeline sweeping left-to-right across event intervals.
- **Concrete Tracing Exemplar:** Meeting Rooms II (e.g., `intervals = [[0,30],[5,10],[15,20]]`).
- **Visual Architecture / Data-Flow Diagram:**

```
Timeline:  0 .... 5 .... 10 .... 15 .... 20 .... 30
Mtg 1:    [========================================] (0..30)
Mtg 2:           [========]                         (5..10)
Mtg 3:                           [========]         (15..20)

Min-Heap of End Times:
At t=0:  Push 30 -> Heap: [30] (1 room)
At t=5:  5 < 30 (Overlap!) -> Push 10 -> Heap: [10, 30] (2 rooms)
At t=15: 15 >= 10 (Room freed!) -> Pop 10, Push 20 -> Heap: [20, 30] (2 rooms)
```

- **Canonical Code Skeleton:**
```csharp
public int MinMeetingRooms(int[][] intervals)
{
    if (intervals == null || intervals.Length == 0) return 0;
    Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));

    var minHeap = new PriorityQueue<int, int>(); // Stores end times
    minHeap.Enqueue(intervals[0][1], intervals[0][1]);

    for (int i = 1; i < intervals.Length; i++)
    {
        if (intervals[i][0] >= minHeap.Peek())
        {
            minHeap.Dequeue(); // Room freed up!
        }
        minHeap.Enqueue(intervals[i][1], intervals[i][1]); // Allocate room
    }
    return minHeap.Count;
}
```


- **Step-by-Step State Trace:**

| Step | Interval `[start, end]` | Heap Top (Earliest End) | Overlap Condition (`start < top`) | Min-Heap State | Rooms Needed |
|:---:|:---:|:---:|:---:|:---|:---:|
| Init | - | - | - | `[]` | 0 |
| 1 | `[0, 30]` | None | False | `[30]` | 1 |
| 2 | `[5, 10]` | 30 | True ($5 < 30$) | `[10, 30]` | 2 |
| 3 | `[15, 20]`| 10 | False ($15 \ge 10$) | `[20, 30]` (Pop 10, Push 20) | 2 |

- **Diagnostic Triggers:** "Meeting rooms II", "Merge intervals", "Non-overlapping intervals".
- **Boundary Conditions:** Always sort intervals by start time `a[0] - b[0]` first.
- **Real-World Application:** Calendar scheduling engines, hotel room allocation, cloud VM provisioning.

### [PAT-24] Trie Prefix Search & Retrieval

- **Invariant (Neutral):** A tree structure where each node represents a character. Root-to-node path forms a string prefix, enabling $O(L)$ word search and prefix retrieval independent of dictionary size $N$, where $L$ is word length.
- **Mental Model:** A character decision tree branching at each character of a dictionary.
- **Concrete Tracing Exemplar:** Implement Trie (Insert "apple", Search "app").
- **Visual Architecture / Data-Flow Diagram:**

```
Root ──► 'a' ──► 'p' ──► 'p' (isWord=true: "app")
                          │
                         'l' ──► 'e' (isWord=true: "apple")
```

- **Canonical Code Skeleton:**
```csharp
public class TrieNode
{
    public TrieNode[] Children = new TrieNode[26];
    public bool IsWord = false;
}

public class Trie
{
    private TrieNode root = new TrieNode();

    public void Insert(string word)
    {
        TrieNode curr = root;
        foreach (char c in word)
        {
            int idx = c - 'a';
            if (curr.Children[idx] == null) curr.Children[idx] = new TrieNode();
            curr = curr.Children[idx];
        }
        curr.IsWord = true;
    }

    public bool Search(string word)
    {
        TrieNode node = GetNode(word);
        return node != null && node.IsWord;
    }

    public bool StartsWith(string prefix)
    {
        return GetNode(prefix) != null;
    }

    private TrieNode GetNode(string str)
    {
        TrieNode curr = root;
        foreach (char c in str)
        {
            int idx = c - 'a';
            if (curr.Children[idx] == null) return null;
            curr = curr.Children[idx];
        }
        return curr;
    }
}
```


- **Step-by-Step State Trace:**

| Operation | Input String | Target Nodes Traversed | Prefix Found? | `isWord` Flag at End | Return Value |
|:---:|:---:|:---|:---:|:---:|:---:|
| `insert` | `"apple"` | Root -> 'a' -> 'p' -> 'p' -> 'l' -> 'e' | Yes (Created) | Set `isWord = true` at 'e' | void |
| `search` | `"apple"` | Root -> 'a' -> 'p' -> 'p' -> 'l' -> 'e' | Yes | True | `true` |
| `startsWith`| `"app"` | Root -> 'a' -> 'p' -> 'p' | Yes | - | `true` |
| `search` | `"app"` | Root -> 'a' -> 'p' -> 'p' | Yes | False (before insert) | `false` |

- **Diagnostic Triggers:** "Implement Trie", "Word search II (grid + dictionary)", "Replace words / autocomplete".
- **Boundary Conditions:** Use `c - 'a'` for lowercase alphabets. Set `isWord = true` at termination node.
- **Real-World Application:** Autocomplete search suggestions, IP routing prefix tables, spell checkers.

### [PAT-25] Priority Queue / Min-Max Heap Filtering

- **Invariant (Neutral):** A binary heap maintains partial order invariants (parent $\le$ child for min-heap), providing $O(1)$ access to the extremal (minimum or maximum) element and $O(\log K)$ insertion/extraction over a dynamic collection of size $K$.
- **Mental Model:** A priority queue maintaining a moving leaderboard of top $K$ candidates.
- **Concrete Tracing Exemplar:** Kth Largest Element in an Array (e.g., `nums = [3, 2, 1, 5, 6, 4], k = 2`).
- **Visual Architecture / Data-Flow Diagram:**

```
Input Stream: 3, 2, 1, 5, 6, 4 (k=2)

Min-Heap of Size k=2:
After [3, 2]: Heap = [2, 3] (Root is min: 2)
Elem 1: 1 <= 2 -> Skip
Elem 5: 5 > 2  -> Pop 2, Push 5 -> Heap = [3, 5]
Elem 6: 6 > 3  -> Pop 3, Push 6 -> Heap = [5, 6]
Elem 4: 4 <= 5 -> Skip

Result: Heap Root = 5 (2nd Largest Element)
```

- **Canonical Code Skeleton:**
```csharp
public int[] TopKFrequent(int[] nums, int k)
{
    var freqMap = new Dictionary<int, int>();
    foreach (int n in nums)
    {
        freqMap[n] = freqMap.GetValueOrDefault(n, 0) + 1;
    }

    var minHeap = new PriorityQueue<int, int>();

    foreach (var entry in freqMap)
    {
        minHeap.Enqueue(entry.Key, entry.Value);
        if (minHeap.Count > k) minHeap.Dequeue();
    }

    var result = new int[k];
    for (int i = 0; i < k; i++)
    {
        result[i] = minHeap.Dequeue();
    }
    return result;
}
```


- **Step-by-Step State Trace (Input: `nums = [3, 2, 1, 5, 6, 4], k = 2`):**

| Step `i` | `nums[i]` | Action | Min-Heap State (Size $\le 2$) | Heap Root (`peek()`) |
|:---:|:---:|:---|:---|:---:|
| 0 | 3 | Push 3 | `[3]` | 3 |
| 1 | 2 | Push 2 | `[2, 3]` | 2 |
| 2 | 1 | $1 \le 2 \implies$ Skip | `[2, 3]` | 2 |
| 3 | 5 | $5 > 2 \implies$ Pop 2, Push 5 | `[3, 5]` | 3 |
| 4 | 6 | $6 > 3 \implies$ Pop 3, Push 6 | `[5, 6]` | 5 |
| 5 | 4 | $4 \le 5 \implies$ Skip | `[5, 6]` | 5 |
| End | - | Return `peek()` | `[5, 6]` | **5** |

- **Diagnostic Triggers:** "Find Kth largest/smallest element", "Merge K sorted lists", "Task priority scheduler".
- **Boundary Conditions:** To find $K$-th *largest*, use a *Min-Heap* of size $K$. To find $K$-th *smallest*, use a *Max-Heap* of size $K$.
- **Real-World Application:** Real-time top-K leaderboard engines, event scheduler timer queues.


> **Note on Mathematical and Bit Manipulation Patterns:** Several common interview problems rely on mathematical properties (XOR for finding missing/duplicate numbers, modular arithmetic, Gauss's sum formula) or bitwise operations (bitmask DP, bit counting). These techniques are cross-cutting tools that complement the structural patterns above rather than forming standalone patterns. When you encounter a problem involving XOR properties, power-of-two checks, or bitmask state encoding, recognize these as mathematical invariants that can be combined with the canonical patterns.
