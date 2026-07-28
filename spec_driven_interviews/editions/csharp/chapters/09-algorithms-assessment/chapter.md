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

![Big-O Time Complexity Comparison Graph](visuals/big_o_comparison.jpg){width=85%}

**The Constraint-to-Complexity Rule:** Read the problem constraints FIRST. If N ≤ 10^4, O(N²) is acceptable. If N ≤ 10^5, you need O(N log N) or better. If N ≤ 10^6, you need O(N). This single rule eliminates 50% of wrong algorithm choices before you write a line of code.

![Constraint-to-Complexity Flowchart](visuals/constraint_flowchart.jpg){width=85%}

---

# The 24 Canonical Programming Patterns

The following catalog defines the 24 fundamental patterns of computational problem-solving. Each pattern represents a proven, invariant structure for solving a specific class of problems.

---

## Module 1: Array & String Mechanics

### [PAT-01] Direct Indexing & Frequency Buckets

- **Invariant:** When the input domain is finite (e.g., ASCII characters, digits $0..9$), a fixed-size array (`int[256]`) provides $O(1)$ direct-indexing lookup without hash overhead.
- **Mental Model:** Use the array index itself as the key.
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

- **Diagnostic Triggers:** "First non-repeating character", "Anagram check", "Character frequency".
- **Boundary Conditions:** Ensure array size covers the domain (`256` for ASCII, `26` for lowercase English).
- **Real-World Application:** High-speed network packet inspection, audit log frequency counting.

---

### [PAT-02] In-Place Mutation & Two-Pointer Compaction

- **Invariant:** A `write` pointer tracks the boundary of valid elements while a `read` pointer scans the array, mutating data in-place in $O(1)$ extra space.
- **Mental Model:** Filter or compact elements in a single pass without allocating a new array.
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

- **Diagnostic Triggers:** "In-place removal", "Compact array", "Move zeroes to end".
- **Boundary Conditions:** Handle empty array or single-element array upfront.
- **Real-World Application:** Memory defragmentation, log stream sanitization.

---

### [PAT-03] Prefix Sums & Range Query Invariants

- **Invariant:** The sum of elements between indices $i$ and $j$ equals `prefix[j + 1] - prefix[i]`, turning range sum queries into $O(1)$ operations.
- **Mental Model:** Precompute cumulative totals so any subarray sum is computed by subtraction.
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

- **Diagnostic Triggers:** "Subarray sum equals K", "Range sum queries", "Equal number of 0s and 1s".
- **Boundary Conditions:** Always initialize `prefCounts.put(0, 1)` to account for subarrays starting at index 0.
- **Real-World Application:** Financial ledger balance auditing, telemetry interval aggregation.

---

## Module 2: Windowing & Pointer Navigation

### [PAT-04] Dynamic Sliding Window (Variable Size)

- **Invariant:** Maintain a window `[left...right]`. Expand `right` to include elements. When constraint is violated, shrink from `left` until valid.
- **Mental Model:** An expanding and contracting net scanning an array.
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

- **Diagnostic Triggers:** "Longest/shortest subarray satisfying condition X", "At most K distinct elements".
- **Boundary Conditions:** Set-based windows must shrink BEFORE expanding; HashMap/Sum-based windows expand FIRST then shrink.
- **Real-World Application:** Sliding-window rate limiters, network throughput monitoring.

---

### [PAT-05] Fixed-Size Monotonic Deque Window

- **Invariant:** Maintain a `Deque` of indices where corresponding values are strictly decreasing from front to back. Front always holds the maximum of the current window.
- **Mental Model:** A sliding window of fixed size $K$ that tracks max/min in $O(1)$ amortized time.
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

- **Diagnostic Triggers:** "Maximum/minimum in every window of size K".
- **Boundary Conditions:** Deque stores INDICES, not values. Window is full when `i >= k - 1`.
- **Real-World Application:** Real-time SLA monitoring, financial tick-data peak detection.

---

### [PAT-06] Converging Two-Pointers

- **Invariant:** Two pointers start at opposite ends (`left = 0`, `right = n - 1`) of a sorted array and move inward based on comparison with target.
- **Mental Model:** Squeezing the search space from both boundaries.
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

- **Diagnostic Triggers:** "Sorted array + find pair", "Container with most water", "Palindrome validation".
- **Boundary Conditions:** Array MUST be sorted. Loop condition is `left < right` (pointers must not overlap for pairs).
- **Real-World Application:** Order matching engines, debit-credit balance pairing.

---

### [PAT-07] Fast & Slow Pointers (Floyd's Cycle Detection)

- **Invariant:** `slow` moves 1 step while `fast` moves 2 steps. If a cycle exists, `fast` will eventually catch `slow`.
- **Mental Model:** Two runners on a circular track.
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

- **Diagnostic Triggers:** "Detect cycle in linked list", "Find duplicate number", "Happy number".
- **Boundary Conditions:** Check `fast != null && fast.next != null` to avoid `NullPointerException`.
- **Real-World Application:** Circular reference detection in graph engines, deadlock detection.

---

## Module 3: Stacks, Queues & Monotonic Structures

### [PAT-08] LIFO Matching & Expression Parsing

- **Invariant:** Push open symbols onto a stack. When a closing symbol is encountered, pop and verify it matches the expected opening symbol.
- **Mental Model:** Last-in, first-out validation of nested structures.
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

- **Diagnostic Triggers:** "Valid parentheses", "Evaluate expression", "Simplify file path".
- **Boundary Conditions:** Stack must be empty at the end. Check `stack.isEmpty()` before popping.
- **Real-World Application:** JSON/XML syntax parsers, compiler AST validation, undo stacks.

---

### [PAT-09] Monotonic Stack ("The Waiting Room")

- **Invariant:** Stack holds unresolved element indices in decreasing order. When a larger element arrives, it pops colder elements and resolves their answers.
- **Mental Model:** A waiting room where people stay until someone taller arrives to liberate them.
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

- **Diagnostic Triggers:** "Next greater element", "Daily temperatures", "Largest rectangle in histogram".
- **Boundary Conditions:** Store INDICES on stack, not values. Unresolved items remain `0` or `-1`.
- **Real-World Application:** Stock price drop alerts, automated threshold breach notifications.

---

## Module 4: Search Space & Decision Trees

### [PAT-10] Monotonic Partition Binary Search

- **Invariant:** In a rotated sorted array, at least one half (left or right) is always strictly sorted.
- **Mental Model:** Halving search space by identifying the sorted partition.
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

- **Diagnostic Triggers:** "Search in rotated sorted array", "Find minimum in rotated sorted array".
- **Boundary Conditions:** Use `nums[left] <= nums[mid]` (with `<=`) to handle single-element partitions.
- **Real-World Application:** Distributed partition log search, sharded database key lookups.

---

### [PAT-11] Binary Search on Solution Range

- **Invariant:** When the answer lies within a known numeric range `[min...max]` and a predicate function `feasible(x)` is monotonic, binary search finds the optimal value.
- **Mental Model:** Guess the answer, test if it works, halve the range.
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

- **Diagnostic Triggers:** "Find minimum capacity", "Koko eating bananas", "Split array largest sum".
- **Boundary Conditions:** Define correct range bounds `[lo, hi]` upfront.
- **Real-World Application:** Capacity planning, thread pool sizing, rate limit optimization.

---

### [PAT-12] Backtracking & State-Space Pruning

- **Invariant:** Explore decision paths recursively; when a path violates constraints, backtrack (undo state change) and try the next branch.
- **Mental Model:** Exploring a maze by dropping breadcrumbs and stepping back when hitting a dead end.
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

- **Diagnostic Triggers:** "Generate all permutations/combinations", "Sudoku solver", "N-Queens".
- **Boundary Conditions:** Always make a deep copy `new ArrayList<>(path)` when adding to results.
- **Real-World Application:** Constraint satisfaction solvers, security permission path traversal.

---

## Module 5: Graph & Grid Traversals

### [PAT-13] Level-by-Level BFS Wavefront

- **Invariant:** Queue processes nodes layer-by-layer (`int size = queue.size()`). First time target is popped = shortest path in unweighted graph/grid.
- **Mental Model:** Water ripples expanding outward in concentric circles.
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

- **Diagnostic Triggers:** "Shortest path in grid", "Minimum steps to reach goal", "Word ladder".
- **Boundary Conditions:** ALWAYS mark `visited = true` on `offer()`, NOT on `poll()`.
- **Real-World Application:** Network routing protocols, social network distance calculation.

---

### [PAT-14] Multi-Source BFS Parallel Spreading

- **Invariant:** Push ALL starting origin points into the Queue at time $t=0$. The wavefront expands from all origins simultaneously.
- **Mental Model:** Multiple fires starting at different spots and spreading at equal speed.
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

- **Diagnostic Triggers:** "Rotting oranges", "Walls and gates", "Multi-point fire propagation".
- **Boundary Conditions:** Track remaining fresh target count to avoid extra minute increment.
- **Real-World Application:** Multi-datacenter cache invalidation, rumor/virus propagation modeling.

---

### [PAT-15] DFS Component Sinking & Flood Fill

- **Invariant:** Traverse connected component recursively; mutate cell value (`'1' -> '0'`) to mark visited and eliminate memory overhead.
- **Mental Model:** Sinking an island as you walk over it so you never visit it again.
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

- **Diagnostic Triggers:** "Number of islands", "Surrounded regions", "Flood fill".
- **Boundary Conditions:** Base case must check bounds BEFORE accessing `grid[r][c]`.
- **Real-World Application:** Image segmentation, cluster isolation, GIS landmass detection.

---

### [PAT-16] Topological Sort (Kahn's & DFS)

- **Invariant:** Process nodes with in-degree 0 first. Reduces in-degree of neighbors. If processed count $< N$, a cycle exists.
- **Mental Model:** Resolving build dependencies in order.
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

- **Diagnostic Triggers:** "Course schedule", "Task dependency ordering", "Build order".
- **Boundary Conditions:** Return empty array if `idx != numCourses` (cycle detected).
- **Real-World Application:** Maven/Gradle build execution, CI/CD pipeline stage ordering.

---

### [PAT-17] Disjoint Set Union (Union-Find)

- **Invariant:** Maintain connected sets using parent pointers with path compression and rank optimization for near $O(1)$ amortized `find` and `union`.
- **Mental Model:** Merging social groups and checking if two people share the same root leader.
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

- **Diagnostic Triggers:** "Redundant connection", "Number of connected components", "Accounts merge".
- **Boundary Conditions:** Path compression `parent[i] = find(parent[i])` is essential for optimal speed.
- **Real-World Application:** Network topology clustering, distributed consensus membership tracking.

---

### [PAT-18] Weighted Shortest Path (Dijkstra / Min-Heap)

- **Invariant:** Use a `PriorityQueue` ordered by distance. Always expand the unvisited node with the smallest tentative distance.
- **Mental Model:** Exploring shortest path on a map with varying road costs.
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

- **Diagnostic Triggers:** "Network delay time", "Cheapest flight within K stops", "Shortest path with weights".
- **Boundary Conditions:** PriorityQueue stores `[node, total_distance]`. Skip already finalized nodes (`dist.containsKey(node)`).
- **Real-World Application:** Latency-based API gateway routing, Google Maps route optimization.

---

## Module 6: Dynamic Programming & Optimization

### [PAT-19] 1D Choice Optimization (O(1) Space DP)

- **Invariant:** State `dp[i]` depends only on `dp[i - 1]` and `dp[i - 2]`. Space can be optimized from $O(N)$ array to 2 variables (`prev1`, `prev2`).
- **Mental Model:** Making optimal choice between taking current item or skipping it.
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

- **Diagnostic Triggers:** "House robber", "Climbing stairs", "Min cost climbing stairs".
- **Boundary Conditions:** Handle single-element input upfront.
- **Real-World Application:** Capacity allocation, CPU time-slot scheduling.

---

### [PAT-20] 0/1 & Unbounded Knapsack DP

- **Invariant:** `dp[w]` represents max value for capacity `w`. Iterate items and update capacity backwards for 0/1 (use item once) or forwards for unbounded (use item infinitely).
- **Mental Model:** Packing a backpack with items to maximize value without exceeding weight capacity.
- **Canonical Code Skeleton (Coin Change - Unbounded):**
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

- **Diagnostic Triggers:** "Coin change", "Partition equal subset sum", "Knapsack capacity".
- **Boundary Conditions:** Fill array with sentinel value (`amount + 1`) representing infinity.
- **Real-World Application:** Resource packing in cloud instances, currency change calculators.

---

### [PAT-21] 2D Grid Path Optimization

- **Invariant:** `dp[r][c]` represents min/max value to reach cell `(r, c)`, which depends on `dp[r - 1][c]` (from top) and `dp[r][c - 1]` (from left).
- **Mental Model:** Walking down and right on a grid accumulating values.
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

- **Diagnostic Triggers:** "Minimum path sum", "Unique paths in grid", "Dungeon game".
- **Boundary Conditions:** Initialize first row and first column carefully.
- **Real-World Application:** Cost-effective data routing across grid-structured networks.

---

### [PAT-22] String Alignment & Sequence DP

- **Invariant:** `dp[i][j]` represents optimal alignment score for prefix `s1[0..i-1]` and `s2[0..j-1]`.
- **Mental Model:** 2D grid matching characters of two strings.
- **Canonical Code Skeleton (Longest Common Subsequence):**
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

- **Diagnostic Triggers:** "Longest common subsequence", "Edit distance", "Wildcard matching".
- **Boundary Conditions:** Matrix dimensions are `(m + 1) x (n + 1)`. Access chars using `i - 1` and `j - 1`.
- **Real-World Application:** Git diff algorithms, DNA sequence alignment, text similarity search.

---

### [PAT-23] Sweep-Line & Interval Scheduling

- **Invariant:** Sort intervals by start time. Use a pointer or heap to process overlapping boundaries.
- **Mental Model:** Sweeping a vertical timeline left-to-right across time intervals.
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

- **Diagnostic Triggers:** "Meeting rooms II", "Merge intervals", "Non-overlapping intervals".
- **Boundary Conditions:** Always sort intervals by start time `a[0] - b[0]` first.
- **Real-World Application:** Calendar scheduling engines, hotel room allocation, cloud VM provisioning.

---

### [PAT-24] Trie Prefix Search & Retrieval

- **Invariant:** Tree structure where each node represents a character. Root-to-node path forms a string prefix, enabling $O(L)$ word lookup where $L$ is word length.
- **Mental Model:** Dictionary tree branching by character.
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

- **Diagnostic Triggers:** "Implement Trie", "Word search II (grid + dictionary)", "Replace words / autocomplete".
- **Boundary Conditions:** Use `c - 'a'` for lowercase alphabets. Set `isWord = true` at termination node.
- **Real-World Application:** Autocomplete search suggestions, IP routing prefix tables, spell checkers.

---

### [PAT-25] Priority Queue / Min-Max Heap

**Diagnostic Trigger:** "Find the K-th largest/smallest", "Merge K sorted lists", "Schedule tasks by priority", or any problem requiring efficient access to the minimum or maximum element while dynamically inserting.

**Invariant:** The heap property is maintained: for a min-heap, every parent node is ≤ its children. This guarantees O(1) access to the minimum and O(log N) insertion/extraction.

**Canonical Skeleton:**
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


**Complexity:** O(N log K) time, O(N + K) space.

> **Note on Mathematical and Bit Manipulation Patterns:** Several common interview problems rely on mathematical properties (XOR for finding missing/duplicate numbers, modular arithmetic, Gauss's sum formula) or bitwise operations (bitmask DP, bit counting). These techniques are cross-cutting tools that complement the structural patterns above rather than forming standalone patterns. When you encounter a problem involving XOR properties, power-of-two checks, or bitmask state encoding, recognize these as mathematical invariants that can be combined with the canonical patterns.
