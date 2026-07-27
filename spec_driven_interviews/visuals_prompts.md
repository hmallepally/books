# Visual Generation Prompts — Spec-Driven Coding Interviews Book
# Store all image generation prompts here for reproducibility.

## Chapter 10 (Q2): 2D Prefix Sum Query — Inclusion-Exclusion

**File:** `chapters/10-q2-matrix-simulation/visuals/prefix_sum_2d_query.png`

**Prompt:**
```
A clean, educational diagram explaining the 2D Prefix Sum Query using the inclusion-exclusion principle on a grid matrix.

Show a 5x5 grid with numbers. Use colored overlapping rectangles to illustrate the formula:
query(r1,c1, r2,c2) = sum[r2+1][c2+1] - sum[r1][c2+1] - sum[r2+1][c1] + sum[r1][c1]

Step-by-step visual breakdown with 4 panels:
Panel 1: "Start with FULL rectangle" - Show the big rectangle from (0,0) to (r2,c2) shaded in BLUE, labeled "sum[r2+1][c2+1]"
Panel 2: "Subtract TOP rows" - Show the rectangle from (0,0) to (r1-1, c2) shaded in RED with a minus sign, labeled "- sum[r1][c2+1]"
Panel 3: "Subtract LEFT columns" - Show the rectangle from (0,0) to (r2, c1-1) shaded in ORANGE with a minus sign, labeled "- sum[r2+1][c1]"
Panel 4: "Add back OVERLAP" - Show the small rectangle from (0,0) to (r1-1, c1-1) shaded in GREEN with a plus sign, labeled "+ sum[r1][c1]" because it was subtracted twice

Final result: The TARGET sub-rectangle from (r1,c1) to (r2,c2) highlighted in PURPLE, showing the clean result.

Use a white background, clean grid lines, clear labels, and a professional textbook style. Include the formula at the bottom. Make it look like a computer science textbook illustration.
```

* * *

## Chapter 10 (Q2): Spiral Boundary Traversal

**File:** `chapters/10-q2-matrix-simulation/visuals/spiral_traversal.png`

**Prompt:**
```
A clean, educational diagram explaining Spiral Matrix Boundary Traversal step by step.

Show a 5x5 grid matrix with numbers 1-25. Illustrate the spiral traversal pattern with 4 colored arrows showing the traversal order of the FIRST layer:

Step 1 (RED arrow going RIGHT): "Traverse TOP row" - arrow goes from (top, left) to (top, right), showing matrix[top][j] for j = left to right. Label: "top++"
Step 2 (GREEN arrow going DOWN): "Traverse RIGHT column" - arrow goes from (top, right) down to (bottom, right), showing matrix[i][right]. Label: "right--"
Step 3 (BLUE arrow going LEFT): "Traverse BOTTOM row" - arrow goes from (bottom, right) to (bottom, left), showing matrix[bottom][j] for j = right to left. Label: "bottom--"
Step 4 (ORANGE arrow going UP): "Traverse LEFT column" - arrow goes from (bottom, left) up to (top, left), showing matrix[i][left]. Label: "left++"

Show a second smaller 3x3 inner grid representing Layer 2 after boundaries contracted inward.

Include labels showing: top=0, bottom=4, left=0, right=4 for the initial boundaries.
Show the boundary variables contracting after each step.

Use a white background, clean grid lines with cell values visible, numbered arrows showing traversal order 1→2→3→4, and a professional textbook style. Make it look like a computer science textbook illustration.
```

* * *

## Chapter 9 (Q1): Two-Pointer Convergence Pattern

**File:** `chapters/09-q1-implementation/visuals/two_pointer_convergence.png`

**Prompt:**
```
A clean, educational diagram showing the Two-Pointer Convergence pattern on an array.

Show an array of 8 characters: ['r','a','c','e','c','a','r','s']. Place a LEFT pointer (green arrow) at index 0 and a RIGHT pointer (blue arrow) at index 7.

Show 4 steps vertically:
Step 1: left=0 ('r'), right=7 ('s') → 'r' != 's' → NOT palindrome (or show comparison)
Step 2: For a palindrome example ['r','a','c','e','c','a','r'], left=0 right=6 → 'r'=='r' ✓ → move both inward
Step 3: left=1 right=5 → 'a'=='a' ✓ → move both inward
Step 4: left=2 right=4 → 'c'=='c' ✓ → continue until left >= right

Show arrows converging toward center. Label the invariant: "Everything outside the pointers has been verified."

Use a white background, clean array cells with indices, colored pointer arrows, and a professional textbook style.
```

* * *

## Chapter 9 (Q1): Read-Write Pointer Pattern (Array Compaction)

**File:** `chapters/09-q1-implementation/visuals/read_write_pointer.png`

**Prompt:**
```
A clean, educational diagram showing the Read/Write Pointer pattern for in-place array compaction.

Example: Remove all zeros from [0, 1, 0, 3, 12] → [1, 3, 12, 0, 0]

Show the array with two pointers:
- WRITE pointer (red, below array) - marks where the next non-zero goes
- READ pointer (blue, above array) - scans every element

Show 5 steps:
Step 1: read=0 (value 0, skip), write=0
Step 2: read=1 (value 1, write it at write=0), write becomes 1
Step 3: read=2 (value 0, skip), write stays at 1
Step 4: read=3 (value 3, write it at write=1), write becomes 2
Step 5: read=4 (value 12, write it at write=2), write becomes 3
Final: Fill remaining with 0 → [1, 3, 12, 0, 0]

Label the invariant: "Everything before WRITE is the clean result. READ always moves forward."

Use a white background, clean array cells, colored pointer arrows, and a professional textbook style.
```

* * *

## Chapter 10 (Q2): BFS Level-by-Level on Grid (Rotting Oranges)

**File:** `chapters/10-q2-matrix-simulation/visuals/bfs_grid_levels.png`

**Prompt:**
```
A clean, educational diagram showing BFS Level-by-Level traversal on a grid, using the "Rotting Oranges" problem.

Show a 3x3 grid where:
- Cell (0,0) = rotten orange (dark brown/black)
- Cells (0,1), (1,0) = fresh oranges (orange)
- Cell (1,1) = fresh orange (orange)
- Cell (2,2) = fresh orange (orange)
- Other cells = empty (white)

Show 3 time steps:
t=0: Initial state - one rotten orange at (0,0), queue contains [(0,0)]
t=1: (0,0) spreads to (0,1) and (1,0) - these turn rotten. queue.size()=1 processes one level
t=2: (0,1) and (1,0) spread to (1,1) - it turns rotten. queue.size()=2 processes one level
t=3: (1,1) spreads - but (2,2) is not adjacent, so it can't be reached

Show the BFS wavefront expanding outward like ripples. Color code: brown=rotten, orange=fresh, gray=just rotted this step.

Label: "int size = queue.size() captures the ENTIRE current wavefront before processing"

Use a white background, clean grid, colored cells, and a professional textbook style.
```

* * *

## Chapter 11 (Q3): Sliding Window — Expand and Contract

**File:** `chapters/11-q3-hashmaps-sliding-windows/visuals/sliding_window.png`

**Prompt:**
```
A clean, educational diagram showing the Dynamic Sliding Window pattern.

Problem: Find the longest substring without repeating characters in "abcabcbb"

Show the string as an array of characters with indices 0-7.

Show 5 key states of the window:
State 1: left=0, right=2, window="abc" (length 3) - all unique ✓, expand right
State 2: left=0, right=3, window="abca" - 'a' repeats! Contract left
State 3: left=1, right=3, window="bca" (length 3) - all unique ✓, expand right
State 4: left=1, right=4, window="bcab" - 'b' repeats! Contract left
State 5: left=2, right=4, window="cab" (length 3) - all unique ✓

Show the window as a colored bracket/highlight below the array. Use GREEN for valid windows and RED flash for the moment a duplicate is detected.

Label: "RIGHT expands to explore. LEFT contracts to restore the invariant."
Show max_length = 3 tracked throughout.

Use a white background, clean character cells with indices, colored window brackets, and a professional textbook style.
```

* * *

## Chapter 11 (Q3): HashMap Frequency Signature

**File:** `chapters/11-q3-hashmaps-sliding-windows/visuals/hashmap_frequency.png`

**Prompt:**
```
A clean, educational diagram showing the HashMap Frequency Signature pattern for checking anagrams.

Show two strings: s1 = "abc" and s2 = "cbad"

Panel 1: Build frequency map for s1
{'a': 1, 'b': 1, 'c': 1} — shown as a small table/dictionary visualization

Panel 2: Slide a window of size 3 across s2
Window "cba" → frequency {'c':1, 'b':1, 'a':1} → MATCHES s1's frequency! → Anagram found at index 0
Window "bad" → frequency {'b':1, 'a':1, 'd':1} → Does NOT match

Show the sliding window moving across s2 with the frequency map updating:
- When right pointer enters, increment count
- When left pointer leaves, decrement count
- Compare maps after each slide

Label: "Two strings are anagrams if and only if their character frequency signatures are identical."

Use a white background, clean string cells, frequency table boxes, colored highlights for matches, and a professional textbook style.
```

* * *

## Chapter 12 (Q4): Binary Search on Rotated Sorted Array

**File:** `chapters/12-q4-optimization-dp/visuals/rotated_sorted_array.png`

**Prompt:**
```
A clean, educational diagram explaining Binary Search on a Rotated Sorted Array.

Show the array [4, 5, 6, 7, 0, 1, 2] with indices 0-6.

Panel 1: "What is a rotated sorted array?"
Show the original sorted array [0,1,2,4,5,6,7] and an arrow showing it being "rotated" at the pivot point, resulting in [4,5,6,7,0,1,2]. The pivot/minimum is at index 4 (value 0).

Panel 2: "The Monotonic Partition Invariant"
Show the array split into TWO sorted halves:
- Left half [4,5,6,7] — sorted, all values >= arr[0]
- Right half [0,1,2] — sorted, all values < arr[0]
Draw a "cliff" or step-down between index 3 (value 7) and index 4 (value 0).

Panel 3: "Binary Search Decision"
Show mid = 3 (value 7). Since arr[mid] >= arr[lo], the LEFT half is sorted.
- If target is in [arr[lo], arr[mid]], search left
- Otherwise, search right
Show arrows indicating which half to search based on target value.

Label: "Key insight: At least ONE half is always sorted. Check if target falls in the sorted half."

Use a white background, clean array cells with values and indices, colored halves (blue for left sorted, green for right sorted), and a professional textbook style.
```

* * *

## Chapter 12 (Q4): DP State Transition — Climbing Stairs

**File:** `chapters/12-q4-optimization-dp/visuals/dp_climbing_stairs.png`

**Prompt:**
```
A clean, educational diagram explaining Dynamic Programming state transition using the Climbing Stairs problem.

Show a staircase with 5 steps. The question: "How many ways to reach step N if you can climb 1 or 2 steps at a time?"

Panel 1: "Recursive tree" (top-down) — show the overlapping subproblems
f(5) branches to f(4) and f(3)
f(4) branches to f(3) and f(2)
f(3) branches to f(2) and f(1)
Highlight the OVERLAPPING calls to f(3) and f(2) in red — these are computed multiple times.

Panel 2: "DP Table" (bottom-up)
Show a 1D array: dp[0]=1, dp[1]=1, dp[2]=2, dp[3]=3, dp[4]=5, dp[5]=8
Arrow showing: dp[i] = dp[i-1] + dp[i-2]
Highlight that this is the Fibonacci recurrence.

Panel 3: "Space Optimization"
Show that you only need prev=dp[i-2] and curr=dp[i-1] — two variables instead of a full array.
dp[5] = 5 + 3 = 8

Label: "Recognize the Fibonacci pattern → compress from O(N) space to O(1)"

Use a white background, clean staircase illustration, tree diagram, DP table with arrows, and a professional textbook style.
```
