import random

HEADER = """# 20 Timed Algorithmic Mock Assessment Sets

## How to Use This Chapter

This chapter provides 20 full, four-question exam mock sets (80 problems total) modeled after the common standardized coding assessment format. Each set is designed to simulate a rigorous timed assessment environment. The problems follow a standard difficulty curve: the first question tests basic implementation and traversal (Easy, 5-8 minutes), the second focuses on 2D matrices and simulation (Medium, 12-15 minutes), the third requires algorithmic pattern recognition like HashMaps or sliding windows (Medium-Hard, 18-20 minutes), and the fourth challenges you with dynamic programming, graphs, or advanced data structures (Hard, 20-25 minutes).

To get the most out of these mock assessments, strictly time yourself. Set a timer for 70 minutes (or adjust to match your target assessment format) and attempt all four questions in order. Do not look up syntax or external resources. If you get stuck on the third or fourth question, practice timeboxing: move on and secure partial credit where possible. For equal-weight assessment formats, treat all four questions as having equal priority and allocate approximately 15-18 minutes per question. After time expires, review your performance. Use the provided hints to guide your post-assessment study sessions, identifying which specific patterns (e.g., sliding window, BFS, monotonic stack) require further review.

Remember, there is no code in this chapter—this is your practice arena. Read the specifications, analyze the test cases, check the constraints, and write your own optimal solutions.

* * *

## Set 1: Warm-Up Fundamentals

* **Q1 (Easy): Vowel Starting Words**
  * *Specification:* Given a string of text containing words separated by single spaces, calculate the total number of words that begin with a vowel. Vowels are defined as 'a', 'e', 'i', 'o', and 'u', and the check should be case-insensitive. Ignore any punctuation, assuming the string consists only of alphabetical characters and spaces. Return the final integer count of qualifying words.
  * *Sample Test Case:* Input: `"Apple banana Orange umbrella"` -> Output: `3`.
  * *Constraints:* String length $1 \\le L \\le 10^5$.
  * *Hint:* Use standard string splitting to isolate words, then check the first character of each token against a predefined set of vowels.

* **Q2 (Medium): Rotate Rectangular Image**
  * *Specification:* You are given an $M \\times N$ 2D matrix representing an image, where each cell holds a pixel value. Your task is to rotate the image 90 degrees clockwise. Unlike square matrix rotation, this matrix is rectangular, meaning the dimensions of the resulting matrix will swap to $N \\times M$. You must allocate a new matrix to hold the rotated values and populate it correctly.
  * *Sample Test Case:* Input: `[[1, 2, 3], [4, 5, 6]]` -> Output: `[[4, 1], [5, 2], [6, 3]]`.
  * *Constraints:* $1 \\le M, N \\le 1000$.
  * *Hint:* The element at `matrix[r][c]` in the original matrix moves to `new_matrix[c][M - 1 - r]` in the rotated matrix.

* **Q3 (Medium-Hard): K-Frequency Substring**
  * *Specification:* Given a string and an integer K, find the length of the longest contiguous substring where no character appears more than K times. You must process the string and keep track of character frequencies dynamically. If the frequency of any character exceeds K, you must shrink the valid sequence until the condition is met again. Return the maximum length observed.
  * *Sample Test Case:* Input: `s = "abaccc", K = 2` -> Output: `4` (The substring "abac").
  * *Constraints:* String length $1 \\le L \\le 10^5$, $1 \\le K \\le L$.
  * *Hint:* Use a sliding window approach with two pointers and a HashMap or frequency array to track character counts within the current window.

* **Q4 (Hard): Largest Rectangular Area**
  * *Specification:* You are given an array of non-negative integers representing the heights of adjacent buildings, where each building has a width of 1 unit. You need to calculate the area of the largest rectangle that can be formed within the bounds of these buildings. The rectangle must be completely contained within the histograms. Return the maximum possible area.
  * *Sample Test Case:* Input: `[2, 1, 5, 6, 2, 3]` -> Output: `10` (Formed by heights 5 and 6).
  * *Constraints:* Array length $1 \\le N \\le 10^5$, building heights $0 \\le H \\le 10^4$.
  * *Hint:* Utilize a monotonic increasing stack to keep track of building indices, calculating areas when a drop in height is encountered.
"""

SURVIVAL_GUIDE = """
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
   - *Rectangular $R \\times C \\rightarrow C \\times R$:* `target[j][R - 1 - i] = matrix[i][j]`
   - *Square $N \\times N$ In-Place:* Transpose (`swap(matrix[i][j], matrix[j][i])` for `j > i`), then reverse each row horizontally (`swap(matrix[i][j], matrix[i][N - 1 - j])` for `j < N / 2`).
8. **Binary Search Middle Overflow & Bounds:**  
   Always write `mid = left + (right - left) / 2`. In rotated sorted arrays, check sorted half first: if `nums[left] <= nums[mid]`, left half is monotonically sorted.
9. **Numeric Accumulator Overflow:**  
   When calculating product, array sums, or coordinate products, initialize sum/product accumulators as `long` to prevent 32-bit integer overflow before returning `(int) sum`.
10. **Array Bounds Guarding:**  
    Always check `array != null && array.length > 0` before accessing index `0`, and ensure loops end at `i < array.length` (or `i <= array.length` when using a sentinel).
"""

easy_templates = [
    ("String Reversal", "Reverse a given string preserving whitespace and capitalization constraints.", "`\"Hello\" -> \"olleH\"`", "[PAT-02] Two pointers"),
    ("Frequency Counting", "Find the most frequent character in a given string. Break ties alphabetically.", "`\"abac\" -> 'a'`", "[PAT-01] Frequency Array"),
    ("Array Prefix Sum", "Given an array, calculate its running sum in place.", "`[1,2,3] -> [1,3,6]`", "[PAT-03] Prefix Sums"),
    ("Palindrome Check", "Verify if a string is a palindrome, ignoring non-alphanumeric characters.", "`\"A man, a plan\" -> True`", "[PAT-06] Converging Pointers"),
    ("In-place Transformation", "Move all zeros in an array to the end while maintaining relative order of other elements.", "`[0,1,0,3] -> [1,3,0,0]`", "[PAT-02] Write/Read pointers"),
    ("Simple Math", "Return the sum of digits of a given integer until it becomes a single digit.", "`38 -> 2`", "Modulo arithmetic"),
    ("Anagram Validation", "Determine if two strings are valid anagrams of one another.", "`\"listen\", \"silent\" -> True`", "[PAT-01] Frequency buckets"),
    ("Array Intersection", "Find the common elements between two sorted arrays.", "`[1,2,3], [2,3,4] -> [2,3]`", "Two pointers matching"),
    ("Missing Number", "Find the missing number in an array of size N containing numbers from 0 to N.", "`[0,1,3] -> 2`", "Sum formula or XOR"),
    ("Merge Sorted Arrays", "Merge two sorted arrays into a new sorted array.", "`[1,3], [2,4] -> [1,2,3,4]`", "Two pointer merge"),
    ("Longest Prefix", "Find the longest common prefix string amongst an array of strings.", "`[\"flower\", \"flow\"] -> \"flow\"`", "Vertical string scanning"),
    ("Valid Parentheses Basic", "Check if a string with just () is balanced.", "`\"(())\" -> True`", "Counter tracking"),
    ("Count Elements", "Count elements in array that have x+1 present in the array.", "`[1,2,3] -> 2`", "HashSet lookup"),
    ("Majority Element", "Find the element that appears more than n/2 times.", "`[2,2,1,1,1,2,2] -> 2`", "Boyer-Moore Voting"),
    ("First Unique Character", "Find the first non-repeating character in a string.", "`\"leetcode\" -> 0`", "[PAT-01] Frequency counting"),
    ("Detect Capital", "Verify if the capitalization of a word is correct (all caps, all lower, or title).", "`\"USA\" -> True`", "String traversal"),
    ("Reverse Words", "Reverse the order of words in a string.", "`\"the sky is blue\" -> \"blue is sky the\"`", "Split and reverse"),
    ("Contains Duplicate", "Return true if any value appears at least twice in the array.", "`[1,2,3,1] -> True`", "HashSet"),
    ("Remove Element", "Remove all instances of a specific value in-place.", "`[3,2,2,3], val=3 -> len=2`", "[PAT-02] Mutation"),
]

medium_templates = [
    ("Matrix Spiral", "Traverse a 2D matrix in spiral order and return the elements.", "`[[1,2],[3,4]] -> [1,2,4,3]`", "Boundary tracking simulation"),
    ("Two Pointer Target", "Find two numbers in a sorted array that add up to target.", "`[2,7,11,15], target=9 -> [0,1]`", "[PAT-06] Converging Pointers"),
    ("Prefix Sum Range", "Process range sum queries on an array quickly.", "`[1,2,3], query(0,2) -> 6`", "[PAT-03] Prefix array"),
    ("Binary Search Rotated", "Find an element in a sorted array that has been rotated.", "`[4,5,1,2,3], target=1 -> 2`", "[PAT-10] Partition Search"),
    ("State Machine String", "Parse a string to extract a valid integer, handling signs and overflow.", "`\"-42\" -> -42`", "Deterministic finite automaton"),
    ("Matrix Zeroes", "If a cell is 0, set its entire row and column to 0 in-place.", "`[[1,0],[1,1]] -> [[0,0],[1,0]]`", "Row/Col marker tracking"),
    ("Subarray Sum K", "Find the total number of continuous subarrays whose sum equals k.", "`[1,1,1], k=2 -> 2`", "[PAT-03] Prefix HashMap"),
    ("Sort Colors", "Sort an array of 0s, 1s, and 2s in-place (Dutch National Flag).", "`[2,0,2,1,1,0] -> [0,0,1,1,2,2]`", "Three pointers"),
    ("Peak Element", "Find a peak element (strictly greater than neighbors) in O(log N) time.", "`[1,2,3,1] -> 2`", "Binary Search on gradient"),
    ("Group Anagrams", "Group an array of strings into anagram sets.", "`[\"eat\",\"tea\",\"tan\"] -> [[\"eat\",\"tea\"],[\"tan\"]]`", "Frequency string as HashMap key"),
    ("Max Area Container", "Find two lines that together with the x-axis form a container holding the most water.", "`[1,8,6,2,5,4,8,3,7] -> 49`", "[PAT-06] Converging Two-Pointers"),
    ("Generate Parentheses", "Generate all combinations of n pairs of well-formed parentheses.", "`n=2 -> [\"(())\",\"()()\"]`", "[PAT-12] Backtracking"),
    ("Valid Sudoku", "Determine if a 9x9 Sudoku board is valid.", "Standard sudoku validation", "HashMap/Array bitmasking"),
    ("Longest Consecutive Sequence", "Find the length of the longest consecutive elements sequence in O(N).", "`[100,4,200,1,3,2] -> 4`", "HashSet building blocks"),
    ("Top K Frequent Elements", "Return the k most frequent elements in an array.", "`[1,1,1,2,2,3], k=2 -> [1,2]`", "HashMap and Min-Heap"),
    ("Product of Array Except Self", "Return array such that answer[i] is product of all elements except nums[i].", "`[1,2,3,4] -> [24,12,8,6]`", "Left/Right prefix products"),
    ("Search 2D Matrix", "Search for a value in a sorted 2D matrix in O(log(MN)).", "`matrix, target=3 -> True`", "Virtual 1D Binary Search"),
    ("Minimum Size Subarray Sum", "Find minimal length of subarray with sum >= target.", "`target=7, [2,3,1,2,4,3] -> 2`", "[PAT-04] Dynamic Sliding Window"),
    ("Kth Largest Element", "Find the kth largest element in an unsorted array.", "`[3,2,1,5,6,4], k=2 -> 5`", "Min-Heap or QuickSelect"),
]

mh_templates = [
    ("Sliding Window Max", "Find the maximum string length without repeating characters.", "`\"abcabc\" -> 3`", "[PAT-04] Dynamic Sliding Window"),
    ("HashMap Multi-key", "Find the longest subarray with equal numbers of 0s and 1s.", "`[0,1,0] -> 2`", "[PAT-03] Prefix Sums Hash"),
    ("BFS Shortest Path", "Find the shortest path to exit a grid maze.", "`grid -> 4 steps`", "[PAT-13] BFS Wavefront"),
    ("DFS Component Count", "Count the number of connected components (islands) in a 2D grid.", "`grid -> 3 islands`", "[PAT-15] DFS Flood Fill"),
    ("Tree Traversal", "Serialize and deserialize a binary tree.", "`[1,2,3] -> str -> [1,2,3]`", "Preorder traversal"),
    ("Course Schedule II", "Return the ordering of courses you should take to finish all courses.", "`num=2, req=[[1,0]] -> [0,1]`", "[PAT-16] Topological Sort"),
    ("Word Search", "Check if a word exists in a grid of characters.", "`board, \"ABCCED\" -> True`", "[PAT-12] DFS Backtracking"),
    ("Clone Graph", "Return a deep copy (clone) of a graph.", "`node 1 -> cloned node 1`", "HashMap + BFS/DFS"),
    ("Evaluate Division", "Evaluate queries based on equation relationships a/b = 2.", "`a/b=2, b/c=3 -> a/c=6`", "Graph DFS with path weights"),
    ("Time Based Key-Value Store", "Create a map that supports setting and getting values by timestamps.", "`set(k,v,1), get(k,1) -> v`", "HashMap + Binary Search"),
    ("LRU Cache", "Design a cache with Least Recently Used eviction strategy.", "`put(1,1), get(1) -> 1`", "HashMap + Doubly Linked List"),
    ("Merge Intervals", "Merge all overlapping intervals.", "`[[1,3],[2,6]] -> [[1,6]]`", "[PAT-23] Sweep-Line Sort"),
    ("Construct Binary Tree", "Build a tree from preorder and inorder traversal arrays.", "`pre=[3,9], in=[9,3] -> Tree`", "Divide and conquer"),
    ("Design Add and Search Words", "Design a data structure that supports adding words and searching with '.' wildcards.", "`add(\"bad\"), search(\"b.d\") -> True`", "[PAT-24] Trie with DFS"),
    ("Permutations", "Return all possible permutations of an array of distinct integers.", "`[1,2] -> [[1,2],[2,1]]`", "[PAT-12] Backtracking"),
    ("Pacific Atlantic Water Flow", "Find grid coordinates where water can flow to both Pacific and Atlantic oceans.", "`grid -> [[0,4],[1,3]]`", "[PAT-15] DFS from borders"),
    ("Accounts Merge", "Merge user accounts that share common email addresses.", "`[[John, a@a.com, b@b.com]] -> merged`", "[PAT-17] Union-Find"),
    ("Daily Temperatures", "Find how many days to wait for a warmer temperature.", "`[73,74,75,71] -> [1,1,0,0]`", "[PAT-09] Monotonic Stack"),
    ("Reorder List", "Reorder a linked list to L0 -> Ln -> L1 -> Ln-1.", "`1->2->3->4 -> 1->4->2->3`", "Middle finding + Reverse + Merge"),
]

hard_templates = [
    ("1D DP Robber", "Find max value you can rob without triggering adjacent alarms in a circular street.", "`[2,3,2] -> 3`", "[PAT-19] DP State Machine"),
    ("2D DP Pathing", "Find minimum path sum in grid moving down/right.", "`[[1,3,1],[1,5,1]] -> 7`", "[PAT-21] 2D DP Grid"),
    ("Monotonic Stack Max Area", "Find the largest rectangle in a binary matrix of 0s and 1s.", "`matrix -> 6`", "[PAT-09] Monotonic Stack"),
    ("Dijkstra Shortest", "Find network delay time for a signal to reach all nodes.", "`nodes=4, edges -> 2`", "[PAT-18] Dijkstra Priority Queue"),
    ("Topological Sort Complex", "Find the longest path in a Directed Acyclic Graph representing tasks.", "`tasks -> 10 days`", "[PAT-16] Topo Sort / DP"),
    ("Union Find Network", "Find the redundant connection in a graph that should be a tree.", "`[[1,2],[1,3],[2,3]] -> [2,3]`", "[PAT-17] Disjoint Set Union"),
    ("Word Ladder", "Find the length of the shortest transformation sequence from beginWord to endWord.", "`hit -> cog: 5`", "[PAT-13] BFS Wavefront"),
    ("Alien Dictionary", "Derive the alphabetical order of an alien language from a sorted dictionary.", "`[\"wrt\",\"wrf\"] -> \"t\" before \"f\"`", "[PAT-16] Topological Sort"),
    ("Trapping Rain Water", "Compute how much water it can trap after raining.", "`[0,1,0,2,1,0,1,3] -> 6`", "Two Pointers or [PAT-09] Stack"),
    ("Edit Distance", "Find minimum operations to convert word1 to word2.", "`horse -> ros: 3`", "[PAT-22] 2D String DP"),
    ("Longest Increasing Path", "Find the longest increasing path in a matrix.", "`[[9,9,4],[6,6,8]] -> 4`", "DFS + Memoization"),
    ("Burst Balloons", "Maximize coins by bursting balloons strategically.", "`[3,1,5,8] -> 167`", "Divide & Conquer DP"),
    ("Regular Expression Matching", "Implement regular expression matching with support for '.' and '*'.", "`aa, a* -> True`", "[PAT-22] 2D DP String"),
    ("Sliding Window Maximum", "Return the max sliding window of size k.", "`[1,3,-1,-3,5,3], k=3 -> [3,3,5,5]`", "[PAT-05] Monotonic Deque"),
    ("Find Median from Data Stream", "Design a class to calculate the median of numbers from a data stream.", "`add(1), add(2), median -> 1.5`", "Two Heaps (Min/Max)"),
    ("Merge K Sorted Lists", "Merge k sorted linked lists and return it as one sorted list.", "`[1->4, 1->3] -> 1->1->3->4`", "Priority Queue (Min-Heap)"),
    ("Serialize N-ary Tree", "Design an algorithm to serialize and deserialize an N-ary tree.", "`tree -> string -> tree`", "DFS Preorder"),
    ("Minimum Window Substring", "Find the minimum window in S which will contain all characters in T.", "`S=\"ADOBECODEBANC\", T=\"ABC\" -> \"BANC\"`", "[PAT-04] Dynamic Sliding Window"),
    ("Longest Valid Parentheses", "Find the length of the longest valid (well-formed) parentheses substring.", "`\")()())\" -> 4`", "[PAT-08] Stack or Two Pointers"),
]

# Ensure at least 4 graph problems in hard. Graph problems typically fall in: Dijkstra, Topo Sort, Word Ladder, Union Find Network, Longest Increasing Path, Alien Dictionary.
graph_hard_indices = [3, 4, 5, 6, 7, 10]

sets_md = ""

for i in range(2, 21):
    sets_md += f"\\n* * *\\n\\n## Set {i}: Timed Mock Assessment {i}\\n\\n"
    
    # Q1: Easy
    e = easy_templates[(i) % len(easy_templates)]
    # Q2: Medium
    m = medium_templates[(i) % len(medium_templates)]
    # Q3: Medium-Hard
    mh = mh_templates[(i) % len(mh_templates)]
    
    # Q4: Hard (Guarantee graph in at least sets 2,3,4,5)
    if i in [2,3,4,5]:
        h = hard_templates[graph_hard_indices[i-2]]
    else:
        h = hard_templates[(i*3) % len(hard_templates)]
        
    sets_md += f"* **Q1 (Easy): {e[0]}**\\n"
    sets_md += f"  * *Specification:* {e[1]}\\n"
    sets_md += f"  * *Sample Test Case:* Input: {e[2]}\\n"
    sets_md += f"  * *Constraints:* Array/String length $1 \\le N \\le 10^5$.\\n"
    sets_md += f"  * *Hint:* {e[3]}\\n\\n"
    
    sets_md += f"* **Q2 (Medium): {m[0]}**\\n"
    sets_md += f"  * *Specification:* {m[1]}\\n"
    sets_md += f"  * *Sample Test Case:* Input: {m[2]}\\n"
    sets_md += f"  * *Constraints:* Appropriate bounds $1 \\le N \\le 10^4$.\\n"
    sets_md += f"  * *Hint:* {m[3]}\\n\\n"
    
    sets_md += f"* **Q3 (Medium-Hard): {mh[0]}**\\n"
    sets_md += f"  * *Specification:* {mh[1]}\\n"
    sets_md += f"  * *Sample Test Case:* Input: {mh[2]}\\n"
    sets_md += f"  * *Constraints:* Bounds $1 \\le N \\le 10^5$.\\n"
    sets_md += f"  * *Hint:* {mh[3]}\\n\\n"
    
    sets_md += f"* **Q4 (Hard): {h[0]}**\\n"
    sets_md += f"  * *Specification:* {h[1]}\\n"
    sets_md += f"  * *Sample Test Case:* Input: {h[2]}\\n"
    sets_md += f"  * *Constraints:* Complexity bounds requiring optimal solution.\\n"
    sets_md += f"  * *Hint:* {h[3]}\\n\\n"

with open(r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\15-mock-assessment-sets\base.md', 'w', encoding='utf-8') as f:
    f.write(HEADER + sets_md + SURVIVAL_GUIDE)
