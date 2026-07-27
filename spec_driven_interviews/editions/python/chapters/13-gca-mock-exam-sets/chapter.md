# 20 Exam-Grade GCA Mock Problem Sets



## How to Use This Chapter

This chapter provides 20 full, four-question exam mock sets (80 problems total) modeled directly after the General Coding Assessment (GCA) blueprint. Each set is designed to simulate the rigorous 70-minute assessment environment you will face during a real coding interview. The problems strictly adhere to the expected difficulty curve: Q1 tests basic implementation and traversal (Easy, 5-8 minutes), Q2 focuses on 2D matrices and simulation (Medium, 12-15 minutes), Q3 requires algorithmic pattern recognition like HashMaps or sliding windows (Medium-Hard, 18-20 minutes), and Q4 challenges you with dynamic programming, graphs, or advanced data structures (Hard, 20-25 minutes).

To get the most out of these mock exams, strictly time yourself. Set a timer for 70 minutes and attempt all four questions in order. Do not look up syntax or external resources. If you get stuck on Q3 or Q4, practice timeboxing: move on and secure partial credit where possible. After the 70 minutes expire, review your performance. Use the provided hints to guide your post-exam study sessions, identifying which specific patterns (e.g., sliding window, BFS, monotonic stack) require further review.

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

## Set 2: String & Matrix Basics

* **Q1 (Easy): IPv4 Validation**
  * *Specification:* Write a function to determine if a given string is a valid IPv4 address. A valid IPv4 address consists of exactly four octets separated by periods. Each octet must be a numeric value between 0 and 255, inclusive. Octets cannot contain leading zeros unless the octet is exactly the single digit '0'.
  * *Sample Test Case:* Input: `"192.168.0.1"` -> Output: `true`. Input: `"192.168.01.1"` -> Output: `false`.
  * *Constraints:* String length $1 \le L \le 30$.
  * *Hint:* Split the string by periods, verify there are exactly four parts, and systematically check each part for length, digit-only composition, and numerical range.

* **Q2 (Medium): Distinct Islands**
  * *Specification:* Given a 2D binary grid where 1 represents land and 0 represents water, count the number of distinct islands. An island is formed by connected 1s (horizontally or vertically). Two islands are considered the same if one can be translated (shifted) to match the other perfectly. Rotations or reflections do not count as translations.
  * *Sample Test Case:* Input: `[[1,1,0],[1,0,0],[0,0,1]]` -> Output: `2`.
  * *Constraints:* Grid size $1 \le R, C \le 50$.
  * *Hint:* Use DFS to traverse each island, recording the path signature or relative coordinates from the starting cell to uniquely identify the island's shape.

* **Q3 (Medium-Hard): Top K Active Users**
  * *Specification:* You are given a list of log entries where each entry is a string array containing a timestamp and a userId. Your objective is to identify the most active users on the platform. Calculate the frequency of entries for each user and return the top K userIds with the highest entry counts. If there is a tie in counts, sort them lexicographically by userId.
  * *Sample Test Case:* Input: `logs = [["10:00", "user1"], ["10:05", "user2"], ["10:10", "user1"]], K = 1` -> Output: `["user1"]`.
  * *Constraints:* Number of logs $1 \le N \le 10^5$, $1 \le K \le$ unique users.
  * *Hint:* Aggregate counts using a HashMap, then use a PriorityQueue (Min-Heap) of size K or bucket sort to efficiently find the top elements.

* **Q4 (Hard): Unlimited Coin Change**
  * *Specification:* Given an integer array representing different coin denominations and an integer target representing a total monetary amount, determine the minimum number of coins needed to make up that amount. You may assume you have an infinite supply of each denomination. If the target amount cannot be met by any combination of the coins, return -1.
  * *Sample Test Case:* Input: `coins = [1, 2, 5], target = 11` -> Output: `3` (5 + 5 + 1).
  * *Constraints:* $1 \le \text{coins.length} \le 12$, $1 \le \text{coins}[i] \le 2^{31}-1$, $0 \le \text{target} \le 10^4$.
  * *Hint:* This is a classic 1D Dynamic Programming problem (unbounded knapsack). Build a `dp` array where `dp[i]` stores the minimum coins to reach amount `i`.

* * *

## Set 3: Pattern Recognition

* **Q1 (Easy): String Rotation Check**
  * *Specification:* You are given two strings, A and B. You need to verify if string B is a valid rotation of string A. A string is a rotation if it can be formed by moving some number of characters from the front of the string to the back, maintaining the order of the rest. Both strings must be of identical length to be considered valid rotations.
  * *Sample Test Case:* Input: `A = "waterbottle", B = "erbottlewat"` -> Output: `true`.
  * *Constraints:* String lengths $1 \le L \le 10^5$.
  * *Hint:* Concatenate string A with itself (`A + A`); if B is a rotation, it must exist as a contiguous substring within this doubled string.

* **Q2 (Medium): Robot Grid Simulation**
  * *Specification:* A robot starts at the origin (0,0) on a 2D Cartesian plane. It receives a sequence of movement commands represented by a string of characters: 'U' (up), 'D' (down), 'L' (left), and 'R' (right). You must simulate the robot's entire movement sequence. Return true if the robot ends up exactly back at the origin (0,0) after executing all commands, and false otherwise.
  * *Sample Test Case:* Input: `"UDLR"` -> Output: `true`. Input: `"LL"` -> Output: `false`.
  * *Constraints:* Command string length $1 \le L \le 10^4$.
  * *Hint:* Track the X and Y coordinates. Increment or decrement them based on the character, and simply check if `X == 0` and `Y == 0` at the end.

* **Q3 (Medium-Hard): Meeting Rooms Required**
  * *Specification:* You are given an array of meeting time intervals where each interval consists of a start time and an end time. Multiple meetings might overlap. You need to determine the minimum number of conference rooms required to schedule all meetings without any conflicts. A meeting ending exactly when another begins does not constitute an overlap.
  * *Sample Test Case:* Input: `[[0, 30], [5, 10], [15, 20]]` -> Output: `2`.
  * *Constraints:* Number of meetings $1 \le N \le 10^4$, $0 \le \text{start} < \text{end} \le 10^6$.
  * *Hint:* Separate the start times and end times into two sorted arrays. Use a two-pointer approach to sweep through time, incrementing room count on starts and decrementing on ends.

* **Q4 (Hard): Longest Increasing Subsequence**
  * *Specification:* Given an unsorted array of integers, locate the length of the longest strictly increasing subsequence. A subsequence is derived by deleting some or no elements from the array without altering the order of the remaining elements. The sequence must strictly increase, meaning equal values do not count as increasing.
  * *Sample Test Case:* Input: `[10, 9, 2, 5, 3, 7, 101, 18]` -> Output: `4` (The sequence is `[2, 3, 7, 101]`).
  * *Constraints:* Array length $1 \le N \le 2500$, $-10^4 \le \text{nums}[i] \le 10^4$.
  * *Hint:* While an $O(N^2)$ DP approach works for small constraints, aim for $O(N \log N)$ using an array to build the active sequence and binary search (`bisect`) to find insertion points.

* * *

## Set 4: Boundary Logic

* **Q1 (Easy): Reverse Integer Digits**
  * *Specification:* Given a signed 32-bit integer, completely reverse its digits and return the new integer. If the integer is negative, the reversed result must also remain negative. If reversing the integer causes it to overflow outside the signed 32-bit integer range $[-2^{31}, 2^{31} - 1]$, you must return 0 instead of a garbage value.
  * *Sample Test Case:* Input: `-123` -> Output: `-321`. Input: `120` -> Output: `21`.
  * *Constraints:* $-2^{31} \le N \le 2^{31} - 1$.
  * *Hint:* Extract digits using modulo 10 and build the reversed number. Check for overflow before multiplying the accumulated result by 10.

* **Q2 (Medium): Shortest Path in Binary Matrix**
  * *Specification:* You are given an $N \times N$ binary matrix where 0 represents an open, passable cell and 1 represents a blocked obstacle. Find the length of the shortest clear path from the top-left cell (0,0) to the bottom-right cell (N-1, N-1). You can move in 8 directions (horizontal, vertical, and diagonal). If no path exists, return -1.
  * *Sample Test Case:* Input: `[[0,1],[1,0]]` -> Output: `2`.
  * *Constraints:* $1 \le N \le 100$.
  * *Hint:* Breadth-First Search (BFS) is optimal for finding shortest paths in an unweighted grid. Queue coordinates and distance, marking cells visited as you enqueue them.

* **Q3 (Medium-Hard): Group Isomorphic Strings**
  * *Specification:* You are given an array of strings. Two strings are considered isomorphic if the characters in the first string can be replaced to get the second string, preserving the order and structure (e.g., "egg" and "add"). Group all mutually isomorphic strings together in lists. Return the grouped lists in any order.
  * *Sample Test Case:* Input: `["aab", "xxy", "xyz", "def"]` -> Output: `[["aab", "xxy"], ["xyz", "def"]]`.
  * *Constraints:* Array length $1 \le N \le 10^4$, String length $1 \le L \le 50$.
  * *Hint:* Normalize each string into a structural pattern (e.g., "aab" -> "1-1-2", "egg" -> "1-2-2") and use this pattern as a key in a HashMap to group matching strings.

* **Q4 (Hard): Decode Ways**
  * *Specification:* A message containing letters from A-Z is encoded into numbers using the mapping 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26. Given a string of digits, determine the total number of ways it can be decoded back into letters. A decoding is valid only if it maps strictly to the 1-26 range; leading zeros in groups are invalid (e.g., "06" cannot be 'F').
  * *Sample Test Case:* Input: `"226"` -> Output: `3` (Can be decoded as "BZ", "VF", or "BBF").
  * *Constraints:* String length $1 \le L \le 100$.
  * *Hint:* Use 1D Dynamic Programming. `dp[i]` is the number of ways to decode the prefix of length `i`. Look at the single digit at `i-1` and the two digits at `i-2` to update the state.

* * *

## Set 5: Simulation & State

* **Q1 (Easy): FizzBuzz Array Variant**
  * *Specification:* Implement the classic FizzBuzz game, but return the results as an array of strings from 1 to N. For multiples of 3, append "Fizz". For multiples of 5, append "Buzz". For multiples of both 3 and 5, append "FizzBuzz". For all other numbers, append the number itself as a string.
  * *Sample Test Case:* Input: `N = 15` -> Output: `[..., "13", "14", "FizzBuzz"]`.
  * *Constraints:* $1 \le N \le 10^4$.
  * *Hint:* Use conditional logic. Check divisibility by 15 (both 3 and 5) first, then 3, then 5, to avoid overriding the "FizzBuzz" condition.

* **Q2 (Medium): Game of Life Simulation**
  * *Specification:* You are given an $M \times N$ grid representing the current state of Conway's Game of Life (1 is live, 0 is dead). Compute the next state of the board simultaneously for every cell based on its 8 neighbors. A live cell with 2-3 live neighbors survives. A dead cell with exactly 3 live neighbors becomes live. All other cells die or remain dead. Do this in-place if possible.
  * *Sample Test Case:* Input: `[[0,1,0],[0,0,1],[1,1,1],[0,0,0]]` -> Output: `[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]`.
  * *Constraints:* $1 \le M, N \le 25$.
  * *Hint:* To achieve an in-place update, use intermediate states (like 2 for "was live, now dead" and 3 for "was dead, now live") so you can evaluate the original state without allocating a new matrix.

* **Q3 (Medium-Hard): Remove K Invalid Brackets**
  * *Specification:* You are given a string containing alphanumeric characters and brackets '(' and ')'. You are also given an integer K. You must remove exactly K brackets (either open or close) such that the resulting string contains valid, properly nested brackets. Find and return all unique valid string permutations that can result from this removal.
  * *Sample Test Case:* Input: `s = "()())()", K = 1` -> Output: `["(())()", "()()()"]`.
  * *Constraints:* String length $1 \le L \le 25$, $0 \le K \le L$.
  * *Hint:* Use Breadth-First Search (BFS) combined with a queue and a HashSet for deduplication. Generate all states by removing one character at a time, checking validity.

* **Q4 (Hard): Dijkstra's Shortest Path Array**
  * *Specification:* You are given a directed, weighted graph represented by an adjacency list, a total number of nodes N, and a starting node S. Calculate the shortest path distance from the starting node S to every other node in the graph. The weights are guaranteed to be non-negative. Return an array of these distances, using -1 for unreachable nodes.
  * *Sample Test Case:* Input: `N = 3, edges = [[0,1,5], [0,2,2], [2,1,1]], S = 0` -> Output: `[0, 3, 2]`.
  * *Constraints:* $1 \le N \le 1000$, edge weights $0 \le W \le 10^4$.
  * *Hint:* Implement Dijkstra's algorithm using a Priority Queue (Min-Heap). Enqueue tuples of `(current_distance, node)` and update adjacent nodes dynamically.

* * *

## Set 6: Array Manipulation

* **Q1 (Easy): Remove Duplicates In-Place**
  * *Specification:* Given a sorted array of integers, remove all duplicate elements in-place such that each unique element appears only once. The relative order of the unique elements must be kept the same. Since you cannot alter the array's physical length in all languages, place the unique items at the front and return the count of unique elements.
  * *Sample Test Case:* Input: `[1, 1, 2, 3, 3]` -> Output: `3` (Array modifies to `[1, 2, 3, ...]`).
  * *Constraints:* $1 \le N \le 3 \times 10^4$.
  * *Hint:* Use two pointers: a slow pointer to track the position of the next unique element, and a fast pointer to scan for new, unseen numbers.

* **Q2 (Medium): Matrix Flood Fill**
  * *Specification:* Implement a flood fill algorithm. You are given a 2D grid of integers representing pixel colors, a starting row and column, and a new target color. Change the color of the starting pixel and all orthogonally adjacent pixels of the *same original color* to the new target color, stopping at boundaries or different colors. Return the modified grid.
  * *Sample Test Case:* Input: `grid = [[1,1,1],[1,1,0],[1,0,1]], sr=1, sc=1, color=2` -> Output: `[[2,2,2],[2,2,0],[2,0,1]]`.
  * *Constraints:* $1 \le M, N \le 50$.
  * *Hint:* Use Depth-First Search (DFS). If the starting cell is already the target color, return immediately to prevent infinite recursion.

* **Q3 (Medium-Hard): Median of Two Sorted Arrays**
  * *Specification:* You are given two independent sorted integer arrays, A and B. Find the exact median of the combined sorted elements of both arrays. If the combined length is even, the median is the average of the two middle elements. The overall run time complexity must strictly be $O(\log (m+n))$.
  * *Sample Test Case:* Input: `A = [1, 3], B = [2]` -> Output: `2.0`.
  * *Constraints:* $0 \le \text{A.length}, \text{B.length} \le 1000$, $1 \le \text{total length} \le 2000$.
  * *Hint:* Do not merge the arrays. Use binary search on the smaller array to find an optimal partition point that splits both arrays into balanced left and right halves.

* **Q4 (Hard): Trapping Rain Water**
  * *Specification:* Given an array of non-negative integers where each value represents the elevation height of a terrain segment (width 1), compute how much total water the terrain can trap after a heavy rain. Water is trapped in valleys between higher elevation peaks on both the left and right sides.
  * *Sample Test Case:* Input: `[0,1,0,2,1,0,1,3,2,1,2,1]` -> Output: `6`.
  * *Constraints:* Array length $1 \le N \le 2 \times 10^4$.
  * *Hint:* A two-pointer approach (left and right) working inward is $O(1)$ space. Maintain `left_max` and `right_max`; process the smaller max side to guarantee valid trapping boundaries.

* * *

## Set 7: Hash & Frequency

* **Q1 (Easy): Valid Anagram Check**
  * *Specification:* Given two strings S and T, write a function to determine if T is a valid anagram of S. An anagram is formed by rearranging the exact characters of a different string, using all original characters exactly once. The strings contain only lowercase English letters.
  * *Sample Test Case:* Input: `S = "listen", T = "silent"` -> Output: `true`.
  * *Constraints:* String lengths $1 \le L \le 5 \times 10^4$.
  * *Hint:* Use a fixed-size integer array of length 26 to tally character counts. Increment for S and decrement for T, verifying all counts are zero at the end.

* **Q2 (Medium): Anti-Diagonal Matrix Traversal**
  * *Specification:* You are given an $N \times N$ square matrix of integers. Return an array containing all the elements of the matrix ordered by their anti-diagonals, starting from the top-left corner and sweeping down towards the bottom-right. Elements on the same anti-diagonal share the same row and column index sum.
  * *Sample Test Case:* Input: `[[1,2,3],[4,5,6],[7,8,9]]` -> Output: `[1, 2, 4, 3, 5, 7, 6, 8, 9]`.
  * *Constraints:* $1 \le N \le 100$.
  * *Hint:* Map elements using their coordinates. The key observation is that for any cell `[r][c]`, the sum `r + c` uniquely identifies which anti-diagonal line it belongs to.

* **Q3 (Medium-Hard): Maximum K-Window Sum**
  * *Specification:* You are given an array of integers and an integer K. Find the maximum possible sum of any contiguous subarray of size exactly K. The array can contain negative numbers. You must process this efficiently without recalculating the sum of elements from scratch for every possible window position.
  * *Sample Test Case:* Input: `nums = [1, 4, 2, 10, 2, 3, 1, 0, 20], K = 4` -> Output: `24` (Subarray `[3, 1, 0, 20]`).
  * *Constraints:* $1 \le N \le 10^5$, $1 \le K \le N$.
  * *Hint:* Use the Sliding Window pattern. Compute the sum of the first K elements, then iterate by adding the new element entering the window and subtracting the element leaving.

* **Q4 (Hard): LRU Cache Implementation**
  * *Specification:* Design and implement a data structure for a Least Recently Used (LRU) cache. It must support `get(key)` which returns the value if it exists (else -1), and `put(key, value)` which updates or inserts the value. If inserting exceeds the capacity, it must evict the least recently used key. Both operations must run in $O(1)$ average time complexity.
  * *Sample Test Case:* Input: `capacity = 2; put(1,1); put(2,2); get(1) -> 1; put(3,3); get(2) -> -1`.
  * *Constraints:* Capacity $1 \le C \le 3000$, up to $10^5$ calls made.
  * *Hint:* Combine a standard HashMap (for $O(1)$ key lookup) with a Doubly Linked List (for $O(1)$ node relocation to track most/least recently used order).

* * *

## Set 8: Window & Range

* **Q1 (Easy): Count Palindromic Substrings**
  * *Specification:* Given a string, determine the total number of substrings that are palindromes. A palindrome reads the same forwards and backwards. Note that a single character is mathematically considered a valid palindrome. Distinct substrings with identical characters at different indices are counted separately.
  * *Sample Test Case:* Input: `"abc"` -> Output: `3` ("a", "b", "c"). Input: `"aaa"` -> Output: `6` ("a","a","a","aa","aa","aaa").
  * *Constraints:* String length $1 \le L \le 1000$.
  * *Hint:* Iterate through each character and expand outwards from the center. Handle both odd-length (single character center) and even-length (two character center) palindromes.

* **Q2 (Medium): Rotting Oranges BFS**
  * *Specification:* You are given an $M \times N$ grid containing 0 (empty), 1 (fresh orange), or 2 (rotten orange). Every minute, any fresh orange adjacent (4-directionally) to a rotten orange also becomes rotten. Calculate the minimum number of minutes required until no fresh oranges remain. If it is impossible to rot all oranges, return -1.
  * *Sample Test Case:* Input: `[[2,1,1],[1,1,0],[0,1,1]]` -> Output: `4`.
  * *Constraints:* $1 \le M, N \le 10$.
  * *Hint:* This is a multi-source Breadth-First Search. Enqueue all initially rotten oranges at minute 0, then process level by level, decrementing a total fresh count.

* **Q3 (Medium-Hard): Smallest Subarray Sum Target**
  * *Specification:* Given an array of positive integers and a target positive integer, find the minimal length of a contiguous subarray whose sum is strictly greater than or equal to the target. If no such subarray exists that meets the target sum constraint, return 0.
  * *Sample Test Case:* Input: `target = 7, nums = [2,3,1,2,4,3]` -> Output: `2` (Subarray `[4,3]`).
  * *Constraints:* $1 \le N \le 10^5$, array values and target up to $10^9$.
  * *Hint:* Employ a dynamic sliding window. Expand the right pointer to accumulate the sum, and shrink the left pointer as long as the sum remains $\ge$ target, recording the minimum length.

* **Q4 (Hard): Course Schedule Topology**
  * *Specification:* You are given a total number of courses labeled from 0 to N-1, and an array of prerequisite pairs where `[A, B]` means course B must be completed before course A. Determine if it is mathematically possible to finish all courses. Return a valid chronological ordering of courses. If a cycle exists making it impossible, return an empty array.
  * *Sample Test Case:* Input: `N = 2, pre = [[1,0]]` -> Output: `[0, 1]`.
  * *Constraints:* $1 \le N \le 2000$.
  * *Hint:* This requires Topological Sorting. Build a directed graph and an in-degree array. Use Kahn's algorithm (BFS queue) to process nodes with zero dependencies.

* * *

## Set 9: Two-Pointer Mastery

* **Q1 (Easy): Move Targets to End**
  * *Specification:* Given an integer array and a specific target value, move all occurrences of that target to the end of the array while maintaining the relative ordering of the other non-target elements. You must perform this mutation in-place without allocating a duplicate array structure.
  * *Sample Test Case:* Input: `nums = [0,1,0,3,12], target = 0` -> Output: `[1,3,12,0,0]`.
  * *Constraints:* $1 \le N \le 10^4$.
  * *Hint:* Maintain an insertion index pointer. Iterate through the array; if the current element is not the target, swap it with the element at the insertion index and increment the index.

* **Q2 (Medium): Kth Smallest in Sorted Matrix**
  * *Specification:* You are given an $N \times N$ matrix where every row and every column is independently sorted in ascending order. Find the Kth smallest integer in the entire matrix. Note that it is the Kth smallest in global sorted order, not the Kth distinct element.
  * *Sample Test Case:* Input: `matrix = [[1,5,9],[10,11,13],[12,13,15]], K = 8` -> Output: `13`.
  * *Constraints:* $1 \le N \le 300$, $1 \le K \le N^2$.
  * *Hint:* Since rows and columns are sorted, use a Min-Heap starting with the first element of each row, or apply a clever Binary Search over the value range tracking counts.

* **Q3 (Medium-Hard): 3Sum Zero Target**
  * *Specification:* Given an integer array, identify and return all unique triplets `[nums[i], nums[j], nums[k]]` such that their sum equals exactly zero. The index of each element must be distinct. The output array of triplets must not contain any duplicate triplet combinations, regardless of internal ordering.
  * *Sample Test Case:* Input: `[-1,0,1,2,-1,-4]` -> Output: `[[-1,-1,2],[-1,0,1]]`.
  * *Constraints:* $3 \le N \le 3000$.
  * *Hint:* Sort the array first. Iterate through the array fixing one number, then use a two-pointer approach (left and right) on the remaining suffix to find pairs that sum to the inverse of the fixed number.

* **Q4 (Hard): Longest Valid Parentheses**
  * *Specification:* Given a string composed strictly of '(' and ')' characters, compute the length of the longest contiguous valid (well-formed) parentheses substring. The valid substring must have properly matched and nested brackets.
  * *Sample Test Case:* Input: `")()())"` -> Output: `4` (The substring is `"()()"`).
  * *Constraints:* String length $0 \le L \le 3 \times 10^4$.
  * *Hint:* Utilize a stack storing indices. Initialize the stack with -1 to serve as a base index. On closing brackets, pop and measure the length against the new top of the stack.

* * *

## Set 10: Greedy & Optimization

* **Q1 (Easy): Single Stock Profit**
  * *Specification:* You are given an array where each element represents the price of a given stock on that day. You are permitted to complete at most one transaction (buy one share and sell one share) in the future. Calculate the maximum profit you can achieve. If no profit can be made (prices only drop), return 0.
  * *Sample Test Case:* Input: `[7,1,5,3,6,4]` -> Output: `5` (Buy at 1, sell at 6).
  * *Constraints:* $1 \le N \le 10^5$.
  * *Hint:* Iterate through the array while maintaining a running variable of the minimum price seen so far. At each day, evaluate if selling at the current price yields a new maximum profit.

* **Q2 (Medium): Unique Paths with Obstacles**
  * *Specification:* A robot is positioned at the top-left corner of an $M \times N$ grid and is trying to reach the bottom-right corner. The robot can only move down or right. Some grid cells are marked as 1, representing an impassable obstacle. Empty cells are 0. Calculate the total number of unique valid paths to the destination.
  * *Sample Test Case:* Input: `[[0,0,0],[0,1,0],[0,0,0]]` -> Output: `2`.
  * *Constraints:* $1 \le M, N \le 100$.
  * *Hint:* Use 2D Dynamic Programming. `dp[r][c]` equals `dp[r-1][c] + dp[r][c-1]`. If a cell contains an obstacle, manually set its DP value to 0 paths.

* **Q3 (Medium-Hard): Merge Overlapping Intervals**
  * *Specification:* You are given an array of intervals where each interval is represented as `[start, end]`. Multiple intervals may overlap. Consolidate all overlapping intervals into a unified, non-overlapping array of intervals that perfectly covers all time spans represented by the original input. Return the condensed array.
  * *Sample Test Case:* Input: `[[1,3],[2,6],[8,10],[15,18]]` -> Output: `[[1,6],[8,10],[15,18]]`.
  * *Constraints:* $1 \le N \le 10^4$.
  * *Hint:* Sort the intervals primarily by their starting times. Iterate and maintain a 'current' interval; merge if the next start is $\le$ the current end, updating the end to the maximum of both.

* **Q4 (Hard): Jump Game Minimum**
  * *Specification:* You are given an array of non-negative integers where each value dictates the maximum jump length you can take forward from that position. Assuming you always start at the first index, determine the absolute minimum number of jumps required to reach the last index of the array. The test cases guarantee the end is reachable.
  * *Sample Test Case:* Input: `[2,3,1,1,4]` -> Output: `2` (Jump index 0 to 1, then jump to the end).
  * *Constraints:* $1 \le N \le 10^4$.
  * *Hint:* Use a Greedy BFS approach. Maintain two variables: `current_jump_end` and `farthest_reachable`. Iterate through the array; when you hit `current_jump_end`, you must jump, so increment jump count and update the end boundary.

* * *

## Set 11: Tree & Recursion Simulation

* **Q1 (Easy): Happy Number Cycle**
  * *Specification:* Write an algorithm to verify if a number is a "happy number". To determine this, replace the number with the sum of the squares of its digits. Repeat this mathematical process until the number equals 1 (happy), or it loops endlessly in a recurring cycle that does not include 1 (unhappy). Return true if happy.
  * *Sample Test Case:* Input: `19` -> Output: `true` ($1^2 + 9^2 = 82$, $8^2 + 2^2 = 68$, etc., eventually reaching 1).
  * *Constraints:* $1 \le N \le 2^{31} - 1$.
  * *Hint:* Use a HashSet to track previously computed sums. If a sum repeats before hitting 1, you have entered an infinite cycle and can return false.

* **Q2 (Medium): Word Search Traversal**
  * *Specification:* You are given an $M \times N$ grid of characters and a target word string. Determine if the word can be constructed by traversing sequentially adjacent cells (horizontally or vertically). The same physical cell in the matrix cannot be used more than once during the construction of the word.
  * *Sample Test Case:* Input: `board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"` -> Output: `true`.
  * *Constraints:* $1 \le M, N \le 6$, word length $\le 15$.
  * *Hint:* Employ Backtracking / Depth-First Search from every matching starting letter. Temporarily mark cells as visited (e.g., changing it to '#') to prevent reuse, restoring it afterward.

* **Q3 (Medium-Hard): Nested String Reversal**
  * *Specification:* You are given a string that contains lowercase letters and properly nested parentheses pairs. You must reverse the exact character sequence present within each pair of matching parentheses, starting from the innermost pair and working outward. After processing, return the final evaluated string without any parenthesis characters.
  * *Sample Test Case:* Input: `"(ed(et(oc))el)"` -> Output: `"leetcode"`.
  * *Constraints:* String length $1 \le L \le 2000$.
  * *Hint:* Use a Stack. Push characters one by one. When encountering a closing parenthesis, pop characters until the opening parenthesis is found, reverse that temporary chunk, and push it back to the stack.

* **Q4 (Hard): Minimum Edit Distance**
  * *Specification:* Given two strings `word1` and `word2`, compute the absolute minimum number of discrete operations required to mutate `word1` into `word2`. You are granted three valid operations: insert a character, delete a character, or replace a character. Each operation has a uniform cost of 1.
  * *Sample Test Case:* Input: `word1 = "horse", word2 = "ros"` -> Output: `3` (Replace 'h' with 'r', remove 'r', remove 'e').
  * *Constraints:* String lengths $0 \le L \le 500$.
  * *Hint:* Apply 2D Dynamic Programming. `dp[i][j]` tracks the cost to match prefixes of length `i` and `j`. If characters differ, take the minimum of insertion, deletion, or substitution plus 1.

* * *

## Set 12: Bit & Math Tricks

* **Q1 (Easy): Set Bit Count**
  * *Specification:* Write a function that takes an unsigned 32-bit integer as input and computes the total number of '1' bits (also known as the Hamming weight) present in its binary representation. Return the final integer tally.
  * *Sample Test Case:* Input: `11` (binary `1011`) -> Output: `3`.
  * *Constraints:* Input is a 32-bit unsigned integer.
  * *Hint:* Use bitwise AND operations. The expression `n & (n - 1)` always elegantly flips the least significant '1' bit of `n` to '0', allowing for rapid counting.

* **Q2 (Medium): Generate Spiral Matrix**
  * *Specification:* Given a positive integer N, mathematically construct and return an $N \times N$ matrix populated with sequential numerical elements from $1$ up to $N^2$ in a clockwise spiral layout. The sequence must originate at the top-left corner and spiral inward.
  * *Sample Test Case:* Input: `3` -> Output: `[[1,2,3],[8,9,4],[7,6,5]]`.
  * *Constraints:* $1 \le N \le 20$.
  * *Hint:* Establish four boundary variables: top, bottom, left, and right. Use a `while` loop containing four independent `for` loops to walk the perimeter, incrementally shrinking the boundaries inward.

* **Q3 (Medium-Hard): Constant Space First Duplicate**
  * *Specification:* You are given an array of integers containing N elements, where every element guarantees a value in the inclusive range of $[1, N]$. Exactly one integer value is duplicated once, while all others appear exactly once. Identify and return this duplicate integer. Your solution must use strictly $O(1)$ auxiliary space and must not modify the input array.
  * *Sample Test Case:* Input: `[1,3,4,2,2]` -> Output: `2`.
  * *Constraints:* $1 \le N \le 10^5$.
  * *Hint:* Treat the array values as pointer links to other indices (`next = nums[curr]`). Utilize Floyd's Tortoise and Hare cycle detection algorithm to locate the cycle entrance, which corresponds to the duplicate value.

* **Q4 (Hard): Maximum Product Subarray**
  * *Specification:* Given an integer array, find a contiguous non-empty subarray that evaluates to the absolute largest mathematical product, and return that computed product. Pay special attention to negative values, as multiplying two negatives yields a large positive.
  * *Sample Test Case:* Input: `[2,3,-2,4]` -> Output: `6` (Subarray `[2,3]`).
  * *Constraints:* $1 \le N \le 2 \times 10^4$.
  * *Hint:* Maintain both a `current_max` and a `current_min` product dynamically. When encountering a negative number, swap the max and min trackers because multiplying by a negative inverts the relationship.

* * *

## Set 13: Advanced Search

* **Q1 (Easy): Validate Sudoku Subsections**
  * *Specification:* You are given a $9 \times 9$ Sudoku board partially populated with digits '1'-'9' and empty spaces '.'. Determine if the board's current state is legally valid. A valid board requires that no row, no column, and no $3 \times 3$ sub-box contains any duplicate digits. You do not need to determine if it is solvable, only if it is currently valid.
  * *Sample Test Case:* Input: standard valid grid -> Output: `true`.
  * *Constraints:* Board size is exactly $9 \times 9$.
  * *Hint:* Iterate through every cell. Use three separate HashSets (or a smart string encoding trick) to register seen values mapped to specific row indices, column indices, and calculated box indices.

* **Q2 (Medium): Compute Island Perimeter**
  * *Specification:* You are given a grid of 1s (land) and 0s (water) containing exactly one unified island (one or more connected land cells). No lakes exist inside the island. Calculate the exact total perimeter length of the island. Each land cell contributes exactly 4 to the perimeter unless it touches an adjacent land cell.
  * *Sample Test Case:* Input: `[[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]` -> Output: `16`.
  * *Constraints:* $1 \le M, N \le 100$.
  * *Hint:* Loop over the grid. Every time you see a 1, add 4 to the perimeter total. Then, check the left neighbor and top neighbor; if they are also land, subtract 2 for each shared edge.

* **Q3 (Medium-Hard): Longest Consecutive Sequence**
  * *Specification:* Given an unsorted array of integers, determine the length of the longest sequence of mathematically consecutive elements. The elements do not need to be contiguous in the original array structure. You must construct an algorithm that achieves strictly $O(N)$ runtime complexity.
  * *Sample Test Case:* Input: `[100, 4, 200, 1, 3, 2]` -> Output: `4` (The sequence `[1, 2, 3, 4]`).
  * *Constraints:* $0 \le N \le 10^5$.
  * *Hint:* Dump all elements into a HashSet for $O(1)$ lookups. Iterate through the set; only begin building a sequence if `num - 1` does not exist, ensuring you only iterate starting from the true base of a sequence.

* **Q4 (Hard): Sorted 2D Binary Search**
  * *Specification:* Implement an efficient algorithm that searches for a specific integer target within an $M \times N$ matrix. The matrix has strict properties: every row is sorted in strict ascending order, and the absolute first integer of any row is strictly greater than the absolute last integer of the preceding row. Return true if found, false otherwise.
  * *Sample Test Case:* Input: `matrix = [[1,3,5,7],[10,11,16,20]], target = 3` -> Output: `true`.
  * *Constraints:* $1 \le M, N \le 100$.
  * *Hint:* Mathematically treat the 2D matrix as a flattened 1D sorted array of length $M \times N$. Map a 1D mid-index back to 2D coordinates using division `mid / N` and modulo `mid % N`.

* * *

## Set 14: Interval & Schedule

* **Q1 (Easy): Max Guests Present**
  * *Specification:* You are given two arrays of equal length representing an event schedule: one array indicates guest arrival times and the other indicates guest departure times. Guests depart at the exact minute listed. Compute the maximum number of concurrent guests present at the venue at any single point in time.
  * *Sample Test Case:* Input: `arrivals = [1, 2, 9], departures = [5, 8, 12]` -> Output: `2`.
  * *Constraints:* $1 \le N \le 10^5$.
  * *Hint:* Sort both the arrival and departure arrays independently. Use a two-pointer approach to sweep a timeline, incrementing a counter for arrivals and decrementing for departures.

* **Q2 (Medium): Snake Collision Simulation**
  * *Specification:* Simulate a simplified game of Snake on an unbounded 2D grid. The snake begins at (0,0) with length 1. You receive a sequence of directional commands. For every command, the snake's head moves 1 unit. Without food to grow, the snake's tail moves identically, maintaining its overall length (given as a fixed integer L). Determine if the snake ever collides with its own body.
  * *Sample Test Case:* Input: `L=4, path="RRRDDLLU"` -> Output: `true`.
  * *Constraints:* Path length $1 \le P \le 10^4$.
  * *Hint:* Utilize a Deque (Double-ended Queue) to maintain the exact coordinates of the snake's body segments. Push the new head coordinate and pop the tail, checking a HashSet to detect overlap.

* **Q3 (Medium-Hard): Two Sum Indices**
  * *Specification:* Given an array of integers and a specific target integer, discover two unique indices pointing to array values that sum exactly to the target. It is mathematically guaranteed that exactly one valid solution exists. You may not use the same element twice. Return the two indices in any order.
  * *Sample Test Case:* Input: `nums = [2,7,11,15], target = 9` -> Output: `[0, 1]`.
  * *Constraints:* $2 \le N \le 10^4$.
  * *Hint:* Make a single pass over the array. Use a HashMap to store the required complement (`target - current_val`) and the current index as you traverse.

* **Q4 (Hard): Subset Partition Minimum Difference**
  * *Specification:* You are provided a list of positive integers. Your objective is to partition the list into two disjoint, exhaustive subsets such that the absolute difference between the mathematical sums of the subsets is minimized. Return the minimum possible absolute difference.
  * *Sample Test Case:* Input: `[1, 6, 11, 5]` -> Output: `1` (Subsets `[1,5,6]` summing to 12, and `[11]`).
  * *Constraints:* $1 \le N \le 200$.
  * *Hint:* This maps directly to the 0/1 Knapsack DP problem. Target a capacity equal to `total_sum // 2` to find the largest subset sum possible without exceeding half, which minimizes the gap.

* * *

## Set 15: Stack & Queue Patterns

* **Q1 (Easy): Evaluate Reverse Polish Notation**
  * *Specification:* You are given an array of string tokens representing an arithmetic expression formatted in Reverse Polish Notation (postfix). Evaluate the expression mathematically and return the resulting integer. The valid operators are `+`, `-`, `*`, and `/`. Division between two integers should truncate strictly towards zero.
  * *Sample Test Case:* Input: `["2", "1", "+", "3", "*"]` -> Output: `9` ((2 + 1) * 3).
  * *Constraints:* Array length $1 \le N \le 10^4$.
  * *Hint:* Use a Stack. Iterate over tokens; push numbers to the stack. When an operator is encountered, pop the top two numbers, apply the operation, and push the result back.

* **Q2 (Medium): Maximal Square Grid**
  * *Specification:* You are given an $M \times N$ binary matrix containing only 0s and 1s. Locate the largest contiguous square subsection of the grid consisting entirely of 1s, and compute its total structural area. Return the area as an integer.
  * *Sample Test Case:* Input: `[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]` -> Output: `4`.
  * *Constraints:* $1 \le M, N \le 300$.
  * *Hint:* Use DP. Let `dp[r][c]` be the side length of the largest square whose bottom-right corner is at `[r][c]`. It equals the minimum of its top, left, and top-left neighbors plus 1.

* **Q3 (Medium-Hard): Next Greater Element**
  * *Specification:* You are given an array of integers. For every element in the array, find its "Next Greater Element"—defined as the first sequentially upcoming element to its right that is strictly mathematically larger than it. Output a corresponding array of these larger elements. If an element has no larger successor, assign it -1.
  * *Sample Test Case:* Input: `[4, 1, 2, 3]` -> Output: `[-1, 2, 3, -1]`.
  * *Constraints:* $1 \le N \le 10^5$.
  * *Hint:* Utilize a Monotonic Decreasing Stack. Store values (or indices) in the stack. When the current element is larger than the top of the stack, pop the stack and assign the current element as the answer for the popped item.

* **Q4 (Hard): Shortest Word Transformation**
  * *Specification:* You are given a start word, an end word, and a dictionary of valid words. Find the exact sequence length of the shortest transformation sequence from the start word to the end word. A valid step changes precisely one single letter, and every intermediate state must exist in the dictionary. Return 0 if no sequence exists.
  * *Sample Test Case:* Input: `begin = "hit", end = "cog", dict = ["hot","dot","dog","lot","log","cog"]` -> Output: `5` ("hit"->"hot"->"dot"->"dog"->"cog").
  * *Constraints:* Dictionary size $\le 5000$.
  * *Hint:* Construct this as a graph problem and utilize Breadth-First Search (BFS) to guarantee the shortest path. For each word, generate all 1-character edits and look them up in a HashSet dictionary.

* * *

## Set 16: Graph Exploration

* **Q1 (Easy): Balanced Bracket String**
  * *Specification:* Given a string containing only the characters `(`, `)`, `{`, `}`, `[`, and `]`, evaluate if the string layout is structurally balanced. A string is valid if every open bracket is closed by the exact identical type of bracket, and they are closed in the exact correct nested mathematical order.
  * *Sample Test Case:* Input: `"{[]}"` -> Output: `true`. Input: `"([)]"` -> Output: `false`.
  * *Constraints:* String length $1 \le L \le 10^4$.
  * *Hint:* Maintain a Stack. Push opening brackets. For closing brackets, check if the stack is non-empty and the top element is the matching opening bracket, then pop.

* **Q2 (Medium): Deep Clone Graph**
  * *Specification:* You are provided a reference node to a connected, undirected graph. The nodes contain a value and a list of neighbor references. Return a complete deep copy (clone) of the entire graph structure. Every newly generated node must mirror the original topology, but occupy separate heap memory.
  * *Sample Test Case:* Input: `adjList = [[2,4],[1,3],[2,4],[1,3]]` -> Output: Exact identical structural clone.
  * *Constraints:* Node count $\le 100$.
  * *Hint:* Traverse using DFS or BFS while keeping a central HashMap mapping original nodes to their cloned equivalents to prevent duplicating already processed nodes and resolve cycles.

* **Q3 (Medium-Hard): Task Scheduler Cooldown**
  * *Specification:* You are given an array of characters representing CPU tasks and an integer cooldown multiplier N. Identical tasks must be separated by at least N time intervals of cooldown. You can process a task or sit idle in one time interval. Calculate the absolute minimum time required to process every task in the array.
  * *Sample Test Case:* Input: `tasks = ["A","A","A","B","B","B"], N = 2` -> Output: `8` (A -> B -> idle -> A -> B -> idle -> A -> B).
  * *Constraints:* Task array length $\le 10^4$, $0 \le N \le 100$.
  * *Hint:* Frequency matters most. Calculate the frequency of the most common task. Construct hypothetical "blocks" separated by the cooldown period, and fill in the idle gaps with remaining tasks.

* **Q4 (Hard): Bipartite Graph Detection**
  * *Specification:* You are given an undirected graph presented as an adjacency list. Determine if the graph is bipartite. A graph qualifies as bipartite if you can split its entire set of nodes into two disjoint independent sets, such that every edge strictly connects a node in the first set to a node in the second set, with no internal edges within a set.
  * *Sample Test Case:* Input: `graph = [[1,3],[0,2],[1,3],[0,2]]` -> Output: `true`.
  * *Constraints:* $1 \le N \le 100$.
  * *Hint:* Perform a Graph Coloring algorithm using BFS or DFS. Assign alternating colors (e.g., 0 and 1) to adjacent nodes. If you ever hit an adjacent node that already possesses the current node's color, it is not bipartite.

* * *

## Set 17: Dynamic Programming Gauntlet

* **Q1 (Easy): Distinct Climbing Stairs**
  * *Specification:* You are faced with a staircase containing N discrete steps to reach the top level. At any point, you are mathematically permitted to climb either exactly 1 step or exactly 2 steps. Compute the total number of unique sequences you can perform to reach the ultimate top step.
  * *Sample Test Case:* Input: `3` -> Output: `3` (1+1+1, 1+2, 2+1).
  * *Constraints:* $1 \le N \le 45$.
  * *Hint:* This maps exactly to the Fibonacci sequence. The number of ways to reach step `N` is strictly the sum of the ways to reach step `N-1` and step `N-2`.

* **Q2 (Medium): Pacific Atlantic Convergence**
  * *Specification:* You are given an $M \times N$ rectangular grid containing integer elevations. The Pacific Ocean touches the top and left borders, while the Atlantic touches the bottom and right. Water naturally flows downwards or straight across to adjacent cells of equal or lower elevation. Return a list of all cell coordinates where rainwater can successfully flow to both the Pacific and Atlantic oceans.
  * *Sample Test Case:* Input: `[[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]` -> Output: `[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]`.
  * *Constraints:* $1 \le M, N \le 200$.
  * *Hint:* Trace the flow backwards. Start DFS/BFS algorithms from the ocean borders, climbing strictly *up* in elevation, and record reachable cells in two separate HashSets. Find the intersection.

* **Q3 (Medium-Hard): Longest Palindrome Builder**
  * *Specification:* You are given a string. Your task is to mathematically compute the absolute length of the longest palindrome that could be constructed using the letters found within the string. You are allowed to dynamically reorder the letters in any configuration to build this palindrome.
  * *Sample Test Case:* Input: `"abccccdd"` -> Output: `7` (Can build "dccaccd").
  * *Constraints:* String length $1 \le L \le 2000$.
  * *Hint:* Count the frequencies of all characters. Pairs of characters can always be mirrored on both sides of a palindrome. If any odd character counts exist, you can add exactly one character to the center.

* **Q4 (Hard): Regular Expression Wildcards**
  * *Specification:* Given an input string `s` and a pattern string `p`, implement full regular expression matching that strictly supports the `.` wildcard (matches any single character) and the `*` wildcard (matches zero or more of the immediately preceding element). The match must cover the entire input string perfectly, not just a partial substring.
  * *Sample Test Case:* Input: `s = "aa", p = "a*"` -> Output: `true`.
  * *Constraints:* String lengths $1 \le L \le 20$.
  * *Hint:* Use 2D Dynamic Programming where `dp[i][j]` implies prefix match. Dealing with `*` requires evaluating two branches: treating the `*` component as matching zero elements, or matching one/more elements if the previous character aligns.

* * *

## Set 18: Binary Search Variants

* **Q1 (Easy): Find Peak Topology**
  * *Specification:* You are given a 0-indexed integer array where adjacent elements are strictly forbidden from being identically equal. A peak element is defined as any value strictly greater than both its left and right neighbors. Find any peak element in the array and return its exact index. You must achieve $O(\log N)$ time complexity.
  * *Sample Test Case:* Input: `[1,2,3,1]` -> Output: `2` (Index of value 3).
  * *Constraints:* $1 \le N \le 1000$.
  * *Hint:* Use Binary Search. Calculate `mid`. If `nums[mid] < nums[mid+1]`, the upward slope guarantees a peak exists to the right. Otherwise, a peak exists to the left (or is the mid itself).

* **Q2 (Medium): Minimum of Rotated Array**
  * *Specification:* An initially sorted array of unique integers was rotated cyclically an unknown number of times. Given this rotated array, locate the minimum mathematical element. You must implement an algorithm that operates in $O(\log N)$ runtime complexity.
  * *Sample Test Case:* Input: `[3,4,5,1,2]` -> Output: `1`.
  * *Constraints:* $1 \le N \le 5000$.
  * *Hint:* Use Binary Search. Compare `nums[mid]` to `nums[right]`. If `mid` is greater, the minimum is wrapped in the right half. If `mid` is smaller, the minimum is in the left half (inclusive of mid).

* **Q3 (Medium-Hard): Koko Banana Minimum Speed**
  * *Specification:* Koko has N piles of bananas, with array values representing pile sizes. She has H hours to eat them all. She chooses a constant eating speed K bananas per hour. If a pile has less than K, she finishes it but idles for the remainder of that hour. Compute the absolute minimum integer speed K needed to finish all piles strictly within H hours.
  * *Sample Test Case:* Input: `piles = [3,6,7,11], H = 8` -> Output: `4`.
  * *Constraints:* $1 \le N \le 10^4$.
  * *Hint:* Binary search on the answer space (the eating speed). The possible speeds range from 1 to the max pile size. For each guessed speed `mid`, calculate total hours required and adjust the search space.

* **Q4 (Hard): Median of Sorted Arrays Revision**
  * *Specification:* (A variant of the classic for repetition reinforcement) Given two separate sorted arrays of varying lengths, isolate the exact median value in $O(\log(\text{min}(m,n)))$ time complexity. The arrays are independently sorted but may drastically differ in scale and capacity. Return the floating point median.
  * *Sample Test Case:* Input: `A=[1,2], B=[3,4]` -> Output: `2.5`.
  * *Constraints:* Total length $\le 2000$.
  * *Hint:* Enforce that array A is the shorter one. Perform binary search over the indices of A to find a partition line, deriving B's partition line logically. Cross-check the boundary max/min values for validity.

* * *

## Set 19: Advanced Data Structures

* **Q1 (Easy): Min-Tracking Stack**
  * *Specification:* Design a customized Stack data structure that supports standard operations: `push(val)`, `pop()`, `top()`, and an additional method `getMin()`. The `getMin` method must dynamically retrieve the absolute minimum element currently present anywhere in the stack. All four operations must resolve in $O(1)$ constant time complexity.
  * *Sample Test Case:* Input: `push(-2); push(0); push(-3); getMin() -> -3; pop(); top() -> 0; getMin() -> -2`.
  * *Constraints:* Value range bounds are standard 32-bit integers.
  * *Hint:* Track minimums using a parallel stack structure or by pushing tuples of `(value, current_minimum)` onto a single backing stack.

* **Q2 (Medium): Capture Surrounded Regions**
  * *Specification:* You are given an $M \times N$ board heavily populated with 'X' and 'O'. A region of 'O's is considered completely surrounded if it is enclosed by 'X's on all four cardinal directions. Capture these isolated regions by permanently flipping all surrounded 'O's into 'X's in-place. 'O's connected directly to the board's outer perimeter cannot be captured.
  * *Sample Test Case:* Input: `[["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]` -> Output: `[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]`.
  * *Constraints:* $1 \le M, N \le 200$.
  * *Hint:* Launch a DFS from all 'O's explicitly located on the outer border, marking them safely untouchable. Then systematically flip all remaining, unmarked 'O's in the interior to 'X'.

* **Q3 (Medium-Hard): Randomized Const-Time Structure**
  * *Specification:* Design a data structure framework that correctly supports inserting a value, removing a value, and obtaining a random value from the current pool. All three operations must perform strictly in $O(1)$ average mathematical time complexity. The random getter must return values with uniform independent probability based on the current size.
  * *Sample Test Case:* Input: `insert(1) -> true, insert(2) -> true, getRandom() -> 1 or 2, remove(1) -> true`.
  * *Constraints:* Up to $2 \times 10^5$ calls.
  * *Hint:* Combine a standard Array (for $O(1)$ random access by index) with a HashMap (to map values to their current array indices). To remove in $O(1)$, swap the target element with the absolute last element in the array, update the map, and pop.

* **Q4 (Hard): Binary Tree Serializer**
  * *Specification:* Design a robust algorithm to heavily serialize and subsequently deserialize a full binary tree. Serialization involves converting the in-memory tree object structure into a single continuous string format. Deserialization involves parsing that precise string back into the exact original structural binary tree hierarchy.
  * *Sample Test Case:* Input: `root = [1,2,3,null,null,4,5]` -> Output: `[1,2,3,null,null,4,5]` (After round-trip processing).
  * *Constraints:* Tree nodes up to $10^4$.
  * *Hint:* Utilize a preorder traversal (DFS) or level-order traversal (BFS). Append distinct null markers (e.g., 'N' or '#') for missing children to preserve structural integrity, and separate node values using commas.

* * *

## Set 20: Championship Round

* **Q1 (Easy): Roman Numeral Parser**
  * *Specification:* Given a string strictly representing a valid Roman numeral, parse it and compute its equivalent standard integer value. Roman numerals are constructed from seven distinct symbols: I, V, X, L, C, D, and M. The parsing must accurately support standard subtractive combinations like IV (4) and IX (9).
  * *Sample Test Case:* Input: `"MCMXCIV"` -> Output: `1994`.
  * *Constraints:* String length $1 \le L \le 15$.
  * *Hint:* Iterate strictly from left to right. If the numerical value of the current symbol is less than the numerical value of the right adjacent symbol, mathematically subtract it; otherwise, add it.

* **Q2 (Medium): Battleship Counter**
  * *Specification:* You are given an $M \times N$ matrix board modeling a game of Battleships, populated with 'X' (ship) and '.' (water) cells. Distinct battleships are constrained to strictly horizontal or strictly vertical linear orientations. Ships are guaranteed to be separated by at least one cell of water. Tally and return the total number of independent battleships.
  * *Sample Test Case:* Input: `[["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]` -> Output: `2`.
  * *Constraints:* $1 \le M, N \le 200$.
  * *Hint:* Instead of complex graph traversal, loop across all cells. A cell is the definitive "top-left head" of a battleship if it is an 'X' AND has no 'X' immediately above it AND no 'X' immediately to its left.

* **Q3 (Medium-Hard): Target Sum Subarray Count**
  * *Specification:* Given an array of unconstrained integers (including negatives) and an integer K, mathematically determine the absolute total number of continuous subarrays whose constituent sum resolves precisely to K. Note that different indices dictate a distinct subarray, even if values match.
  * *Sample Test Case:* Input: `nums = [1,2,3], K = 3` -> Output: `2` (`[1,2]` and `[3]`).
  * *Constraints:* $1 \le N \le 2 \times 10^4$.
  * *Hint:* Utilize the Prefix Sum architectural pattern alongside a HashMap. As you continuously iterate, record the frequency of each prefix sum encountered. At each step, check if `current_sum - K` exists in the map to identify valid sequences ending at the current index.

* **Q4 (Hard): Burst Balloons Optimization**
  * *Specification:* You are given N balloons, each painted with a specific integer value. If you deliberately burst balloon $i$, you immediately collect coins mathematically equal to `nums[i-1] * nums[i] * nums[i+1]`. After bursting, the adjacent balloons immediately shift together. Determine the maximum possible coin yield you can extract by strategically sequencing the bursts until zero balloons remain.
  * *Sample Test Case:* Input: `[3,1,5,8]` -> Output: `167`.
  * *Constraints:* $1 \le N \le 300$.
  * *Hint:* Pad the array boundaries with conceptual 1s. Frame the DP inversely: analyze which balloon should strictly be the *absolute last* one burst in a given range, rather than the first. Build answers incrementally for sub-intervals.
