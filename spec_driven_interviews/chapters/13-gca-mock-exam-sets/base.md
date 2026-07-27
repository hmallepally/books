# 20 Exam-Grade GCA Mock Problem Sets

## How to Use This Chapter

This chapter provides 20 full, four-question exam mock sets (80 problems total) modeled directly after the General Coding Assessment (GCA) blueprint. Each set is designed to simulate the rigorous 70-minute assessment environment you will face during a real coding interview. The problems strictly adhere to the expected difficulty curve: Q1 tests basic implementation and traversal (Easy, 5-8 minutes), Q2 focuses on 2D matrices and simulation (Medium, 12-15 minutes), Q3 requires algorithmic pattern recognition like HashMaps or sliding windows (Medium-Hard, 18-20 minutes), and Q4 challenges you with dynamic programming, graphs, or advanced data structures (Hard, 20-25 minutes).

To get the most out of these mock exams, strictly time yourself. Set a timer for 70 minutes and attempt all four questions in order. Do not look up syntax or external resources. If you get stuck on Q3 or Q4, practice timeboxing: move on and secure partial credit where possible. After the 70 minutes expire, review your performance. Use the provided hints to guide your post-exam study sessions, identifying which specific patterns (e.g., sliding window, BFS, monotonic stack) require further review.

Remember, there is no code in this chapter—this is your practice arena. Read the specifications, analyze the test cases, check the constraints, and write your own optimal solutions.

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
