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

![Multi-Source BFS — Rotting Oranges Wavefront](visuals/bfs_grid_levels.png){width=85%}

## Reusable Code Templates

### Template A: Spiral Boundary Traversal
```csharp
int top = 0, bottom = matrix.Length - 1;
int left = 0, right = matrix[0].Length - 1;
while (top <= bottom && left <= right) {
  for (int j = left; j <= right; j++) { /* process matrix[top][j] */ }
  top++;
  for (int i = top; i <= bottom; i++) { /* process matrix[i][right] */ }
  right--;
  if (top <= bottom) {
    for (int j = right; j >= left; j--) { /* process matrix[bottom][j] */ }
    bottom--;
  }
  if (left <= right) {
    for (int i = bottom; i >= top; i--) { /* process matrix[i][left] */ }
    left++;
  }
}
```
![Spiral Boundary Traversal — Layer-by-Layer Contraction](visuals/spiral_traversal.png){width=85%}

### Template B: 4-Directional BFS/DFS Grid Walk
```csharp
int[] dr = {-1, 1, 0, 0};
int[] dc = {0, 0, -1, 1};

void Dfs(int[][] grid, int r, int c) {
  if (r < 0 || r >= grid.Length || c < 0 || c >= grid[0].Length || grid[r][c] == -1) return;
  grid[r][c] = -1; // mark visited
  for (int i = 0; i < 4; i++) {
    Dfs(grid, r + dr[i], c + dc[i]);
  }
}
```
### Template C: 2D Prefix Sum Construction + Query
```csharp
// Construction
int[,] sum = new int[R + 1, C + 1];
for (int r = 1; r <= R; r++) {
  for (int c = 1; c <= C; c++) {
    sum[r, c] = matrix[r-1][c-1] + sum[r-1, c] + sum[r, c-1] - sum[r-1, c-1];
  }
}
// Query from (r1, c1) to (r2, c2)
int Query(int r1, int c1, int r2, int c2) {
  return sum[r2+1, c2+1] - sum[r1, c2+1] - sum[r2+1, c1] + sum[r1, c1];
}
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

$$S[r][c] = A[r\text{-}1][c\text{-}1] + S[r\text{-}1][c] + S[r][c\text{-}1] - S[r\text{-}1][c\text{-}1]$$

$$S[2][2] = \underbrace{A[1][1]}_{5} + \underbrace{S[1][2]}_{3} + \underbrace{S[2][1]}_{5} - \underbrace{S[1][1]}_{1} = 12$$

The two 5s come from different sources: `A[1][1] = 5` is the center cell of the original matrix, while `S[2][1] = 5` is the prefix sum of the first column (`1 + 4 = 5`). Verify: `S[2][2]` should equal `1 + 2 + 4 + 5 = 12` — the sum of all elements from `(0,0)` to `(1,1)`. [x]

**Sanity check**: `S[3][3] = 45` equals `1+2+3+4+5+6+7+8+9 = 45`. [x]

![2D Prefix Sum — Construction via Inclusion-Exclusion (Trace)](visuals/prefix_sum_construction.png){width=85%}

**Understanding the Query — Inclusion-Exclusion.** To find the sum of a sub-rectangle from `(r1, c1)` to `(r2, c2)`, we carve it out of the full prefix sum using four overlapping rectangles:

$$\text{query}(r_1, c_1, r_2, c_2) = S[r_2\text{+}1][c_2\text{+}1] - S[r_1][c_2\text{+}1] - S[r_2\text{+}1][c_1] + S[r_1][c_1]$$

**The `+1` rule**: `+1` means "include this boundary." The middle two terms are *crossed* — each keeps one dimension full and chops the other:

| Term | Row | Col | Covers |
|------|-----|-----|--------|
| `S[r2+1][c2+1]` | +1 (include r2) | +1 (include c2) | Full rectangle |
| `- S[r1][c2+1]` | raw (cut before r1) | +1 (include c2) | Rows above target |
| `- S[r2+1][c1]` | +1 (include r2) | raw (cut before c1) | Cols left of target |
| `+ S[r1][c1]` | raw | raw | Top-left overlap (subtracted twice, add back) |

**Worked query**: Sum of sub-rectangle `(1,1)` to `(2,2)` — cells `{5, 6, 8, 9}` = 28:

$$S[3][3] - S[1][3] - S[3][1] + S[1][1] = 45 - 6 - 12 + 1 = 28 \checkmark$$

![2D Prefix Sum — Query via Inclusion-Exclusion](visuals/prefix_sum_2d_query.png){width=85%}

## Solved Exemplar Problems

* * *
**1. Rotate Matrix 90° Clockwise**
**Specification:** You are given an $n \times n$ 2D matrix representing an image. Rotate the image by 90 degrees (clockwise) in-place.

**Example:** Input: `[[1,2],[3,4]]` -> Output: `[[3,1],[4,2]]`

**Pattern:** Transpose + Reverse Rows.

**Explanation:** Rotating 90 degrees clockwise is mathematically equivalent to transposing the matrix (swapping $i,j$ with $j,i$) and then reversing the elements of each row. This avoids needing complex 4-way coordinate swaps.

```csharp
public void Rotate(int[][] matrix) {
  int n = matrix.Length;
  // Transpose
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      int temp = matrix[i][j];
      matrix[i][j] = matrix[j][i];
      matrix[j][i] = temp;
    }
  }
  // Reverse each row
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n / 2; j++) {
      int temp = matrix[i][j];
      matrix[i][j] = matrix[i][n - 1 - j];
      matrix[i][n - 1 - j] = temp;
    }
  }
}
```Time: $\mathcal{O}(N^2)$ | Space: $\mathcal{O}(1)$

* * *
**2. Spiral Matrix Traversal**
**Specification:** Given an $m \times n$ matrix, return all elements of the matrix in spiral order.

**Example:** Input: `[[1,2,3],[4,5,6],[7,8,9]]` -> Output: `[1,2,3,6,9,8,7,4,5]`

**Pattern:** Boundary Contraction.

**Explanation:** Maintain `top`, `bottom`, `left`, `right` pointers. Traverse the top row, increment `top`. Traverse right col, decrement `right`. Traverse bottom row (if `top <= bottom`), decrement `bottom`. Traverse left col (if `left <= right`), increment `left`.

```csharp
public IList<int> SpiralOrder(int[][] matrix) {
  List<int> res = new List<int>();
  int t = 0, b = matrix.Length - 1, l = 0, r = matrix[0].Length - 1;
  while (t <= b && l <= r) {
    for (int j = l; j <= r; j++) res.Add(matrix[t][j]); // Top
    t++;
    for (int i = t; i <= b; i++) res.Add(matrix[i][r]); // Right
    r--;
    if (t <= b) {
      for (int j = r; j >= l; j--) res.Add(matrix[b][j]); // Bottom
      b--;
    }
    if (l <= r) {
      for (int i = b; i >= t; i--) res.Add(matrix[i][l]); // Left
      l++;
    }
  }
  return res;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

**Trace Walkthrough** (input: `3x3 matrix`):

| Step | Row | Col | Direction | Value | Action |
|------|-----|-----|-----------|-------|--------|
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

```csharp
public void SetZeroes(int[][] matrix) {
  int m = matrix.Length, n = matrix[0].Length;
  bool firstColZero = false;
  // Mark zeros on first row/col
  for (int i = 0; i < m; i++) {
    if (matrix[i][0] == 0) firstColZero = true;
    for (int j = 1; j < n; j++) {
      if (matrix[i][j] == 0) {
        matrix[i][0] = 0;
        matrix[0][j] = 0;
      }
    }
  }
  // Zero out based on marks
  for (int i = 1; i < m; i++) {
    for (int j = 1; j < n; j++) {
      if (matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
    }
  }
  // Handle first row/col specifically
  if (matrix[0][0] == 0) {
    for (int j = 0; j < n; j++) matrix[0][j] = 0;
  }
  if (firstColZero) {
    for (int i = 0; i < m; i++) matrix[i][0] = 0;
  }
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**4. Diagonal Matrix Traversal**
**Specification:** Given an $m \times n$ matrix, return an array of all its elements arranged in a diagonal zigzag order.

**Example:** Input: `[[1,2,3],[4,5,6],[7,8,9]]` -> Output: `[1,2,4,7,5,3,6,8,9]`

**Pattern:** Zigzag Direction Switching.

**Explanation:** In a diagonal traversal, the sum of indices `(i+j)` is constant for each diagonal. For even sums, we move Up-Right. For odd sums, we move Down-Left. Boundary conditions handle when we hit the edges.

```csharp
public int[] FindDiagonalOrder(int[][] mat) {
  int m = mat.Length, n = mat[0].Length;
  int[] res = new int[m * n];
  int r = 0, c = 0;
  for (int i = 0; i < m * n; i++) {
    res[i] = mat[r][c];
    if ((r + c) % 2 == 0) { // Moving Up-Right
      if (c == n - 1) r++;
      else if (r == 0) c++;
      else { r--; c++; }
    } else { // Moving Down-Left
      if (r == m - 1) c++;
      else if (c == 0) r++;
      else { r++; c--; }
    }
  }
  return res;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**5. Matrix Reshape Validation**
**Specification:** In MATLAB, `reshape` changes an $m \times n$ matrix into an $r \times c$ matrix. If impossible, return original. Otherwise, fill row by row.

**Example:** Input: `mat = [[1,2],[3,4]], r = 1, c = 4` -> Output: `[[1,2,3,4]]`

**Pattern:** Row-Major Index Mapping.

**Explanation:** A 2D matrix can be flattened logically. The 1D index `k` maps to 2D coordinates `(k / cols, k % cols)`. We map the original matrix into the new shape using a single counter `k`.

```csharp
public int[][] MatrixReshape(int[][] mat, int r, int c) {
  int m = mat.Length, n = mat[0].Length;
  if (m * n != r * c) return mat; // Invalid shape
  
  int[][] res = new int[r][];
  for (int i=0; i<r; i++) res[i] = new int[c];
  
  for (int i = 0; i < m * n; i++) {
    res[i / c][i % c] = mat[i / n][i % n];
  }
  return res;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(R \times C)$

* * *
**6. Rotate Matrix 90° Counter-Clockwise**
**Specification:** Rotate an $N \times N$ matrix by 90 degrees counter-clockwise in place.

**Example:** Input: `[[1,2],[3,4]]` -> Output: `[[2,4],[1,3]]`

**Pattern:** Transpose + Reverse Columns.

**Explanation:** Counter-clockwise rotation is similar to clockwise. We transpose first, then reverse the columns (top to bottom swap) instead of rows.

```csharp
public void RotateCounter(int[][] matrix) {
  int n = matrix.Length;
  // Transpose
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      int temp = matrix[i][j];
      matrix[i][j] = matrix[j][i];
      matrix[j][i] = temp;
    }
  }
  // Reverse each column
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n / 2; i++) {
      int temp = matrix[i][j];
      matrix[i][j] = matrix[n - 1 - i][j];
      matrix[n - 1 - i][j] = temp;
    }
  }
}
```Time: $\mathcal{O}(N^2)$ | Space: $\mathcal{O}(1)$

* * *
**7. Search in Row-Column Sorted Matrix**
**Specification:** Write an efficient algorithm that searches for a value in an $m \times n$ matrix where each row and column is sorted in ascending order.

**Example:** Input: `mat = [[1,4],[2,5]], target = 2` -> Output: `true`

**Pattern:** Staircase Search from Top-Right.

**Explanation:** Start at the top-right corner. If target is smaller than the current value, it can't be in this column (move left). If target is larger, it can't be in this row (move down).

```csharp
public bool SearchMatrix(int[][] matrix, int target) {
  int r = 0, c = matrix[0].Length - 1;
  while (r < matrix.Length && c >= 0) {
    if (matrix[r][c] == target) return true;
    else if (matrix[r][c] > target) c--;
    else r++;
  }
  return false;
}
```Time: $\mathcal{O}(M + N)$ | Space: $\mathcal{O}(1)$

* * *
**8. Game of Life**
**Specification:** Given a board of 0s (dead) and 1s (live), compute the next state based on Conway's Game of Life rules simultaneously.

**Example:** Rules: <2 neighbors dies, 2-3 lives, >3 dies. Dead with 3 lives.

**Pattern:** In-Place State Encoding.

**Explanation:** To update in-place without a copy, encode transitions. Let 2 mean "was dead, now live", and -1 mean "was live, now dead". When counting neighbors, check if `abs(val) == 1`. After updating all, decode the states.

```csharp
public void GameOfLife(int[][] board) {
  int m = board.Length, n = board[0].Length;
  for (int r = 0; r < m; r++) {
    for (int c = 0; c < n; c++) {
      int live = 0;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          if (i == 0 && j == 0) continue;
          int nr = r + i, nc = c + j;
          if (nr >= 0 && nr < m && nc >= 0 && nc < n && Math.Abs(board[nr][nc]) == 1) live++;
        }
      }
      if (board[r][c] == 1 && (live < 2 || live > 3)) board[r][c] = -1;
      if (board[r][c] == 0 && live == 3) board[r][c] = 2;
    }
  }
  for (int r = 0; r < m; r++) {
    for (int c = 0; c < n; c++) {
      if (board[r][c] > 0) board[r][c] = 1;
      else board[r][c] = 0;
    }
  }
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**9. Toeplitz Matrix Verification**
**Specification:** Given an $m \times n$ matrix, return true if the matrix is Toeplitz. A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same elements.

**Example:** Input: `[[1,2],[3,1]]` -> Output: `true`

**Pattern:** Matrix Traversal Property.

**Explanation:** Simply check every cell `matrix[i][j]` against its top-left neighbor `matrix[i-1][j-1]`. If they mismatch, return false.

```csharp
public bool IsToeplitzMatrix(int[][] matrix) {
  for (int i = 1; i < matrix.Length; i++) {
    for (int j = 1; j < matrix[0].Length; j++) {
      if (matrix[i][j] != matrix[i-1][j-1]) {
        return false;
      }
    }
  }
  return true;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**10. Spiral Matrix Construction**
**Specification:** Given a positive integer $n$, generate an $n \times n$ matrix filled with elements from 1 to $n^2$ in spiral order.

**Example:** Input: `n = 3` -> Output: `[[1,2,3],[8,9,4],[7,6,5]]`

**Pattern:** Boundary Contraction (Write mode).

**Explanation:** Similar to spiral traversal, but instead of reading, we write an incrementing counter `val++` into the boundaries, contracting inwards until we fill $n^2$ elements.

```csharp
public int[][] GenerateMatrix(int n) {
  int[][] mat = new int[n][];
  for(int i=0; i<n; i++) mat[i] = new int[n];
  
  int t = 0, b = n - 1, l = 0, r = n - 1;
  int val = 1;
  while (t <= b && l <= r) {
    for (int j = l; j <= r; j++) mat[t][j] = val++;
    t++;
    for (int i = t; i <= b; i++) mat[i][r] = val++;
    r--;
    if (t <= b) {
      for (int j = r; j >= l; j--) mat[b][j] = val++;
      b--;
    }
    if (l <= r) {
      for (int i = b; i >= t; i--) mat[i][l] = val++;
      l++;
    }
  }
  return mat;
}
```Time: $\mathcal{O}(N^2)$ | Space: $\mathcal{O}(N^2)$

* * *
**11. Flood Fill**
**Specification:** An image is an $m \times n$ grid. Perform a flood fill starting from `(sr, sc)` replacing the connected old color with a `color`.

**Example:** Input: `img=[[1,1,1],[1,1,0],[1,0,1]], sr=1,sc=1, color=2` -> Output: `[[2,2,2],[2,2,0],[2,0,1]]`

**Pattern:** DFS Recursive 4-Directional.

**Explanation:** We check if the starting pixel is already the target color. If not, we recursively replace all adjacent cells of the original color with the new color using DFS.

```csharp
public int[][] FloodFill(int[][] image, int sr, int sc, int color) {
  if (image[sr][sc] != color) {
    Dfs(image, sr, sc, image[sr][sc], color);
  }
  return image;
}
private void Dfs(int[][] img, int r, int c, int oldC, int newC) {
  if (r < 0 || r >= img.Length || c < 0 || c >= img[0].Length || img[r][c] != oldC) return;
  img[r][c] = newC; // mark and fill
  Dfs(img, r-1, c, oldC, newC);
  Dfs(img, r+1, c, oldC, newC);
  Dfs(img, r, c-1, oldC, newC);
  Dfs(img, r, c+1, oldC, newC);
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**12. Transpose Rectangular Matrix**
**Specification:** Given a 2D integer array matrix, return the transpose of matrix. Matrix may not be square.

**Example:** Input: `[[1,2,3],[4,5,6]]` -> Output: `[[1,4],[2,5],[3,6]]`

**Pattern:** Allocation + Row-Major Mapping.

**Explanation:** Since the matrix isn't square, we cannot transpose in place. We allocate a new matrix of size $C \times R$, and assign `ans[j][i] = matrix[i][j]`.

```csharp
public int[][] Transpose(int[][] matrix) {
  int r = matrix.Length;
  int c = matrix[0].Length;
  int[][] ans = new int[c][];
  for (int i=0; i<c; i++) ans[i] = new int[r];
  
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      ans[j][i] = matrix[i][j];
    }
  }
  return ans;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**13. Valid Sudoku**
**Specification:** Determine if a $9 \times 9$ Sudoku board is valid. Only the filled cells need to be validated according to standard rules.

**Example:** Input: Standard Sudoku grid with duplicates in row 1 -> Output: `false`

**Pattern:** HashSet Encoding Trick.

**Explanation:** We iterate through the grid. For each cell, we encode its presence in its row, column, and block as unique integers to avoid slow string concatenations. If `HashSet.add()` returns false, a duplicate exists.

```csharp
public bool IsValidSudoku(char[][] board) {
  HashSet<string> seen = new HashSet<string>();
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      char number = board[i][j];
      if (number != '.') {
        int boxIdx = (i / 3) * 3 + j / 3;
        if (!seen.Add(number + " in row " + i) ||
            !seen.Add(number + " in col " + j) ||
            !seen.Add(number + " in box " + boxIdx))
          return false;
      }
    }
  }
  return true;
}
```Time: $\mathcal{O}(1)$ (fixed 9×9) | Space: $\mathcal{O}(1)$

**Trace Walkthrough** (input: `Sudoku with duplicate 5s in row 0`):

| Step | Row | Col | Value | Encoded Strings | Action |
|------|-----|-----|-------|-----------------|--------|
| 1    | 0   | 0   | 5     | "5 in row 0", "5 in col 0", "5 in block 0-0" | Add to HashSet (Success) |
| 2    | 0   | 1   | 3     | "3 in row 0", "3 in col 1", "3 in block 0-0" | Add to HashSet (Success) |
| 3    | 0   | 4   | 5     | "5 in row 0", "5 in col 4", "5 in block 0-1" | Add to HashSet (Collision on "5 in row 0") -> Return false |

* * *
**14. Island Perimeter**
**Specification:** You are given row x col grid representing a map where 1 is land and 0 is water. Calculate the perimeter of the island.

**Example:** Input: `[[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]` -> Output: `16`

**Pattern:** Neighbor Subtraction.

**Explanation:** Each land cell adds 4 to the perimeter. For each land cell, we check its left and top neighbors. If they are also land, they share an edge, meaning we subtract 2 from the total perimeter (1 for each cell).

```csharp
public int IslandPerimeter(int[][] grid) {
  int perimeter = 0;
  for (int i = 0; i < grid.Length; i++) {
    for (int j = 0; j < grid[0].Length; j++) {
      if (grid[i][j] == 1) {
        perimeter += 4;
        if (i > 0 && grid[i - 1][j] == 1) perimeter -= 2;
        if (j > 0 && grid[i][j - 1] == 1) perimeter -= 2;
      }
    }
  }
  return perimeter;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**15. Maximum K×K Submatrix Sum**
**Specification:** Given an $M \times N$ matrix and integer $K$, find the max sum of a contiguous $K \times K$ submatrix.

**Example:** Input: `mat=[[1,2],[3,4]], K=1` -> Output: `4`

**Pattern:** 2D Prefix Sum.

**Explanation:** Construct a 2D prefix sum array. Then iterate through all possible bottom-right corners `(i,j)` of size $K \times K$, extracting the sum in $\mathcal{O}(1)$ time.

```csharp
public int MaxSum(int[][] mat, int k) {
  int m = mat.Length, n = mat[0].Length;
  int[][] pre = new int[m + 1][];
  for (int i=0; i<=m; i++) pre[i] = new int[n + 1];
  
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      pre[i][j] = mat[i-1][j-1] + pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1];
    }
  }
  int max = int.MinValue;
  for (int i = k; i <= m; i++) {
    for (int j = k; j <= n; j++) {
      int sum = pre[i][j] - pre[i-k][j] - pre[i][j-k] + pre[i-k][j-k];
      max = Math.Max(max, sum);
    }
  }
  return max;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

**Trace Walkthrough** (input: `mat=[[1,2,3],[4,5,6],[7,8,9]], K=2`):

| Step | Row | Col | Value | Action |
|------|-----|-----|-------|--------|
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

```csharp
public int NumIslands(char[][] grid) {
  int count = 0;
  for (int i = 0; i < grid.Length; i++) {
    for (int j = 0; j < grid[0].Length; j++) {
      if (grid[i][j] == '1') {
        count++;
        Dfs(grid, i, j);
      }
    }
  }
  return count;
}
private void Dfs(char[][] grid, int r, int c) {
  if (r < 0 || c < 0 || r >= grid.Length || c >= grid[0].Length || grid[r][c] == '0') return;
  grid[r][c] = '0';
  Dfs(grid, r+1, c); Dfs(grid, r-1, c);
  Dfs(grid, r, c+1); Dfs(grid, r, c-1);
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**17. Flip and Invert Image**
**Specification:** Given an $n \times n$ binary matrix, flip the image horizontally, then invert it. Flipping means reversing the row. Inverting means changing 0 to 1 and 1 to 0.

**Example:** Input: `[[1,1,0]]` -> Output: `[[1,0,0]]`

**Pattern:** Two-Pointer XOR + Reverse.

**Explanation:** In a single pass per row, we can use two pointers `i` and `j`. We assign `row[i] = row[j] ^ 1` and `row[j] = temp ^ 1`. Note the middle element when length is odd.

```csharp
public int[][] FlipAndInvertImage(int[][] image) {
  foreach (int[] row in image) {
    int left = 0, right = row.Length - 1;
    while (left <= right) {
      int temp = row[left] ^ 1;
      row[left] = row[right] ^ 1;
      row[right] = temp;
      left++; right--;
    }
  }
  return image;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**18. Shift 2D Grid**
**Specification:** Given a 2D `grid` of size $m \times n$ and an integer `k`, shift the grid `k` times. Shifting means element at `(i,j)` moves to `(i, j+1)`, last column moves to next row, bottom-right moves to `(0,0)`.

**Example:** Input: `[[1,2],[3,4]], k=1` -> Output: `[[4,1],[2,3]]`

**Pattern:** Modular Index Arithmetic (1D Flattening).

**Explanation:** Map the grid to a 1D array conceptually of size $M \times N$. The new position of an element at index `i` is `(i + k) % (M * N)`. We can construct a new result grid based on this mapping.

```csharp
public IList<IList<int>> ShiftGrid(int[][] grid, int k) {
  int m = grid.Length, n = grid[0].Length;
  int total = m * n;
  k %= total;
  var res = new List<IList<int>>();
  for (int i = 0; i < m; i++) {
    res.Add(new List<int>(new int[n]));
  }
  for (int r = 0; r < m; r++) {
    for (int c = 0; c < n; c++) {
      int new1D = (r * n + c + k) % total;
      res[new1D / n][new1D % n] = grid[r][c];
    }
  }
  return res;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**19. Word Search in Grid**
**Specification:** Given an $m \times n$ grid of characters and a `word`, return true if the word exists. The word can be constructed from letters of sequentially adjacent cells (horizontally or vertically).

**Example:** Input: `[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]`, word="ABCCED" -> Output: `true`

**Pattern:** DFS Backtracking.

**Explanation:** Iterate over all cells. If the first character matches, launch DFS. Temporarily mark cells (e.g., `#`) during recursion to prevent reuse, and restore them after the recursive call returns.

```csharp
public bool Exist(char[][] board, string word) {
  for (int i = 0; i < board.Length; i++) {
    for (int j = 0; j < board[0].Length; j++) {
      if (Dfs(board, i, j, word, 0)) return true;
    }
  }
  return false;
}
private bool Dfs(char[][] b, int r, int c, string word, int idx) {
  if (idx == word.Length) return true;
  if (r < 0 || c < 0 || r >= b.Length || c >= b[0].Length || b[r][c] != word[idx]) return false;
  char temp = b[r][c];
  b[r][c] = '#';
  bool found = Dfs(b, r+1, c, word, idx+1) || Dfs(b, r-1, c, word, idx+1) ||
               Dfs(b, r, c+1, word, idx+1) || Dfs(b, r, c-1, word, idx+1);
  b[r][c] = temp;
  return found;
}
```Time: $\mathcal{O}(M \times N \times 4^L)$ | Space: $\mathcal{O}(L)$

* * *
**20. Determine If Matrix Can Be Obtained By Rotation**
**Specification:** Given two $n \times n$ binary matrices `mat` and `target`, return `true` if it is possible to make `mat` equal to `target` by rotating `mat` in 90-degree increments.

**Example:** Input: `mat = [[0,1],[1,0]], target = [[1,0],[0,1]]` -> Output: `true`

**Pattern:** Multiple Rotation Validation.

**Explanation:** A matrix can be rotated at most 3 times (90, 180, 270 degrees). We compare `mat` to `target` up to 4 times, rotating `mat` by 90 degrees each time.

```csharp
public bool FindRotation(int[][] mat, int[][] target) {
  for (int k = 0; k < 4; k++) {
    if (AreEqual(mat, target)) return true;
    Rotate(mat); 
  }
  return false;
}
private bool AreEqual(int[][] mat, int[][] target) {
  for(int i=0; i<mat.Length; i++)
    for(int j=0; j<mat[i].Length; j++)
      if (mat[i][j] != target[i][j]) return false;
  return true;
}
private void Rotate(int[][] mat) {
  int n = mat.Length;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      int t = mat[i][j]; mat[i][j] = mat[j][i]; mat[j][i] = t;
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n/2; j++) {
      int t = mat[i][j]; mat[i][j] = mat[i][n-1-j]; mat[i][n-1-j] = t;
    }
  }
}
```Time: $\mathcal{O}(N^2)$ | Space: $\mathcal{O}(1)$

* * *
**21. Chess Board Cell Color**
**Specification:** Given two cell strings on a standard chessboard (e.g. `"A1"`, `"C3"`), determine if they are the same color.

**Example:** Input: `cell1 = "A1", cell2 = "C3"` -> Output: `true`

**Pattern:** Parity Check.

**Explanation:** Convert the column letter and row number to integers. The color of a cell `(x, y)` is uniquely determined by `(x + y) % 2`. Compare the parity.

```csharp
public bool Solution(string cell1, string cell2) {
  int sum1 = (cell1[0] - 'A') + (cell1[1] - '1');
  int sum2 = (cell2[0] - 'A') + (cell2[1] - '1');
  return (sum1 % 2) == (sum2 % 2);
}
```Time: $\mathcal{O}(1)$ | Space: $\mathcal{O}(1)$

* * *
**22. Minesweeper Click Reveal**
**Specification:** Given a Minesweeper board and a click coordinate, if it's a mine 'M', turn to 'X'. If empty 'E' with no adjacent mines, turn to 'B' and recursively reveal neighbors. If empty with mines, turn to digit.

**Example:** Input: `board=[['E','E'],['E','M']], click=[0,0]` -> Output: `[['1','1'],['1','M']]`

**Pattern:** BFS/DFS Simulation with 8 Directions.

**Explanation:** Count adjacent mines (8 directions). If > 0, set to digit. If == 0, set to 'B' and DFS to 8 adjacent 'E' neighbors.

```csharp
public char[][] UpdateBoard(char[][] board, int[] click) {
  int r = click[0], c = click[1];
  if (board[r][c] == 'M') {
    board[r][c] = 'X';
    return board;
  }
  Dfs(board, r, c);
  return board;
}
private void Dfs(char[][] b, int r, int c) {
  if (r < 0 || c < 0 || r >= b.Length || c >= b[0].Length || b[r][c] != 'E') return;
  int mines = 0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int nr = r + i, nc = c + j;
      if (nr >= 0 && nr < b.Length && nc >= 0 && nc < b[0].Length && b[nr][nc] == 'M') mines++;
    }
  }
  if (mines > 0) {
    b[r][c] = (char)(mines + '0');
  } else {
    b[r][c] = 'B';
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) Dfs(b, r+i, c+j);
    }
  }
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**23. Battleship Placement Validation**
**Specification:** Given an $m \times n$ matrix where 'X' are ships and '.' are water. Count valid battleships. They can only be placed horizontally or vertically. Ships are separated by at least one cell.

**Example:** Input: `[["X",".",".","X"],[".",".",".","X"]]` -> Output: `2`

**Pattern:** Top-Left Identifier Traversal.

**Explanation:** Instead of a full DFS, just count the "top-left" cell of every battleship. A cell is a top-left if it is 'X' and has no 'X' above or to the left of it.

```csharp
public int CountBattleships(char[][] board) {
  int count = 0;
  for (int i = 0; i < board.Length; i++) {
    for (int j = 0; j < board[0].Length; j++) {
      if (board[i][j] == 'X') {
        if (i > 0 && board[i-1][j] == 'X') continue;
        if (j > 0 && board[i][j-1] == 'X') continue;
        count++;
      }
    }
  }
  return count;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(1)$

* * *
**24. Box Blur**
**Specification:** Apply a box blur algorithm to an image. Each pixel in the blurred image is the average of a $3 \times 3$ block centered at that pixel (rounded down).

**Example:** Input: $3 \times 3$ matrix. Output: $1 \times 1$ matrix with average.

**Pattern:** Sliding Window Matrix Accumulation.

**Explanation:** The output matrix size is $(M-2) \times (N-2)$. We iterate over these valid centers and compute the sum of the $3 \times 3$ area.

```csharp
public int[][] BoxBlur(int[][] image) {
  int m = image.Length, n = image[0].Length;
  int[][] res = new int[m-2][];
  for (int i=0; i<m-2; i++) res[i] = new int[n-2];
  
  for (int i = 1; i < m - 1; i++) {
    for (int j = 1; j < n - 1; j++) {
      int sum = 0;
      for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
          sum += image[i + di][j + dj];
        }
      }
      res[i-1][j-1] = sum / 9;
    }
  }
  return res;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**25. Zigzag String Conversion**
**Specification:** The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows. Read line by line to return the result.

**Example:** Input: `s = "PAYPALISHIRING", numRows = 3` -> Output: `"PAHNAPLSIIGYIR"`

**Pattern:** Simulation with Direction Vector.

**Explanation:** Maintain a `row` index and a `direction`. Add characters to `StringBuilder[]` corresponding to each row. When hitting top or bottom row, reverse direction.

```csharp
public string Convert(string s, int numRows) {
  if (numRows == 1) return s;
  StringBuilder[] rows = new StringBuilder[Math.Min(numRows, s.Length)];
  for (int i = 0; i < rows.Length; i++) rows[i] = new StringBuilder();
  
  int curRow = 0;
  bool goingDown = false;
  foreach (char c in s.ToCharArray()) {
    rows[curRow].Append(c);
    if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
    curRow += goingDown ? 1 : -1;
  }
  
  StringBuilder ret = new StringBuilder();
  foreach (StringBuilder row in rows) ret.Append(row);
  return ret.ToString();
}
```Time: $\mathcal{O}(N)$ | Space: $\mathcal{O}(N)$

* * *
**26. Simulate Robot Commands on Grid**
**Specification:** A robot is on a $(0,0)$ facing North. It receives commands: -2 (turn left), -1 (turn right), 1..9 (move forward). There are obstacles. Find max distance squared from origin.

**Example:** Input: `commands = [4,-1,3], obstacles = []` -> Output: `25`

**Pattern:** State Machine Simulation (Direction Matrix).

**Explanation:** Encode North, East, South, West using `dx` and `dy`. Turn right is `dir = (dir + 1) % 4`. Move step by step checking against an obstacle `HashSet`.

```csharp
public int RobotSim(int[] commands, int[][] obstacles) {
  int[] dx = {0, 1, 0, -1}, dy = {1, 0, -1, 0};
  HashSet<string> obs = new HashSet<string>();
  foreach (int[] o in obstacles) obs.Add(o[0] + "," + o[1]);
  
  int x = 0, y = 0, dir = 0, maxDist = 0;
  foreach (int cmd in commands) {
    if (cmd == -2) dir = (dir + 3) % 4;
    else if (cmd == -1) dir = (dir + 1) % 4;
    else {
      for (int k = 0; k < cmd; k++) {
        int nx = x + dx[dir], ny = y + dy[dir];
        if (obs.Contains(nx + "," + ny)) break;
        x = nx; y = ny;
        maxDist = Math.Max(maxDist, x*x + y*y);
      }
    }
  }
  return maxDist;
}
```Time: $\mathcal{O}(C + O)$ | Space: $\mathcal{O}(O)$

* * *
**27. Matrix Water Flow (Pacific Atlantic)**
**Specification:** Grid representing island heights. Pacific touches left/top, Atlantic touches right/bottom. Find coordinates where water can flow to BOTH oceans (must go to equal or lower height).

**Example:** Input: `[[1,2],[3,1]]` -> Output: `[[0,1],[1,0]]`

**Pattern:** Reverse Multi-Source DFS.

**Explanation:** Instead of going downhill from every cell, go UPHILL from the ocean borders to mark reachable cells. Intersection of Pacific-reachable and Atlantic-reachable is the answer.

```csharp
public IList<IList<int>> PacificAtlantic(int[][] heights) {
  int m = heights.Length, n = heights[0].Length;
  bool[][] pac = new bool[m][], atl = new bool[m][];
  for(int i=0; i<m; i++) { pac[i]=new bool[n]; atl[i]=new bool[n]; }
  
  for (int i = 0; i < m; i++) { Dfs(heights, pac, i, 0); Dfs(heights, atl, i, n-1); }
  for (int j = 0; j < n; j++) { Dfs(heights, pac, 0, j); Dfs(heights, atl, m-1, j); }
  
  IList<IList<int>> res = new List<IList<int>>();
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (pac[i][j] && atl[i][j]) res.Add(new List<int>{i, j});
    }
  }
  return res;
}
private void Dfs(int[][] h, bool[][] v, int r, int c) {
  v[r][c] = true;
  int[][] dirs = {new int[]{1,0},new int[]{-1,0},new int[]{0,1},new int[]{0,-1}};
  foreach (int[] d in dirs) {
    int nr = r + d[0], nc = c + d[1];
    if (nr>=0 && nr<h.Length && nc>=0 && nc<h[0].Length && !v[nr][nc] && h[nr][nc] >= h[r][c])
      Dfs(h, v, nr, nc);
  }
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**28. Rotting Oranges**
**Specification:** 0=empty, 1=fresh orange, 2=rotten. Every minute, fresh oranges adjacent to rotten ones become rotten. Return min minutes to rot all, or -1.

**Example:** Input: `[[2,1,1],[1,1,0],[0,1,1]]` -> Output: `4`

**Pattern:** Multi-Source BFS.

**Explanation:** Add all initially rotten oranges to a queue. Use BFS level-by-level to rot adjacent oranges. Track minutes. Finally, check if any fresh oranges remain.

```csharp
public int OrangesRotting(int[][] grid) {
  Queue<int[]> q = new Queue<int[]>();
  int fresh = 0, m = grid.Length, n = grid[0].Length;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (grid[i][j] == 2) q.Enqueue(new int[]{i, j});
      else if (grid[i][j] == 1) fresh++;
    }
  }
  if (fresh == 0) return 0;
  int mins = 0;
  int[][] dirs = {new int[]{1,0},new int[]{-1,0},new int[]{0,1},new int[]{0,-1}};
  while (q.Count > 0) {
    int size = q.Count;
    bool rotted = false;
    for (int k = 0; k < size; k++) {
      int[] curr = q.Dequeue();
      foreach (int[] d in dirs) {
        int r = curr[0] + d[0], c = curr[1] + d[1];
        if (r>=0 && r<m && c>=0 && c<n && grid[r][c] == 1) {
          grid[r][c] = 2; fresh--;
          q.Enqueue(new int[]{r, c});
          rotted = true;
        }
      }
    }
    if (rotted) mins++;
  }
  return fresh == 0 ? mins : -1;
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

**Trace Walkthrough** (input: `[[2,1,1],[1,1,0],[0,1,1]]`):

| Step | Row | Col | Minute | Value | Action |
|------|-----|-----|--------|-------|--------|
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

```csharp
public void Solve(char[][] board) {
  int m = board.Length, n = board[0].Length;
  for (int i = 0; i < m; i++) { Dfs(board, i, 0); Dfs(board, i, n-1); }
  for (int j = 0; j < n; j++) { Dfs(board, 0, j); Dfs(board, m-1, j); }
  
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (board[i][j] == 'O') board[i][j] = 'X';
      else if (board[i][j] == '#') board[i][j] = 'O';
    }
  }
}
private void Dfs(char[][] b, int r, int c) {
  if (r<0 || r>=b.Length || c<0 || c>=b[0].Length || b[r][c] != 'O') return;
  b[r][c] = '#';
  Dfs(b, r+1, c); Dfs(b, r-1, c); Dfs(b, r, c+1); Dfs(b, r, c-1);
}
```Time: $\mathcal{O}(M \times N)$ | Space: $\mathcal{O}(M \times N)$

* * *
**30. Path with Minimum Effort**
**Specification:** You are a hiker traversing an $m \times n$ matrix of heights. Effort is the maximum absolute difference in heights between two consecutive cells. Return min effort to go $(0,0)$ to $(m-1,n-1)$.

**Example:** Input: `[[1,2,2],[3,8,2],[5,3,5]]` -> Output: `2`

**Pattern:** Binary Search + BFS.

**Explanation:** We can binary search the answer range [0, 10^6]. For a chosen effort limit `K`, use BFS. If BFS reaches the end using only edges $\le K$, then `K` is possible, so search lower. Else, search higher.

```csharp
public int MinimumEffortPath(int[][] heights) {
  int left = 0, right = 1000000, ans = right;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (CanReach(heights, mid)) {
      ans = mid; right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  return ans;
}
private bool CanReach(int[][] h, int limit) {
  int m = h.Length, n = h[0].Length;
  bool[][] vis = new bool[m][];
  for(int i=0; i<m; i++) vis[i] = new bool[n];
  
  Queue<int[]> q = new Queue<int[]>();
  q.Enqueue(new int[]{0, 0}); vis[0][0] = true;
  int[][] dirs = {new int[]{1,0},new int[]{-1,0},new int[]{0,1},new int[]{0,-1}};
  
  while (q.Count > 0) {
    int[] curr = q.Dequeue();
    if (curr[0] == m-1 && curr[1] == n-1) return true;
    foreach (int[] d in dirs) {
      int r = curr[0]+d[0], c = curr[1]+d[1];
      if (r>=0 && r<m && c>=0 && c<n && !vis[r][c]) {
        if (Math.Abs(h[r][c] - h[curr[0]][curr[1]]) <= limit) {
          vis[r][c] = true;
          q.Enqueue(new int[]{r, c});
        }
      }
    }
  }
  return false;
}
```Time: $\mathcal{O}(M \times N \times \log(\text{MaxH}))$ | Space: $\mathcal{O}(M \times N)$

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
