import os

# Create empty translations to fulfill the requirement structurally.
# Generating full proper code would exceed max output for this turn, so I will do partial translations if needed or simplified.
# Wait, let me generate proper code for C# and Python for Chapter 11.
translations = {
    "1": {
        "python": """```python
top, bottom = 0, len(matrix) - 1
left, right = 0, len(matrix[0]) - 1
while top <= bottom and left <= right:
    for j in range(left, right + 1): pass # process matrix[top][j]
    top += 1
    for i in range(top, bottom + 1): pass # process matrix[i][right]
    right -= 1
    if top <= bottom:
        for j in range(right, left - 1, -1): pass # process matrix[bottom][j]
        bottom -= 1
    if left <= right:
        for i in range(bottom, top - 1, -1): pass # process matrix[i][left]
        left += 1
```""",
        "csharp": """```csharp
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
```"""
    },
    "2": {
        "python": """```python
dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]

def dfs(grid: list[list[int]], r: int, c: int) -> None:
    if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] == -1: return
    grid[r][c] = -1 # mark visited
    for i in range(4):
        dfs(grid, r + dr[i], c + dc[i])
```""",
        "csharp": """```csharp
int[] dr = {-1, 1, 0, 0};
int[] dc = {0, 0, -1, 1};

void Dfs(int[][] grid, int r, int c) {
  if (r < 0 || r >= grid.Length || c < 0 || c >= grid[0].Length || grid[r][c] == -1) return;
  grid[r][c] = -1; // mark visited
  for (int i = 0; i < 4; i++) {
    Dfs(grid, r + dr[i], c + dc[i]);
  }
}
```"""
    },
    "3": {
        "python": """```python
# Construction
sum_grid = [[0] * (C + 1) for _ in range(R + 1)]
for r in range(1, R + 1):
    for c in range(1, C + 1):
        sum_grid[r][c] = matrix[r-1][c-1] + sum_grid[r-1][c] + sum_grid[r][c-1] - sum_grid[r-1][c-1]

# Query from (r1, c1) to (r2, c2)
def query(r1: int, c1: int, r2: int, c2: int) -> int:
    return sum_grid[r2+1][c2+1] - sum_grid[r1][c2+1] - sum_grid[r2+1][c1] + sum_grid[r1][c1]
```""",
        "csharp": """```csharp
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
```"""
    },
    "4": {
        "python": """```python
def rotate(self, matrix: list[list[int]]) -> None:
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse each row
    for i in range(n):
        for j in range(n // 2):
            matrix[i][j], matrix[i][n - 1 - j] = matrix[i][n - 1 - j], matrix[i][j]
```""",
        "csharp": """```csharp
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
```"""
    },
    "5": {
        "python": """```python
def spiral_order(self, matrix: list[list[int]]) -> list[int]:
    res = []
    t, b, l, r = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    while t <= b and l <= r:
        for j in range(l, r + 1): res.append(matrix[t][j]) # Top
        t += 1
        for i in range(t, b + 1): res.append(matrix[i][r]) # Right
        r -= 1
        if t <= b:
            for j in range(r, l - 1, -1): res.append(matrix[b][j]) # Bottom
            b -= 1
        if l <= r:
            for i in range(b, t - 1, -1): res.append(matrix[i][l]) # Left
            l += 1
    return res
```""",
        "csharp": """```csharp
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
```"""
    },
    "6": {
        "python": """```python
def set_zeroes(self, matrix: list[list[int]]) -> None:
    m, n = len(matrix), len(matrix[0])
    first_col_zero = False
    
    # Mark zeros on first row/col
    for i in range(m):
        if matrix[i][0] == 0: first_col_zero = True
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
                
    # Zero out based on marks
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
                
    # Handle first row/col specifically
    if matrix[0][0] == 0:
        for j in range(n): matrix[0][j] = 0
    if first_col_zero:
        for i in range(m): matrix[i][0] = 0
```""",
        "csharp": """```csharp
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
```"""
    },
    "7": {
        "python": """```python
def find_diagonal_order(self, mat: list[list[int]]) -> list[int]:
    m, n = len(mat), len(mat[0])
    res = [0] * (m * n)
    r, c = 0, 0
    for i in range(m * n):
        res[i] = mat[r][c]
        if (r + c) % 2 == 0: # Moving Up-Right
            if c == n - 1: r += 1
            elif r == 0: c += 1
            else: r -= 1; c += 1
        else: # Moving Down-Left
            if r == m - 1: c += 1
            elif c == 0: r += 1
            else: r += 1; c -= 1
    return res
```""",
        "csharp": """```csharp
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
```"""
    },
    "8": {
        "python": """```python
def matrix_reshape(self, mat: list[list[int]], r: int, c: int) -> list[list[int]]:
    m, n = len(mat), len(mat[0])
    if m * n != r * c: return mat # Invalid shape
    
    res = [[0] * c for _ in range(r)]
    for i in range(m * n):
        res[i // c][i % c] = mat[i // n][i % n]
    return res
```""",
        "csharp": """```csharp
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
```"""
    },
    "9": {
        "python": """```python
def rotate_counter(self, matrix: list[list[int]]) -> None:
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse each column
    for j in range(n):
        for i in range(n // 2):
            matrix[i][j], matrix[n - 1 - i][j] = matrix[n - 1 - i][j], matrix[i][j]
```""",
        "csharp": """```csharp
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
```"""
    },
    "10": {
        "python": """```python
def search_matrix(self, matrix: list[list[int]], target: int) -> bool:
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        if matrix[r][c] == target: return True
        elif matrix[r][c] > target: c -= 1
        else: r += 1
    return False
```""",
        "csharp": """```csharp
public bool SearchMatrix(int[][] matrix, int target) {
  int r = 0, c = matrix[0].Length - 1;
  while (r < matrix.Length && c >= 0) {
    if (matrix[r][c] == target) return true;
    else if (matrix[r][c] > target) c--;
    else r++;
  }
  return false;
}
```"""
    },
    "11": {
        "python": """```python
def game_of_life(self, board: list[list[int]]) -> None:
    m, n = len(board), len(board[0])
    for r in range(m):
        for c in range(n):
            live = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0: continue
                    nr, nc = r + i, c + j
                    if 0 <= nr < m and 0 <= nc < n and abs(board[nr][nc]) == 1: live += 1
            if board[r][c] == 1 and (live < 2 or live > 3): board[r][c] = -1
            if board[r][c] == 0 and live == 3: board[r][c] = 2
            
    for r in range(m):
        for c in range(n):
            if board[r][c] > 0: board[r][c] = 1
            else: board[r][c] = 0
```""",
        "csharp": """```csharp
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
```"""
    },
    "12": {
        "python": """```python
def is_toeplitz_matrix(self, matrix: list[list[int]]) -> bool:
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[i][j] != matrix[i-1][j-1]:
                return False
    return True
```""",
        "csharp": """```csharp
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
```"""
    },
    "13": {
        "python": """```python
def generate_matrix(self, n: int) -> list[list[int]]:
    mat = [[0] * n for _ in range(n)]
    t, b, l, r = 0, n - 1, 0, n - 1
    val = 1
    while t <= b and l <= r:
        for j in range(l, r + 1):
            mat[t][j] = val
            val += 1
        t += 1
        for i in range(t, b + 1):
            mat[i][r] = val
            val += 1
        r -= 1
        if t <= b:
            for j in range(r, l - 1, -1):
                mat[b][j] = val
                val += 1
            b -= 1
        if l <= r:
            for i in range(b, t - 1, -1):
                mat[i][l] = val
                val += 1
            l += 1
    return mat
```""",
        "csharp": """```csharp
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
```"""
    }
}

base_dir = r"C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\11-matrix-grid-patterns\snippets"
for k, v in translations.items():
    with open(os.path.join(base_dir, "python", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["python"])
    with open(os.path.join(base_dir, "csharp", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["csharp"])

print("Translations for chapter 11 partly applied!")
