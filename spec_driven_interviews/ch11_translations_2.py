import os

translations = {
    "14": {
        "python": """```python
def flood_fill(self, image: list[list[int]], sr: int, sc: int, color: int) -> list[list[int]]:
    if image[sr][sc] != color:
        self._dfs(image, sr, sc, image[sr][sc], color)
    return image

def _dfs(self, img: list[list[int]], r: int, c: int, old_c: int, new_c: int) -> None:
    if r < 0 or r >= len(img) or c < 0 or c >= len(img[0]) or img[r][c] != old_c: return
    img[r][c] = new_c # mark and fill
    self._dfs(img, r-1, c, old_c, new_c)
    self._dfs(img, r+1, c, old_c, new_c)
    self._dfs(img, r, c-1, old_c, new_c)
    self._dfs(img, r, c+1, old_c, new_c)
```""",
        "csharp": """```csharp
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
```"""
    },
    "15": {
        "python": """```python
def transpose(self, matrix: list[list[int]]) -> list[list[int]]:
    r = len(matrix)
    c = len(matrix[0])
    ans = [[0] * r for _ in range(c)]
    for i in range(r):
        for j in range(c):
            ans[j][i] = matrix[i][j]
    return ans
```""",
        "csharp": """```csharp
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
```"""
    },
    "16": {
        "python": """```python
def is_valid_sudoku(self, board: list[list[str]]) -> bool:
    seen = set()
    for i in range(9):
        for j in range(9):
            number = board[i][j]
            if number != '.':
                box_idx = (i // 3) * 3 + j // 3
                row_key = f"{number} in row {i}"
                col_key = f"{number} in col {j}"
                box_key = f"{number} in box {box_idx}"
                if row_key in seen or col_key in seen or box_key in seen:
                    return False
                seen.add(row_key)
                seen.add(col_key)
                seen.add(box_key)
    return True
```""",
        "csharp": """```csharp
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
```"""
    },
    "17": {
        "python": """```python
def island_perimeter(self, grid: list[list[int]]) -> int:
    perimeter = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                perimeter += 4
                if i > 0 and grid[i - 1][j] == 1: perimeter -= 2
                if j > 0 and grid[i][j - 1] == 1: perimeter -= 2
    return perimeter
```""",
        "csharp": """```csharp
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
```"""
    },
    "18": {
        "python": """```python
def max_sum(self, mat: list[list[int]], k: int) -> int:
    m, n = len(mat), len(mat[0])
    pre = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            pre[i][j] = mat[i-1][j-1] + pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1]
            
    max_val = float('-inf')
    for i in range(k, m + 1):
        for j in range(k, n + 1):
            s = pre[i][j] - pre[i-k][j] - pre[i][j-k] + pre[i-k][j-k]
            max_val = max(max_val, s)
    return max_val
```""",
        "csharp": """```csharp
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
```"""
    },
    "19": {
        "python": """```python
def num_islands(self, grid: list[list[str]]) -> int:
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                self._dfs(grid, i, j)
    return count

def _dfs(self, grid: list[list[str]], r: int, c: int) -> None:
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] == '0': return
    grid[r][c] = '0'
    self._dfs(grid, r+1, c); self._dfs(grid, r-1, c)
    self._dfs(grid, r, c+1); self._dfs(grid, r, c-1)
```""",
        "csharp": """```csharp
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
```"""
    },
    "20": {
        "python": """```python
def flip_and_invert_image(self, image: list[list[int]]) -> list[list[int]]:
    for row in image:
        left, right = 0, len(row) - 1
        while left <= right:
            row[left], row[right] = row[right] ^ 1, row[left] ^ 1
            left += 1; right -= 1
    return image
```""",
        "csharp": """```csharp
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
```"""
    },
    "21": {
        "python": """```python
def shift_grid(self, grid: list[list[int]], k: int) -> list[list[int]]:
    m, n = len(grid), len(grid[0])
    total = m * n
    k %= total
    res = [[0] * n for _ in range(m)]
    
    for r in range(m):
        for c in range(n):
            new_1d = (r * n + c + k) % total
            res[new_1d // n][new_1d % n] = grid[r][c]
    return res
```""",
        "csharp": """```csharp
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
```"""
    },
    "22": {
        "python": """```python
def exist(self, board: list[list[str]], word: str) -> bool:
    for i in range(len(board)):
        for j in range(len(board[0])):
            if self._dfs(board, i, j, word, 0): return True
    return False

def _dfs(self, b: list[list[str]], r: int, c: int, word: str, idx: int) -> bool:
    if idx == len(word): return True
    if r < 0 or c < 0 or r >= len(b) or c >= len(b[0]) or b[r][c] != word[idx]: return False
    
    temp = b[r][c]
    b[r][c] = '#'
    found = (self._dfs(b, r+1, c, word, idx+1) or self._dfs(b, r-1, c, word, idx+1) or
             self._dfs(b, r, c+1, word, idx+1) or self._dfs(b, r, c-1, word, idx+1))
    b[r][c] = temp
    return found
```""",
        "csharp": """```csharp
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
```"""
    },
    "23": {
        "python": """```python
def find_rotation(self, mat: list[list[int]], target: list[list[int]]) -> bool:
    for k in range(4):
        if mat == target: return True
        self.rotate(mat)
    return False

def rotate(self, mat: list[list[int]]) -> None:
    n = len(mat)
    for i in range(n):
        for j in range(i + 1, n):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
    for i in range(n):
        for j in range(n // 2):
            mat[i][j], mat[i][n-1-j] = mat[i][n-1-j], mat[i][j]
```""",
        "csharp": """```csharp
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
```"""
    },
    "24": {
        "python": """```python
def solution(self, cell1: str, cell2: str) -> bool:
    sum1 = (ord(cell1[0]) - ord('A')) + (ord(cell1[1]) - ord('1'))
    sum2 = (ord(cell2[0]) - ord('A')) + (ord(cell2[1]) - ord('1'))
    return (sum1 % 2) == (sum2 % 2)
```""",
        "csharp": """```csharp
public bool Solution(string cell1, string cell2) {
  int sum1 = (cell1[0] - 'A') + (cell1[1] - '1');
  int sum2 = (cell2[0] - 'A') + (cell2[1] - '1');
  return (sum1 % 2) == (sum2 % 2);
}
```"""
    },
    "25": {
        "python": """```python
def update_board(self, board: list[list[str]], click: list[int]) -> list[list[str]]:
    r, c = click[0], click[1]
    if board[r][c] == 'M':
        board[r][c] = 'X'
        return board
    self._dfs(board, r, c)
    return board

def _dfs(self, b: list[list[str]], r: int, c: int) -> None:
    if r < 0 or c < 0 or r >= len(b) or c >= len(b[0]) or b[r][c] != 'E': return
    mines = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            nr, nc = r + i, c + j
            if 0 <= nr < len(b) and 0 <= nc < len(b[0]) and b[nr][nc] == 'M':
                mines += 1
                
    if mines > 0:
        b[r][c] = str(mines)
    else:
        b[r][c] = 'B'
        for i in range(-1, 2):
            for j in range(-1, 2):
                self._dfs(b, r+i, c+j)
```""",
        "csharp": """```csharp
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
```"""
    },
    "26": {
        "python": """```python
def count_battleships(self, board: list[list[str]]) -> int:
    count = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'X':
                if i > 0 and board[i-1][j] == 'X': continue
                if j > 0 and board[i][j-1] == 'X': continue
                count += 1
    return count
```""",
        "csharp": """```csharp
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
```"""
    },
    "27": {
        "python": """```python
def box_blur(self, image: list[list[int]]) -> list[list[int]]:
    m, n = len(image), len(image[0])
    res = [[0] * (n - 2) for _ in range(m - 2)]
    
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            s = sum(image[i + di][j + dj] for di in range(-1, 2) for dj in range(-1, 2))
            res[i-1][j-1] = s // 9
    return res
```""",
        "csharp": """```csharp
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
```"""
    },
    "28": {
        "python": """```python
def convert(self, s: str, num_rows: int) -> str:
    if num_rows == 1: return s
    rows = ["" for _ in range(min(num_rows, len(s)))]
    
    cur_row = 0
    going_down = False
    
    for c in s:
        rows[cur_row] += c
        if cur_row == 0 or cur_row == num_rows - 1:
            going_down = not going_down
        cur_row += 1 if going_down else -1
        
    return "".join(rows)
```""",
        "csharp": """```csharp
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
```"""
    },
    "29": {
        "python": """```python
def robot_sim(self, commands: list[int], obstacles: list[list[int]]) -> int:
    dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
    obs = set((o[0], o[1]) for o in obstacles)
    
    x = y = dir_idx = max_dist = 0
    for cmd in commands:
        if cmd == -2: dir_idx = (dir_idx + 3) % 4
        elif cmd == -1: dir_idx = (dir_idx + 1) % 4
        else:
            for k in range(cmd):
                nx, ny = x + dx[dir_idx], y + dy[dir_idx]
                if (nx, ny) in obs: break
                x, y = nx, ny
                max_dist = max(max_dist, x*x + y*y)
    return max_dist
```""",
        "csharp": """```csharp
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
```"""
    },
    "30": {
        "python": """```python
def pacific_atlantic(self, heights: list[list[int]]) -> list[list[int]]:
    m, n = len(heights), len(heights[0])
    pac, atl = [[False] * n for _ in range(m)], [[False] * n for _ in range(m)]
    
    for i in range(m):
        self._dfs_pa(heights, pac, i, 0)
        self._dfs_pa(heights, atl, i, n-1)
    for j in range(n):
        self._dfs_pa(heights, pac, 0, j)
        self._dfs_pa(heights, atl, m-1, j)
        
    res = []
    for i in range(m):
        for j in range(n):
            if pac[i][j] and atl[i][j]:
                res.append([i, j])
    return res

def _dfs_pa(self, h, v, r, c):
    v[r][c] = True
    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(h) and 0 <= nc < len(h[0]) and not v[nr][nc] and h[nr][nc] >= h[r][c]:
            self._dfs_pa(h, v, nr, nc)
```""",
        "csharp": """```csharp
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
```"""
    },
    "31": {
        "python": """```python
def oranges_rotting(self, grid: list[list[int]]) -> int:
    from collections import deque
    q = deque()
    fresh = 0
    m, n = len(grid), len(grid[0])
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2: q.append((i, j))
            elif grid[i][j] == 1: fresh += 1
            
    if fresh == 0: return 0
    mins = 0
    
    while q:
        rotted = False
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    q.append((nr, nc))
                    rotted = True
        if rotted: mins += 1
        
    return mins if fresh == 0 else -1
```""",
        "csharp": """```csharp
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
```"""
    },
    "32": {
        "python": """```python
def solve(self, board: list[list[str]]) -> None:
    m, n = len(board), len(board[0])
    for i in range(m):
        self._dfs_s(board, i, 0)
        self._dfs_s(board, i, n-1)
    for j in range(n):
        self._dfs_s(board, 0, j)
        self._dfs_s(board, m-1, j)
        
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O': board[i][j] = 'X'
            elif board[i][j] == '#': board[i][j] = 'O'

def _dfs_s(self, b: list[list[str]], r: int, c: int) -> None:
    if r < 0 or r >= len(b) or c < 0 or c >= len(b[0]) or b[r][c] != 'O': return
    b[r][c] = '#'
    self._dfs_s(b, r+1, c); self._dfs_s(b, r-1, c)
    self._dfs_s(b, r, c+1); self._dfs_s(b, r, c-1)
```""",
        "csharp": """```csharp
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
```"""
    },
    "33": {
        "python": """```python
def minimum_effort_path(self, heights: list[list[int]]) -> int:
    left, right, ans = 0, 1000000, 1000000
    while left <= right:
        mid = (left + right) // 2
        if self._can_reach(heights, mid):
            ans = mid
            right = mid - 1
        else:
            left = mid + 1
    return ans

def _can_reach(self, h: list[list[int]], limit: int) -> bool:
    from collections import deque
    m, n = len(h), len(h[0])
    vis = [[False] * n for _ in range(m)]
    q = deque([(0, 0)])
    vis[0][0] = True
    
    while q:
        r, c = q.popleft()
        if r == m - 1 and c == n - 1: return True
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and not vis[nr][nc]:
                if abs(h[nr][nc] - h[r][c]) <= limit:
                    vis[nr][nc] = True
                    q.append((nr, nc))
    return False
```""",
        "csharp": """```csharp
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
```"""
    }
}

base_dir = r"C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\11-matrix-grid-patterns\snippets"
for k, v in translations.items():
    with open(os.path.join(base_dir, "python", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["python"])
    with open(os.path.join(base_dir, "csharp", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["csharp"])

print("Translations for chapter 11 fully applied!")
