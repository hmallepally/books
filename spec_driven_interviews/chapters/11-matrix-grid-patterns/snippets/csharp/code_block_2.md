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