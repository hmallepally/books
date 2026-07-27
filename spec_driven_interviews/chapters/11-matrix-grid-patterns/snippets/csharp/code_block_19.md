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
```