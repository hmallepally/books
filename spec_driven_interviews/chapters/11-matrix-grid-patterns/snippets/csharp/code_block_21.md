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
```