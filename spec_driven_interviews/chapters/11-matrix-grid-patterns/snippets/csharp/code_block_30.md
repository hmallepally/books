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
```