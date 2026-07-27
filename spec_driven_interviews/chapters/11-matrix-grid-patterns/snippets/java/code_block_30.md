```java
public List<List<Integer>> pacificAtlantic(int[][] heights) {
  int m = heights.length, n = heights[0].length;
  boolean[][] pac = new boolean[m][n], atl = new boolean[m][n];
  for (int i = 0; i < m; i++) { dfs(heights, pac, i, 0); dfs(heights, atl, i, n-1); }
  for (int j = 0; j < n; j++) { dfs(heights, pac, 0, j); dfs(heights, atl, m-1, j); }
  
  List<List<Integer>> res = new ArrayList<>();
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (pac[i][j] && atl[i][j]) res.add(Arrays.asList(i, j));
    }
  }
  return res;
}
private void dfs(int[][] h, boolean[][] v, int r, int c) {
  v[r][c] = true;
  int[][] dirs = {{1,0},{-1,0},{0,1},{0,-1}};
  for (int[] d : dirs) {
    int nr = r + d[0], nc = c + d[1];
    if (nr>=0 && nr<h.length && nc>=0 && nc<h[0].length && !v[nr][nc] && h[nr][nc] >= h[r][c])
      dfs(h, v, nr, nc);
  }
}
```
