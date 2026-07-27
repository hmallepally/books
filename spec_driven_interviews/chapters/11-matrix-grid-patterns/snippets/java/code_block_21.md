```java
public List<List<Integer>> shiftGrid(int[][] grid, int k) {
  int m = grid.length, n = grid[0].length;
  int total = m * n;
  k %= total;
  List<List<Integer>> res = new ArrayList<>();
  for (int i = 0; i < m; i++) {
    res.add(new ArrayList<>(Collections.nCopies(n, 0)));
  }
  for (int r = 0; r < m; r++) {
    for (int c = 0; c < n; c++) {
      int new1D = (r * n + c + k) % total;
      res.get(new1D / n).set(new1D % n, grid[r][c]);
    }
  }
  return res;
}
```
