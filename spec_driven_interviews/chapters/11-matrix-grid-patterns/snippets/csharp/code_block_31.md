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
```