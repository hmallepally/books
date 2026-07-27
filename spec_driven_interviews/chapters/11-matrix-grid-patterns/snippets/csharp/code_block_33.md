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
```