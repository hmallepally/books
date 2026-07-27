```java
public int minimumEffortPath(int[][] heights) {
  int left = 0, right = 1000000, ans = right;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (canReach(heights, mid)) {
      ans = mid; right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  return ans;
}
private boolean canReach(int[][] h, int limit) {
  int m = h.length, n = h[0].length;
  boolean[][] vis = new boolean[m][n];
  Queue<int[]> q = new ArrayDeque<>();
  q.offer(new int[]{0, 0}); vis[0][0] = true;
  int[][] dirs = {{1,0},{-1,0},{0,1},{0,-1}};
  
  while (!q.isEmpty()) {
    int[] curr = q.poll();
    if (curr[0] == m-1 && curr[1] == n-1) return true;
    for (int[] d : dirs) {
      int r = curr[0]+d[0], c = curr[1]+d[1];
      if (r>=0 && r<m && c>=0 && c<n && !vis[r][c]) {
        if (Math.abs(h[r][c] - h[curr[0]][curr[1]]) <= limit) {
          vis[r][c] = true;
          q.offer(new int[]{r, c});
        }
      }
    }
  }
  return false;
}
```
