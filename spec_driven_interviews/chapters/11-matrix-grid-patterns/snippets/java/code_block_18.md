```java
public int maxSum(int[][] mat, int k) {
  int m = mat.length, n = mat[0].length;
  int[][] pre = new int[m + 1][n + 1];
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      pre[i][j] = mat[i-1][j-1] + pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1];
    }
  }
  int max = Integer.MIN_VALUE;
  for (int i = k; i <= m; i++) {
    for (int j = k; j <= n; j++) {
      int sum = pre[i][j] - pre[i-k][j] - pre[i][j-k] + pre[i-k][j-k];
      max = Math.max(max, sum);
    }
  }
  return max;
}
```
