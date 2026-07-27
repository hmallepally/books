```csharp
public int MaxSum(int[][] mat, int k) {
  int m = mat.Length, n = mat[0].Length;
  int[][] pre = new int[m + 1][];
  for (int i=0; i<=m; i++) pre[i] = new int[n + 1];
  
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      pre[i][j] = mat[i-1][j-1] + pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1];
    }
  }
  int max = int.MinValue;
  for (int i = k; i <= m; i++) {
    for (int j = k; j <= n; j++) {
      int sum = pre[i][j] - pre[i-k][j] - pre[i][j-k] + pre[i-k][j-k];
      max = Math.Max(max, sum);
    }
  }
  return max;
}
```