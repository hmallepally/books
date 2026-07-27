```java
public int[][] generateMatrix(int n) {
  int[][] mat = new int[n][n];
  int t = 0, b = n - 1, l = 0, r = n - 1;
  int val = 1;
  while (t <= b && l <= r) {
    for (int j = l; j <= r; j++) mat[t][j] = val++;
    t++;
    for (int i = t; i <= b; i++) mat[i][r] = val++;
    r--;
    if (t <= b) {
      for (int j = r; j >= l; j--) mat[b][j] = val++;
      b--;
    }
    if (l <= r) {
      for (int i = b; i >= t; i--) mat[i][l] = val++;
      l++;
    }
  }
  return mat;
}
```
