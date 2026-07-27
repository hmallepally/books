```java
// Construction
int[][] sum = new int[R + 1][C + 1];
for (int r = 1; r <= R; r++) {
  for (int c = 1; c <= C; c++) {
    sum[r][c] = matrix[r-1][c-1] + sum[r-1][c] + sum[r][c-1] - sum[r-1][c-1];
  }
}
// Query from (r1, c1) to (r2, c2)
int query(int r1, int c1, int r2, int c2) {
  return sum[r2+1][c2+1] - sum[r1][c2+1] - sum[r2+1][c1] + sum[r1][c1];
}
```
