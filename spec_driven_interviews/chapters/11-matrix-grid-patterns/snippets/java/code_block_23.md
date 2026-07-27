```java
public boolean findRotation(int[][] mat, int[][] target) {
  for (int k = 0; k < 4; k++) {
    if (Arrays.deepEquals(mat, target)) return true;
    rotate(mat); // uses function from Problem 1
  }
  return false;
}
private void rotate(int[][] mat) {
  int n = mat.length;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      int t = mat[i][j]; mat[i][j] = mat[j][i]; mat[j][i] = t;
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n/2; j++) {
      int t = mat[i][j]; mat[i][j] = mat[i][n-1-j]; mat[i][n-1-j] = t;
    }
  }
}
```
