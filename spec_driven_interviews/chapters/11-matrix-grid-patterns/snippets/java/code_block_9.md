```java
public void rotateCounter(int[][] matrix) {
  int n = matrix.length;
  // Transpose
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      int temp = matrix[i][j];
      matrix[i][j] = matrix[j][i];
      matrix[j][i] = temp;
    }
  }
  // Reverse each column
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n / 2; i++) {
      int temp = matrix[i][j];
      matrix[i][j] = matrix[n - 1 - i][j];
      matrix[n - 1 - i][j] = temp;
    }
  }
}
```
