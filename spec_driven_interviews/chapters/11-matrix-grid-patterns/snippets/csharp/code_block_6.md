```csharp
public void SetZeroes(int[][] matrix) {
  int m = matrix.Length, n = matrix[0].Length;
  bool firstColZero = false;
  // Mark zeros on first row/col
  for (int i = 0; i < m; i++) {
    if (matrix[i][0] == 0) firstColZero = true;
    for (int j = 1; j < n; j++) {
      if (matrix[i][j] == 0) {
        matrix[i][0] = 0;
        matrix[0][j] = 0;
      }
    }
  }
  // Zero out based on marks
  for (int i = 1; i < m; i++) {
    for (int j = 1; j < n; j++) {
      if (matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
    }
  }
  // Handle first row/col specifically
  if (matrix[0][0] == 0) {
    for (int j = 0; j < n; j++) matrix[0][j] = 0;
  }
  if (firstColZero) {
    for (int i = 0; i < m; i++) matrix[i][0] = 0;
  }
}
```