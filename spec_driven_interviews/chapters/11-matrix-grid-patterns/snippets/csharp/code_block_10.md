```csharp
public bool SearchMatrix(int[][] matrix, int target) {
  int r = 0, c = matrix[0].Length - 1;
  while (r < matrix.Length && c >= 0) {
    if (matrix[r][c] == target) return true;
    else if (matrix[r][c] > target) c--;
    else r++;
  }
  return false;
}
```