```csharp
public int[][] Transpose(int[][] matrix) {
  int r = matrix.Length;
  int c = matrix[0].Length;
  int[][] ans = new int[c][];
  for (int i=0; i<c; i++) ans[i] = new int[r];
  
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      ans[j][i] = matrix[i][j];
    }
  }
  return ans;
}
```