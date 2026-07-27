```csharp
public int[][] MatrixReshape(int[][] mat, int r, int c) {
  int m = mat.Length, n = mat[0].Length;
  if (m * n != r * c) return mat; // Invalid shape
  
  int[][] res = new int[r][];
  for (int i=0; i<r; i++) res[i] = new int[c];
  
  for (int i = 0; i < m * n; i++) {
    res[i / c][i % c] = mat[i / n][i % n];
  }
  return res;
}
```