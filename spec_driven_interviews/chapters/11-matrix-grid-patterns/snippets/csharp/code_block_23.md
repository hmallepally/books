```csharp
public bool FindRotation(int[][] mat, int[][] target) {
  for (int k = 0; k < 4; k++) {
    if (AreEqual(mat, target)) return true;
    Rotate(mat); 
  }
  return false;
}
private bool AreEqual(int[][] mat, int[][] target) {
  for(int i=0; i<mat.Length; i++)
    for(int j=0; j<mat[i].Length; j++)
      if (mat[i][j] != target[i][j]) return false;
  return true;
}
private void Rotate(int[][] mat) {
  int n = mat.Length;
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