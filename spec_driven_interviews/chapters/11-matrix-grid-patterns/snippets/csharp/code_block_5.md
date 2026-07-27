```csharp
public IList<int> SpiralOrder(int[][] matrix) {
  List<int> res = new List<int>();
  int t = 0, b = matrix.Length - 1, l = 0, r = matrix[0].Length - 1;
  while (t <= b && l <= r) {
    for (int j = l; j <= r; j++) res.Add(matrix[t][j]); // Top
    t++;
    for (int i = t; i <= b; i++) res.Add(matrix[i][r]); // Right
    r--;
    if (t <= b) {
      for (int j = r; j >= l; j--) res.Add(matrix[b][j]); // Bottom
      b--;
    }
    if (l <= r) {
      for (int i = b; i >= t; i--) res.Add(matrix[i][l]); // Left
      l++;
    }
  }
  return res;
}
```