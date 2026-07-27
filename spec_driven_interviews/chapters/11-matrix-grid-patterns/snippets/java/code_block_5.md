```java
public List<Integer> spiralOrder(int[][] matrix) {
  List<Integer> res = new ArrayList<>();
  int t = 0, b = matrix.length - 1, l = 0, r = matrix[0].length - 1;
  while (t <= b && l <= r) {
    for (int j = l; j <= r; j++) res.add(matrix[t][j]); // Top
    t++;
    for (int i = t; i <= b; i++) res.add(matrix[i][r]); // Right
    r--;
    if (t <= b) {
      for (int j = r; j >= l; j--) res.add(matrix[b][j]); // Bottom
      b--;
    }
    if (l <= r) {
      for (int i = b; i >= t; i--) res.add(matrix[i][l]); // Left
      l++;
    }
  }
  return res;
}
```
