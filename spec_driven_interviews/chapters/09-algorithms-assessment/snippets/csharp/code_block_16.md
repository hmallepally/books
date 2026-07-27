```csharp
public IList<int> SpiralOrder(int[][] matrix) {
    var result = new List<int>();
    if (matrix.Length == 0) return result;
    int r1 = 0, r2 = matrix.Length - 1;
    int c1 = 0, c2 = matrix[0].Length - 1;
    while (r1 <= r2 && c1 <= c2) {
        for (int c = c1; c <= c2; c++) result.Add(matrix[r1][c]);
        for (int r = r1 + 1; r <= r2; r++) result.Add(matrix[r][c2]);
        if (r1 < r2 && c1 < c2) {
            for (int c = c2 - 1; c > c1; c--) result.Add(matrix[r2][c]);
            for (int r = r2; r > r1; r--) result.Add(matrix[r][c1]);
        }
        r1++;
        r2--;
        c1++;
        c2--;
    }
    return result;
}
```