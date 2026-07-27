```csharp
public int MatrixElementsSum(int[][] matrix) {
    int rows = matrix.Length;
    int cols = matrix[0].Length;
    int total = 0;

    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            if (matrix[r][c] == 0) {
                break; // All rooms below are haunted — skip rest of column
            }
            total += matrix[r][c];
        }
    }

    return total;
}
// Time: O(rows * cols), Space: O(1)
```