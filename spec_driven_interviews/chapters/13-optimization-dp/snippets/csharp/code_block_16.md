```csharp
public int MaximalRectangle(char[][] matrix) {
    if (matrix == null || matrix.Length == 0) return 0;
    int cols = matrix[0].Length;
    int[] heights = new int[cols];
    int maxArea = 0;
    
    foreach (char[] row in matrix) {
        // Update histogram heights
        for (int c = 0; c < cols; c++) {
            heights[c] = (row[c] == '1') ? heights[c] + 1 : 0;
        }
        maxArea = Math.Max(maxArea, MaxHistogram(heights));
    }
    return maxArea;
}

private int MaxHistogram(int[] heights) {
    Stack<int> stack = new Stack<int>();
    int max = 0, n = heights.Length;
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        while (stack.Count > 0 && h < heights[stack.Peek()]) {
            int height = heights[stack.Pop()];
            int width = stack.Count == 0 ? i : i - stack.Peek() - 1;
            max = Math.Max(max, height * width);
        }
        stack.Push(i);
    }
    return max;
}
// Time Complexity: O(R * C)
// Space Complexity: O(C)
```