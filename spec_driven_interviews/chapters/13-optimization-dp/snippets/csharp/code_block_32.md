```csharp
public int LargestRectangleArea(int[] heights) {
    Stack<int> stack = new Stack<int>();
    int maxArea = 0, n = heights.Length;
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        while (stack.Count > 0 && h < heights[stack.Peek()]) {
            int height = heights[stack.Pop()];
            int width = stack.Count == 0 ? i : i - stack.Peek() - 1;
            maxArea = Math.Max(maxArea, height * width);
        }
        stack.Push(i);
    }
    return maxArea;
}
// Time Complexity: O(N)
// Space Complexity: O(N)
```