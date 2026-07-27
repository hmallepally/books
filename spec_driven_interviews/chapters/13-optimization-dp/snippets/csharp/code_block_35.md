```csharp
public int MaxArea(int[] height) {
    int maxArea = 0;
    int left = 0, right = height.Length - 1;
    while (left < right) {
        int w = right - left;
        int h = Math.Min(height[left], height[right]);
        maxArea = Math.Max(maxArea, w * h);
        if (height[left] < height[right]) left++;
        else right--;
    }
    return maxArea;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```