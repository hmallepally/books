```csharp
public int MaxArea(int[] height) {
    int maxVal = 0;
    int left = 0;
    int right = height.Length - 1;
    while (left < right) {
        int width = right - left;
        int currentHeight = Math.Min(height[left], height[right]);
        maxVal = Math.Max(maxVal, width * currentHeight);
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    return maxVal;
}
```