```java
public int maxArea(int[] height) {
    int maxVal = 0;
    int left = 0;
    int right = height.length - 1;
    while (left < right) {
        int width = right - left;
        int currentHeight = Math.min(height[left], height[right]);
        maxVal = Math.max(maxVal, width * currentHeight);
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    return maxVal;
}
```