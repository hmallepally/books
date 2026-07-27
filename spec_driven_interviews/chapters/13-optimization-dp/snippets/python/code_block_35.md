```python
def max_area(self, height: list[int]) -> int:
    max_area = 0
    left, right = 0, len(height) - 1
    while left < right:
        w = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, w * h)
        if height[left] < height[right]: left += 1
        else: right -= 1
    return max_area
# Time Complexity: O(N)
# Space Complexity: O(1)
```