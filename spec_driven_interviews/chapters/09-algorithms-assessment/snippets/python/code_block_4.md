```python
def maxArea(height: list[int]) -> int:
    max_val = 0
    left = 0
    right = len(height) - 1
    while left < right:
        width = right - left
        current_height = min(height[left], height[right])
        max_val = max(max_val, width * current_height)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_val
```