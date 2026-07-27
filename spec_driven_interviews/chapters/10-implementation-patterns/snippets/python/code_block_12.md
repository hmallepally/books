```python
def is_monotonic(self, nums: list[int]) -> bool:
    if not nums or len(nums) <= 2:
        return True

    increasing = True
    decreasing = True

    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]: increasing = False
        if nums[i] < nums[i + 1]: decreasing = False

    return increasing or decreasing
# Time: O(N), Space: O(1)
```