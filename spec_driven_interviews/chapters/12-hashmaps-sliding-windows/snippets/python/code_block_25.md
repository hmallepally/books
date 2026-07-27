```python
def min_sub_array_len(self, target: int, nums: list[int]) -> int:
    left = total_sum = 0
    min_val = float('inf')
    for right in range(len(nums)):
        total_sum += nums[right]
        while total_sum >= target:
            min_val = min(min_val, right - left + 1)
            total_sum -= nums[left]
            left += 1
    return 0 if min_val == float('inf') else min_val
# Time Complexity: O(N) | Space Complexity: O(1)
```