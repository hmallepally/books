```python
def max_frequency(self, nums: list[int], k: int) -> int:
    nums.sort()
    left = total_sum = 0
    for right in range(len(nums)):
        total_sum += nums[right]
        if nums[right] * (right - left + 1) - total_sum > k:
            total_sum -= nums[left]
            left += 1
    return len(nums) - left
# Time Complexity: O(N log N) | Space Complexity: O(1)
```