```python
def maximum_unique_subarray(self, nums: list[int]) -> int:
    char_set = set()
    total_sum = max_val = left = 0
    for right in range(len(nums)):
        while nums[right] in char_set:
            char_set.remove(nums[left])
            total_sum -= nums[left] # Remove duplicate
            left += 1
        char_set.add(nums[right])
        total_sum += nums[right]
        max_val = max(max_val, total_sum)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(N)
```