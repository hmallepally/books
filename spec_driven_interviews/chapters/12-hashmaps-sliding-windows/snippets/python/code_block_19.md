```python
def longest_ones(self, nums: list[int], k: int) -> int:
    left = 0
    for right in range(len(nums)):
        if nums[right] == 0: k -= 1
        if k < 0: # Over budget
            if nums[left] == 0: k += 1
            left += 1
    return len(nums) - left # Trick to return max valid length seen
# Time Complexity: O(N) | Space Complexity: O(1)
```