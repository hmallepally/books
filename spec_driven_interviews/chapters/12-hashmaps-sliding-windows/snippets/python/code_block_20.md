```python
def find_duplicates(self, nums: list[int]) -> list[int]:
    res = []
    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0: res.append(abs(num)) # Found duplicate
        else: nums[idx] = -nums[idx] # Mark seen
    return res
# Time Complexity: O(N) | Space Complexity: O(1)
```