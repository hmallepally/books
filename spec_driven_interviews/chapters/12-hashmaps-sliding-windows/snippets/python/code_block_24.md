```python
def first_missing_positive(self, nums: list[int]) -> int:
    i = 0
    while i < len(nums):
        # Swap to correct position if valid
        if 0 < nums[i] <= len(nums) and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        else:
            i += 1
            
    for i in range(len(nums)):
        if nums[i] != i + 1: return i + 1 # Missing
        
    return len(nums) + 1
# Time Complexity: O(N) | Space Complexity: O(1)
```