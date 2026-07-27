```python
def max_product(self, nums: list[int]) -> int:
    if not nums: return 0
    max_val = min_val = result = nums[0]
    
    for i in range(1, len(nums)):
        # If current is negative, max and min will swap roles
        if nums[i] < 0:
            max_val, min_val = min_val, max_val
            
        max_val = max(nums[i], max_val * nums[i])
        min_val = min(nums[i], min_val * nums[i])
        result = max(result, max_val)
        
    return result
# Time Complexity: O(N)
# Space Complexity: O(1)
```