```python
def rob(nums: list[int]) -> int:
    if not nums:
        return 0
    prev2, prev1 = 0, 0
    
    for num in nums:
        curr = max(prev1, prev2 + num) # Skip vs Take
        prev2 = prev1
        prev1 = curr
        
    return prev1
```
