```python
def rob(self, nums: list[int]) -> int:
    if not nums: return 0
    prev1 = 0 # max so far excluding current
    prev2 = 0 # max so far including current (-2)
    
    for num in nums:
        temp = max(prev1, prev2 + num) # rob or don't rob
        prev2 = prev1
        prev1 = temp
        
    return prev1
# Time Complexity: O(N)
# Space Complexity: O(1)
```