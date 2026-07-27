```python
def can_partition(self, nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2 != 0: return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        # Iterate backwards to avoid reusing the same element
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
            
    return dp[target]
# Time Complexity: O(N * Target)
# Space Complexity: O(Target)
```