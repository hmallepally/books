```python
def dp_state_compression(self, nums: list[int]) -> int:
    if not nums: return 0
    prev2 = 0 # dp[i-2]
    prev1 = nums[0] # dp[i-1]
    for i in range(1, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = curr
    return prev1
```