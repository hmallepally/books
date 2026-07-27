```python
def max_coins(self, nums: list[int]) -> int:
    n = len(nums)
    arr = [1] + nums + [1] # Padding with 1s
    
    dp = [[0] * (n + 2) for _ in range(n + 2)]
    
    # len_ is the length of the interval strictly between i and j
    for len_ in range(1, n + 1):
        for i in range(n - len_ + 1):
            j = i + len_ + 1
            # k is the index of the LAST balloon to burst in (i, j)
            for k in range(i + 1, j):
                coins = arr[i] * arr[k] * arr[j] + dp[i][k] + dp[k][j]
                dp[i][j] = max(dp[i][j], coins)
                
    return dp[0][n + 1]
# Time Complexity: O(N^3)
# Space Complexity: O(N^2)
```