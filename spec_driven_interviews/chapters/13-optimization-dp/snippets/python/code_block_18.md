```python
def coin_change(self, coins: list[int], amount: int) -> int:
    dp = [amount + 1] * (amount + 1) # Fill with max invalid value
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
                
    return -1 if dp[amount] > amount else dp[amount]
# Time Complexity: O(Amount * N)
# Space Complexity: O(Amount)
```