```csharp
public int CoinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Array.Fill(dp, amount + 1); // Fill with max invalid value
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        foreach (int coin in coins) {
            if (i >= coin) {
                dp[i] = Math.Min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
// Time Complexity: O(Amount * N)
// Space Complexity: O(Amount)
```