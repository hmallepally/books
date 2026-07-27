```csharp
public int NumDecodings(string s) {
    if (string.IsNullOrEmpty(s) || s[0] == '0') return 0;
    int n = s.Length;
    int[] dp = new int[n + 1];
    dp[0] = 1; 
    dp[1] = 1;
    
    for (int i = 2; i <= n; i++) {
        int oneDigit = int.Parse(s.Substring(i - 1, 1));
        int twoDigits = int.Parse(s.Substring(i - 2, 2));
        
        if (oneDigit >= 1 && oneDigit <= 9) {
            dp[i] += dp[i - 1];
        }
        if (twoDigits >= 10 && twoDigits <= 26) {
            dp[i] += dp[i - 2];
        }
    }
    return dp[n];
}
// Time Complexity: O(N)
// Space Complexity: O(N) which can be optimized to O(1)
```