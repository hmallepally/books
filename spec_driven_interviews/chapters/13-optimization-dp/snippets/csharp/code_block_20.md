```csharp
public bool IsMatch(string s, string p) {
    int m = s.Length, n = p.Length;
    bool[][] dp = new bool[m + 1][];
    for(int i=0; i<=m; i++) dp[i] = new bool[n + 1];
    dp[0][0] = true;
    
    // Match empty string with patterns like a*b*
    for (int j = 1; j <= n; j++) {
        if (p[j - 1] == '*') dp[0][j] = dp[0][j - 2];
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (p[j - 1] == '.' || p[j - 1] == s[i - 1]) {
                dp[i][j] = dp[i - 1][j - 1]; // Single char match
            } else if (p[j - 1] == '*') {
                dp[i][j] = dp[i][j - 2]; // Match zero times
                // If preceding char matches, match one or more times
                if (p[j - 2] == '.' || p[j - 2] == s[i - 1]) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                }
            }
        }
    }
    return dp[m][n];
}
// Time Complexity: O(M * N)
// Space Complexity: O(M * N)
```