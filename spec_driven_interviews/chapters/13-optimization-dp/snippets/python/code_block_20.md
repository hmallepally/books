```python
def is_match(self, s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Match empty string with patterns like a*b*
    for j in range(1, n + 1):
        if p[j - 1] == '*': dp[0][j] = dp[0][j - 2]
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1] # Single char match
            elif p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2] # Match zero times
                # If preceding char matches, match one or more times
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
                    
    return dp[m][n]
# Time Complexity: O(M * N)
# Space Complexity: O(M * N)
```