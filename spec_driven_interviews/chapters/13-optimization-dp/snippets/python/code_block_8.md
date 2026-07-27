```python
def longest_common_subsequence(self, text1: str, text2: str) -> int:
    if len(text1) < len(text2): return self.longest_common_subsequence(text2, text1)
    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        curr = [0] * (n + 1)
        
    return prev[n]
# Time Complexity: O(M * N)
# Space Complexity: O(min(M, N)) - Space compressed DP as taught in the vocabulary section.
```