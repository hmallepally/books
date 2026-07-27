```python
def longest_palindrome(self, s: str) -> str:
    start = end = 0
    for i in range(len(s)):
        len1 = self._expand(s, i, i)
        len2 = self._expand(s, i, i + 1)
        length = max(len1, len2)
        if length > end - start:
            start = i - (length - 1) // 2
            end = i + length // 2
    return s[start:end + 1]

def _expand(self, s: str, l: int, r: int) -> int:
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1; r += 1
    return r - l - 1
# Time Complexity: O(N^2) | Space Complexity: O(1)
```