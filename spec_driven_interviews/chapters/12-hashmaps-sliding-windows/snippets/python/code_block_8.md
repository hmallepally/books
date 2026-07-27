```python
def find_anagrams(self, s: str, p: str) -> list[int]:
    res = []
    if len(s) < len(p): return res
    p_count, s_count = [0] * 26, [0] * 26
    for c in p: p_count[ord(c) - ord('a')] += 1
    for i in range(len(s)):
        s_count[ord(s[i]) - ord('a')] += 1
        if i >= len(p): s_count[ord(s[i - len(p)]) - ord('a')] -= 1 # Contract
        if p_count == s_count: res.append(i - len(p) + 1) # Match
    return res
# Time Complexity: O(N) | Space Complexity: O(1)
```