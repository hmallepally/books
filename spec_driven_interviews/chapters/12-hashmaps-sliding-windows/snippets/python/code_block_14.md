```python
def check_inclusion(self, s1: str, s2: str) -> bool:
    if len(s1) > len(s2): return False
    s1_map, s2_map = [0] * 26, [0] * 26
    for c in s1: s1_map[ord(c) - ord('a')] += 1
    for i in range(len(s2)):
        s2_map[ord(s2[i]) - ord('a')] += 1
        if i >= len(s1): s2_map[ord(s2[i - len(s1)]) - ord('a')] -= 1
        if s1_map == s2_map: return True
    return False
# Time Complexity: O(N) | Space Complexity: O(1)
```