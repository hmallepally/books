```python
def group_anagrams(self, strs: list[str]) -> list[list[str]]:
    from collections import defaultdict
    hash_map = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s: count[ord(c) - ord('a')] += 1 # Build signature
        key = tuple(count)
        hash_map[key].append(s)
    return list(hash_map.values())
# Time Complexity: O(N * L) | Space Complexity: O(N * L)
```