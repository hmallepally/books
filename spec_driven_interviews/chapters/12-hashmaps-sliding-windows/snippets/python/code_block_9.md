```python
def length_of_longest_substring_k_distinct(self, s: str, k: int) -> int:
    from collections import defaultdict
    hash_map = defaultdict(int)
    left = max_val = 0
    for right in range(len(s)):
        c = s[right]
        hash_map[c] += 1
        while len(hash_map) > k: # Invariant broken
            left_char = s[left]
            left += 1
            hash_map[left_char] -= 1
            if hash_map[left_char] == 0: del hash_map[left_char]
        max_val = max(max_val, right - left + 1)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(K)
```