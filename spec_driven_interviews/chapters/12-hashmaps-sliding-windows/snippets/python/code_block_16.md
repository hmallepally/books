```python
def character_replacement(self, s: str, k: int) -> int:
    count = [0] * 26
    max_count = left = max_len = 0
    for right in range(len(s)):
        idx = ord(s[right]) - ord('A')
        count[idx] += 1
        max_count = max(max_count, count[idx])
        if right - left + 1 - max_count > k: # Invalid window
            count[ord(s[left]) - ord('A')] -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
# Time Complexity: O(N) | Space Complexity: O(1)
```