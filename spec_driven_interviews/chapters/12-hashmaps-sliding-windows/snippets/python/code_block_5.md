```python
def length_of_longest_substring(self, s: str) -> int:
    char_set = set()
    left = max_val = 0
    for right in range(len(s)):
        # Contract if duplicate found
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right]) # Add current char
        max_val = max(max_val, right - left + 1)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(min(N, M))
```