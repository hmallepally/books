```python
def min_window(self, s: str, t: str) -> str:
    char_map = [0] * 128
    for c in t: char_map[ord(c)] += 1
    left, count = 0, len(t)
    min_len, min_start = float('inf'), 0
    
    for right in range(len(s)):
        if char_map[ord(s[right])] > 0: count -= 1 # Found required char
        char_map[ord(s[right])] -= 1
        
        while count == 0: # All chars found
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left
            char_map[ord(s[left])] += 1
            if char_map[ord(s[left])] > 0: count += 1 # Removed required char
            left += 1
            
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]
# Time Complexity: O(N) | Space Complexity: O(1)
```