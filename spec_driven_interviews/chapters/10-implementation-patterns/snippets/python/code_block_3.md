```python
def first_uniq_char(self, s: str) -> int:
    if not s:
        return -1

    # Pass 1: Count frequency of each character
    counts = [0] * 256
    for char in s:
        counts[ord(char)] += 1

    # Pass 2: Find first character with frequency exactly 1
    for i, char in enumerate(s):
        if counts[ord(char)] == 1:
            return i

    return -1 # All characters repeat
# Time: O(N), Space: O(1) — the counts list is constant size
```