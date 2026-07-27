```python
def first_unique_char(s: str) -> int:
    counts = [0] * 256
    for c in s:
        counts[ord(c)] += 1
    for i, c in enumerate(s):
        if counts[ord(c)] == 1:
            return i
    return -1
```
