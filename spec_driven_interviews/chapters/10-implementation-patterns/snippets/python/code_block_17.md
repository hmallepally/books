```python
def are_occurrences_equal(self, s: str) -> bool:
    if not s:
        return True

    from collections import Counter
    counts = Counter(s)
    
    expected = 0
    for count in counts.values():
        if count > 0:
            if expected == 0: expected = count
            elif count != expected: return False

    return True
# Time: O(N), Space: O(1)
```