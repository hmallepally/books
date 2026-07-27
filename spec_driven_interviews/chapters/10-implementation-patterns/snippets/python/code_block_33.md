```python
def almost_increasing_sequence(self, sequence: list[int]) -> bool:
    count = 0   # Number of violations
    bad_idx = -1  # Index of first violation

    for i in range(len(sequence) - 1):
        if sequence[i] >= sequence[i + 1]:
            count += 1
            bad_idx = i
            if count > 1: return False # More than one violation

    if count == 0: return True # Already strictly increasing

    # Try removing element at bad_idx
    if bad_idx == 0 or sequence[bad_idx - 1] < sequence[bad_idx + 1]:
        return True

    # Try removing element at bad_idx + 1
    if bad_idx + 2 >= len(sequence) or sequence[bad_idx] < sequence[bad_idx + 2]:
        return True

    return False
# Time: O(N), Space: O(1)
```