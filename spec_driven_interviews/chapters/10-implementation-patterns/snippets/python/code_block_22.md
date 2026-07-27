```python
def plus_one(self, digits: list[int]) -> list[int]:
    for i in range(len(digits) - 1, -1, -1):
        digits[i] += 1
        if digits[i] < 10:
            return digits # No further carry needed
        digits[i] = 0 # Carry to next position

    # All digits were 9 — need a new array [1, 0, 0, ..., 0]
    return [1] + [0] * len(digits)
# Time: O(N), Space: O(1) amortized (O(N) only for all-9s edge case)
```