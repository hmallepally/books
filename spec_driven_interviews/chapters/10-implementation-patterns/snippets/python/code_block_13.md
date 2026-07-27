```python
def neighbor_sum(self, a: list[int]) -> list[int]:
    if not a:
        return []
    n = len(a)
    b = [0] * n

    for i in range(n):
        left_val = a[i - 1] if i > 0 else 0
        right_val = a[i + 1] if i < n - 1 else 0
        b[i] = left_val + a[i] + right_val

    return b
# Time: O(N), Space: O(N) for output array
```