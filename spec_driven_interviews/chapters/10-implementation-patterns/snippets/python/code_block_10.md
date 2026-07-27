```python
def reverse_string(self, s: list[str]) -> None:
    if not s or len(s) <= 1:
        return

    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
# Time: O(N), Space: O(1)
```