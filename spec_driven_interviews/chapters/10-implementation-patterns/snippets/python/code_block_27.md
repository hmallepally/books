```python
def is_lucky(self, n: int) -> bool:
    s = str(n)
    mid = len(s) // 2
    sum1 = 0
    sum2 = 0

    for i in range(mid):
        sum1 += int(s[i])       # First half digit
        sum2 += int(s[i + mid]) # Second half digit

    return sum1 == sum2
# Time: O(D) where D is digit count, Space: O(D) for string conversion
```