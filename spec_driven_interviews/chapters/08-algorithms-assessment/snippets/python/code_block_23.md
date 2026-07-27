```python
def single_number(nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num  # Duplicates cancel: a ^ a = 0, 0 ^ b = b
    return result
```
