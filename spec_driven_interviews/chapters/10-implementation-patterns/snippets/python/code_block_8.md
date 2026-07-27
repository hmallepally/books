```python
def single_number(self, nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num # Pairs cancel, unique value survives
    return result
# Time: O(N), Space: O(1)
```