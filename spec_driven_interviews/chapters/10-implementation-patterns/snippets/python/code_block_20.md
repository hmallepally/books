```python
def two_sum(self, nums: list[int], target: int) -> list[int]:
    seen = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return [] # Should not reach here per problem guarantee
# Time: O(N), Space: O(N)
```