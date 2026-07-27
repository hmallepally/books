```python
def number_of_subarrays(self, nums: list[int], k: int) -> int:
    from collections import defaultdict
    hash_map = defaultdict(int)
    hash_map[0] = 1
    total_sum = count = 0
    for num in nums:
        total_sum += num % 2
        count += hash_map[total_sum - k]
        hash_map[total_sum] += 1
    return count
# Time Complexity: O(N) | Space Complexity: O(N)
```