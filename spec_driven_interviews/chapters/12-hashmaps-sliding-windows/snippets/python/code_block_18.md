```python
def check_subarray_sum(self, nums: list[int], k: int) -> bool:
    hash_map = {0: -1}
    total_sum = 0
    for i, num in enumerate(nums):
        total_sum += num
        mod = total_sum if k == 0 else total_sum % k
        if mod in hash_map:
            if i - hash_map[mod] > 1: return True # Length >= 2
        else:
            hash_map[mod] = i
    return False
# Time Complexity: O(N) | Space Complexity: O(min(N, K))
```