```python
def find_max_length(self, nums: list[int]) -> int:
    hash_map = {0: -1}
    total_sum = max_val = 0
    for i, num in enumerate(nums):
        total_sum += -1 if num == 0 else 1 # Map 0 to -1
        if total_sum in hash_map:
            max_val = max(max_val, i - hash_map[total_sum])
        else:
            hash_map[total_sum] = i # Store first occurrence
    return max_val
# Time Complexity: O(N) | Space Complexity: O(N)
```