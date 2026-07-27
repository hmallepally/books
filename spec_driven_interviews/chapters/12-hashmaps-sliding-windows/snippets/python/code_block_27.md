```python
def contains_nearby_duplicate(self, nums: list[int], k: int) -> bool:
    hash_set = set()
    for i in range(len(nums)):
        if i > k: hash_set.remove(nums[i - k - 1])
        if nums[i] in hash_set: return True
        hash_set.add(nums[i])
    return False
# Time Complexity: O(N) | Space Complexity: O(K)
```