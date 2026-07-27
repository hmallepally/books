```python
def subarrays_with_k_distinct(self, nums: list[int], k: int) -> int:
    return self._at_most_k(nums, k) - self._at_most_k(nums, k - 1)

def _at_most_k(self, nums: list[int], k: int) -> int:
    count = [0] * (len(nums) + 1)
    left = res = distinct = 0
    for right in range(len(nums)):
        if count[nums[right]] == 0: distinct += 1
        count[nums[right]] += 1
        while distinct > k:
            count[nums[left]] -= 1
            if count[nums[left]] == 0: distinct -= 1
            left += 1
        res += right - left + 1
    return res
# Time Complexity: O(N) | Space Complexity: O(N)
```