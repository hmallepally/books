```python
def num_subarray_product_less_than_k(self, nums: list[int], k: int) -> int:
    if k <= 1: return 0
    prod, left, count = 1, 0, 0
    for right in range(len(nums)):
        prod *= nums[right]
        while prod >= k:
            prod //= nums[left]
            left += 1 # Shrink
        count += right - left + 1 # Add valid subarrays
    return count
# Time Complexity: O(N) | Space Complexity: O(1)
```