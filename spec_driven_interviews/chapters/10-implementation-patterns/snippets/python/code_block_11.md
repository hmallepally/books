```python
def pivot_index(self, nums: list[int]) -> int:
    if not nums:
        return -1

    total_sum = sum(nums)
    left_sum = 0
    
    for i, num in enumerate(nums):
        # right_sum = total_sum - left_sum - num
        if left_sum == total_sum - left_sum - num:
            return i
        left_sum += num

    return -1
# Time: O(N), Space: O(1)
```