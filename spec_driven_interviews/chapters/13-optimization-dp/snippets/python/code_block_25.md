```python
def length_of_lis(self, nums: list[int]) -> int:
    tails = [0] * len(nums)
    size = 0
    for x in nums:
        left, right = 0, size
        while left != right:
            mid = left + (right - left) // 2
            if tails[mid] < x:
                left = mid + 1
            else:
                right = mid
        tails[left] = x
        if left == size: size += 1 # Found a larger element, expand LIS
    return size
# Time Complexity: O(N log N)
# Space Complexity: O(N)
```