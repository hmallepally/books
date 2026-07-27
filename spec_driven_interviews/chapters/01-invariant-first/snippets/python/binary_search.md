```python
def binary_search(nums: list[int], target: int) -> int:
    # 1. Enforce Pre-conditions
    if not nums:
        return -1

    left = 0
    right = len(nums) - 1

    # Maintain Invariant: target is in nums[left...right]
    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid  # Post-condition satisfied
        elif nums[mid] < target:
            left = mid + 1  # Invariant maintained
        else:
            right = mid - 1  # Invariant maintained

    return -1  # Search range is empty -> target not in nums
```