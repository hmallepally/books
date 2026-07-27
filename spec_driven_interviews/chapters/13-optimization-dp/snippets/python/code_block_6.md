```python
def search(self, nums: list[int], target: int) -> int:
    if not nums: return -1
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target: return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1 # Target is in the sorted left half
            else:
                left = mid + 1 # Target must be in the right half
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1 # Target is in the sorted right half
            else:
                right = mid - 1 # Target must be in the left half
    return -1
# Time Complexity: O(log N)
# Space Complexity: O(1)
```