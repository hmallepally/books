```python
# Standard Binary Search
def binary_search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target: return mid
        elif nums[mid] < target: left = mid + 1
        else: right = mid - 1
    return -1

# Binary Search on Answer Space (Leftmost valid)
def binary_search_answer_space(min_val: int, max_val: int) -> int:
    left, right = min_val, max_val
    best = -1
    while left <= right:
        mid = left + (right - left) // 2
        if is_valid(mid):
            best = mid
            right = mid - 1 # Try to find a smaller valid answer
        else:
            left = mid + 1
    return best
```