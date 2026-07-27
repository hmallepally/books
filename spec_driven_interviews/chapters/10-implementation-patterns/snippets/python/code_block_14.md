```python
def max_sum_subarray(self, nums: list[int], k: int) -> int:
    if not nums or len(nums) < k or k <= 0:
        return 0

    # Initialize sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide the window: add right element, remove left element
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum
# Time: O(N), Space: O(1)
```