```python
def is_alternating_parity(self, nums: list[int]) -> bool:
    if not nums or len(nums) <= 1:
        return True

    for i in range(len(nums) - 1):
        if (abs(nums[i]) % 2) == (abs(nums[i + 1]) % 2):
            return False

    return True
# Time: O(N), Space: O(1)
```