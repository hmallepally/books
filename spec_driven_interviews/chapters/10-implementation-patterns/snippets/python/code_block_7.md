```python
def remove_duplicates(self, nums: list[int]) -> int:
    if not nums:
        return 0

    write = 1 # First element is always unique
    for read in range(1, len(nums)):
        if nums[read] != nums[write - 1]:
            nums[write] = nums[read]
            write += 1

    return write
# Time: O(N), Space: O(1)
```