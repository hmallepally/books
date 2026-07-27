```python
def remove_element(self, nums: list[int], val: int) -> int:
    if nums is None:
        return 0

    write = 0
    for read in range(len(nums)):
        if nums[read] != val:
            nums[write] = nums[read]
            write += 1

    return write
# Time: O(N), Space: O(1)
```