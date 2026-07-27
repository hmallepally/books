```python
def move_zeroes(self, nums: list[int]) -> None:
    if not nums:
        return

    # Pass 1: Copy all non-zero elements to the front
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write] = nums[read]
            write += 1

    # Pass 2: Fill remaining positions with zeros
    while write < len(nums):
        nums[write] = 0
        write += 1
# Time: O(N), Space: O(1)
```