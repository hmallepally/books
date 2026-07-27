```python
def majority_element(self, nums: list[int]) -> int:
    candidate = nums[0]
    count = 1

    for i in range(1, len(nums)):
        if count == 0:
            candidate = nums[i]
            count = 1
        elif nums[i] == candidate:
            count += 1
        else:
            count -= 1

    return candidate
# Time: O(N), Space: O(1)
```