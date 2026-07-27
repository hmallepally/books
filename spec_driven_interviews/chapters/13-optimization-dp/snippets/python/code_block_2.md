```python
def next_greater_element(self, nums: list[int]) -> list[int]:
    n = len(nums)
    result = [-1] * n
    stack = [] # stores indices
    for i in range(n):
        # Maintain strictly decreasing stack
        while stack and nums[i] > nums[stack[-1]]:
            prev_index = stack.pop()
            result[prev_index] = nums[i] # Found next greater!
        stack.append(i)
    return result
```