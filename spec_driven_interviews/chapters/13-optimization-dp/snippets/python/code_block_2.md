```python
def next_greater_element(self, nums: list[int]) -> list[int]:
    n = len(nums)
    result = [-1] * n # <1>
    stack = [] # <2>
    for i in range(n):
        # Maintain strictly decreasing stack
        while stack and nums[i] > nums[stack[-1]]:
            prev_index = stack.pop()
            result[prev_index] = nums[i] # <3> Found next greater element
        stack.append(i) # <4>
    return result
```