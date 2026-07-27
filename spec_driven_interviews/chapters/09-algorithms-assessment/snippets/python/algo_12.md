```python
def backtrack(res: list[list[int]], path: list[int], nums: list[int], used: list[bool]) -> None:
    if len(path) == len(nums):
        res.append(list(path))
        return
        
    for i in range(len(nums)):
        if used[i]:
            continue
        used[i] = True
        path.append(nums[i])
        backtrack(res, path, nums, used) # Recurse
        path.pop() # Undo (backtrack)
        used[i] = False
```
