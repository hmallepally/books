```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    dq = deque()
    res = []
    
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft() # Expire
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop() # Kill weaker
        dq.append(i)
        if i >= k - 1:
            res.append(nums[dq[0]])
            
    return res
```
