```python
def max_sliding_window(self, nums: list[int], k: int) -> list[int]:
    if not nums or k <= 0: return []
    n = len(nums)
    res = [0] * (n - k + 1)
    res_index = 0
    from collections import deque
    q = deque()
    
    for i in range(n):
        # Remove indices outside the current window
        if q and q[0] < i - k + 1:
            q.popleft()
        # Remove smaller elements (maintain decreasing order)
        while q and nums[q[-1]] < nums[i]:
            q.pop()
        q.append(i)
        
        # Record max for the window
        if i >= k - 1:
            res[res_index] = nums[q[0]]
            res_index += 1
            
    return res
# Time Complexity: O(N) since each element is pushed/popped at most once
# Space Complexity: O(K) for the deque
```