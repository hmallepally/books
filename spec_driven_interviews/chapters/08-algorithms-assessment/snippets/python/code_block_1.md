```python
from collections import deque
from typing import List

class SlidingWindowSolver:
    """
    Implements the Sliding Window Maximum algorithm using a Monotonic Deque.
    """
    def max_sliding_window(self, nums: List[int], k: int) -> List[int]:
        if not nums or k <= 0:
            return []

        n = len(nums)
        result = []
        q = deque()  # Stores array indices

        for i in range(n):
            # 1. Remove indices that are out of the current window boundary
            if q and q[0] < i - k + 1:
                q.popleft()

            # 2. Maintain monotonic invariant: Remove indices of elements smaller
            # than the current element from the tail of the deque
            while q and nums[q[-1]] < nums[i]:
                q.pop()

            # 3. Add current element's index to the tail
            q.append(i)

            # 4. If window size has reached K, store the maximum in the result
            if i >= k - 1:
                result.append(nums[q[0]])

        return result
```
