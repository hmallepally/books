```python
from typing import List

class DuplicateArrayFinder:
    """
    Finds the duplicate number in an array using Floyd's Cycle Detection.
    Time Complexity: O(N) where N is the size of the array.
    Space Complexity: O(1) auxiliary space.
    Constraint: The array must contain N + 1 elements, each between 1 and N.
    """

    def find_duplicate(self, nums: List[int]) -> int:
        # Phase 1: Detect cycle (meeting point)
        slow = nums[0]
        fast = nums[0]

        while True:
            slow = nums[slow]          # Move 1 step
            fast = nums[nums[fast]]    # Move 2 steps
            if slow == fast:
                break

        # Phase 2: Find cycle entrance (duplicate value)
        slow = nums[0] # Reset slow to start
        while slow != fast:
            slow = nums[slow] # Move 1 step
            fast = nums[fast] # Move 1 step

        return slow # The duplicate value
```
