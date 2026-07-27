```python
from typing import List

class TwoPointerSolver:
    """
    Implements the Two-Pointer pattern to find two numbers that sum to a target
    in a 1-indexed sorted array.
    Time Complexity: O(N) where N is the size of the array.
    Space Complexity: O(1) auxiliary space.
    """

    def find_matching_numbers(self, numbers: List[int], target: int) -> List[int]:
        """
        Finds indices of the two numbers that add up to the target.
        Uses two pointers moving from opposite ends inward.
        """
        start = 1
        last = len(numbers)

        while start < last:
            current_sum = numbers[start - 1] + numbers[last - 1]
            if current_sum == target:
                return [start, last]
            if current_sum < target:
                start += 1
            else:
                last -= 1
        return [0, 0] # Returns [0, 0] if no match is found
```
