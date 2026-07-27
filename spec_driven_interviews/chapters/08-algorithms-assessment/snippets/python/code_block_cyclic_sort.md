```python
from typing import List

class CyclicSort:
    """
    Sorts an array containing numbers from 1 to N in-place.
    Time Complexity: O(N) where N is the size of the array.
    Space Complexity: O(1) auxiliary space.
    """

    def sort(self, nums: List[int]) -> None:
        i = 0
        while i < len(nums):
            correct_index = nums[i] - 1  # Value X belongs at index X-1
            if nums[i] != nums[correct_index]:
                # Swap to correct position
                nums[i], nums[correct_index] = nums[correct_index], nums[i]
            else:
                i += 1  # Increment only when correct
```
