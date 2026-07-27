```python
from typing import Optional

class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next = None

class CycleDetector:
    """
    Implements the Fast/Slow Pointer (Tortoise and Hare) pattern to detect cycles
    in a singly linked list.
    Time Complexity: O(N) where N is the number of nodes.
    Space Complexity: O(1) auxiliary space.
    """

    def has_cycle(self, head: Optional[ListNode]) -> bool:
        """
        Detects if a linked list contains a cycle.
        Moves slow pointer by 1 step, fast pointer by 2 steps.
        """
        slow = head
        fast = head

        while fast is not None and fast.next is not None:
            slow = slow.next          # Tortoise: 1 step
            fast = fast.next.next     # Hare: 2 steps

            if slow == fast:
                return True # Fast pointer caught up to slow pointer -> cycle!

        return False # Fast pointer reached the end -> no cycle
```
