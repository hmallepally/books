```python
from typing import Optional

class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next = None

class LinkedListReversal:
    """
    Reverses a singly linked list in-place.
    Time Complexity: O(N) where N is the number of nodes.
    Space Complexity: O(1) auxiliary space.
    """

    def reverse_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head

        while curr is not None:
            next_temp = curr.next  # 1. Save the next node
            curr.next = prev       # 2. Reverse current pointer
            prev = curr            # 3. Move prev forward
            curr = next_temp       # 4. Move curr forward

        return prev # New head node
```
