```python
from collections import deque

# Python has no built-in LinkedList; use deque for similar behavior
linked = deque()
linked.appendleft(1)       # Add to head — O(1)
linked.append(2)           # Add to tail — O(1)
linked[0]                  # View head — O(1)
linked[-1]                 # View tail — O(1)
linked.popleft()           # Remove head — O(1)
linked.pop()               # Remove tail — O(1)
# For true linked-list pointer problems, define a ListNode class
```
