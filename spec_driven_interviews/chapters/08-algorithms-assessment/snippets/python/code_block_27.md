```python
from collections import deque

queue = deque()
queue.append(1)            # Enqueue — O(1)
queue.append(2)
queue[0]                   # View head (returns 1) — O(1)
queue.popleft()            # Dequeue (removes 1) — O(1)
not queue                  # Check if empty (Pythonic)
len(queue)                 # Current element count
```
