```python
from collections import deque

d = deque()

# --- As a double-ended queue ---
d.appendleft(1)            # Add to front — O(1)
d.append(2)                # Add to back — O(1)
d[0]                       # View front without removing — O(1)
d[-1]                      # View back without removing — O(1)
d.popleft()                # Remove from front — O(1)
d.pop()                    # Remove from back — O(1)

# --- As a Stack (LIFO) ---
d.append(42)               # Push onto stack (adds to right)
d[-1]                      # View top element
d.pop()                    # Pop from stack (removes from right)

# --- As a Queue (FIFO) ---
d.append(42)               # Enqueue (adds to right)
d[0]                       # View head
d.popleft()                # Dequeue (removes from left)

not d                      # Check if empty (Pythonic)
len(d)                     # Current element count
```
