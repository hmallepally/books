```python
import heapq

# Min-Heap (default) — smallest element first
min_heap = []
heapq.heappush(min_heap, 30)
heapq.heappush(min_heap, 10)
heapq.heappush(min_heap, 20)
min_heap[0]                # Returns 10 (smallest) — O(1)
heapq.heappop(min_heap)    # Removes 10 — O(log N)

# Max-Heap — negate values as workaround
max_heap = []
heapq.heappush(max_heap, -30)
heapq.heappush(max_heap, -10)
-max_heap[0]               # Returns 30 (largest) — O(1)

# Custom comparator — use tuples (priority, value)
pq = []
heapq.heappush(pq, (priority, value))  # Sort by first tuple element
```
