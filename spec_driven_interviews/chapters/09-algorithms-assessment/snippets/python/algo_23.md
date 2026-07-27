```python
import heapq

def min_meeting_rooms(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])
    
    min_heap = [intervals[0][1]] # Stores end times
    
    for i in range(1, len(intervals)):
        if intervals[i][0] >= min_heap[0]:
            heapq.heappop(min_heap) # Room freed up!
        heapq.heappush(min_heap, intervals[i][1]) # Allocate room
        
    return len(min_heap)
```
