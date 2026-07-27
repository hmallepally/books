```python
import heapq
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    freq_map = Counter(nums)
    
    min_heap = []
    for num, count in freq_map.items():
        heapq.heappush(min_heap, (count, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)
            
    return [num for count, num in min_heap]
```
