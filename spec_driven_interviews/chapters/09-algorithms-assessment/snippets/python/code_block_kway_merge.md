```python
import heapq
from typing import List

class KWayMerge:
    """
    Merges K sorted lists into one sorted list using a Min-Heap.
    Time Complexity: O(N log K) where N is total elements, K is number of lists.
    Space Complexity: O(K) auxiliary space for the heap.
    """

    def merge_k_lists(self, lists: List[List[int]]) -> List[int]:
        min_heap = []
        result = []

        # 1. Initialize heap with the first element of each list
        # Heap stores tuples: (value, list_index, element_index)
        for i in range(len(lists)):
            if lists[i]:
                # We push the tuple into the heap. heapq compares elements element-by-element,
                # so it will sort primarily by lists[i][0] (the value).
                heapq.heappush(min_heap, (lists[i][0], i, 0))

        # 2. Extract min and push the next element from that list
        while min_heap:
            val, list_idx, elem_idx = heapq.heappop(min_heap)
            result.append(val)

            next_elem_idx = elem_idx + 1
            if next_elem_idx < len(lists[list_idx]):
                heapq.heappush(min_heap, (lists[list_idx][next_elem_idx], list_idx, next_elem_idx))

        return result
```
