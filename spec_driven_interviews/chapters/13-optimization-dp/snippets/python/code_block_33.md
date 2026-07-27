```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def merge_k_lists(self, lists: list[ListNode]) -> ListNode:
    import heapq
    
    # Python heapq requires a way to break ties if vals are equal.
    # We can use id(node) or an index.
    pq = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(pq, (head.val, i, head))
            
    dummy = ListNode(0)
    curr = dummy
    
    while pq:
        val, i, min_node = heapq.heappop(pq)
        curr.next = min_node
        curr = curr.next
        if min_node.next:
            heapq.heappush(pq, (min_node.next.val, i, min_node.next))
            
    return dummy.next
# Time Complexity: O(N log K)
# Space Complexity: O(K)
```