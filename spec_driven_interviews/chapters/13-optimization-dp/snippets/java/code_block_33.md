```java
public ListNode mergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode> pq = new PriorityQueue<>((a,b) -> a.val - b.val);
    for (ListNode head : lists) {
        if (head != null) pq.offer(head);
    }
    ListNode dummy = new ListNode(0), curr = dummy;
    while (!pq.isEmpty()) {
        ListNode minNode = pq.poll();
        curr.next = minNode;
        curr = curr.next;
        if (minNode.next != null) pq.offer(minNode.next);
    }
    return dummy.next;
}
// Time Complexity: O(N log K)
// Space Complexity: O(K)
```
