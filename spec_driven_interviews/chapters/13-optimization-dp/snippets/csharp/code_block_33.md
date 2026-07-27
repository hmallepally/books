```csharp
public ListNode MergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode, int> pq = new PriorityQueue<ListNode, int>();
    foreach (ListNode head in lists) {
        if (head != null) pq.Enqueue(head, head.val);
    }
    ListNode dummy = new ListNode(0), curr = dummy;
    while (pq.Count > 0) {
        ListNode minNode = pq.Dequeue();
        curr.next = minNode;
        curr = curr.next;
        if (minNode.next != null) pq.Enqueue(minNode.next, minNode.next.val);
    }
    return dummy.next;
}
// Time Complexity: O(N log K)
// Space Complexity: O(K)
```