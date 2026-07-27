```csharp
namespace AuraPay.Algorithms;

/// <summary>
/// Reverses a singly linked list in-place.
/// Time Complexity: O(N) where N is the number of nodes.
/// Space Complexity: O(1) auxiliary space.
/// </summary>
public class LinkedListReversal
{
    public class ListNode
    {
        public int val;
        public ListNode next;
        public ListNode(int x)
        {
            val = x;
            next = null;
        }
    }

    /// <summary>
    /// Reverses the linked list and returns the new head node.
    /// </summary>
    public ListNode ReverseList(ListNode head)
    {
        ListNode prev = null;
        ListNode curr = head;

        while (curr != null)
        {
            ListNode nextTemp = curr.next; // 1. Save the next node
            curr.next = prev;              // 2. Reverse current pointer
            prev = curr;                   // 3. Move prev forward
            curr = nextTemp;               // 4. Move curr forward
        }

        return prev; // New head node
    }
}
```
