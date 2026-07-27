```csharp
namespace AuraPay.Algorithms;

/// <summary>
/// Implements the Fast/Slow Pointer (Tortoise and Hare) pattern to detect cycles
/// in a singly linked list.
/// Time Complexity: O(N) where N is the number of nodes.
/// Space Complexity: O(1) auxiliary space.
/// </summary>
public class CycleDetector
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
    /// Detects if a linked list contains a cycle.
    /// Moves slow pointer by 1 step, fast pointer by 2 steps.
    /// </summary>
    public bool HasCycle(ListNode head)
    {
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null)
        {
            slow = slow.next;          // Tortoise: 1 step
            fast = fast.next.next;     // Hare: 2 steps

            if (slow == fast)
            {
                return true; // Fast pointer caught up to slow pointer -> cycle!
            }
        }

        return false; // Fast pointer reached the end -> no cycle
    }
}
```
