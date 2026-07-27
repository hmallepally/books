```java
package com.aurapay.algorithms;

/**
 * Reverses a singly linked list in-place.
 * Time Complexity: O(N) where N is the number of nodes.
 * Space Complexity: O(1) auxiliary space.
 */
public class LinkedListReversal {

    public static class ListNode {
        int val;
        ListNode next;
        ListNode(int x) {
            val = x;
            next = null;
        }
    }

    /**
     * Reverses the linked list and returns the new head node.
     */
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;

        while (curr != null) {
            ListNode nextTemp = curr.next; // 1. Save the next node
            curr.next = prev;              // 2. Reverse the current node's pointer
            prev = curr;                   // 3. Move prev one step forward
            curr = nextTemp;               // 4. Move curr one step forward
        }

        return prev; // New head node
    }
}
```
