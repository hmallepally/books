```java
package com.aurapay.algorithms;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Implements the Sliding Window Maximum algorithm using a Monotonic Deque.
 * Time Complexity: O(N) where N is the size of the array.
 * Space Complexity: O(K) where K is the size of the sliding window.
 */
public class SlidingWindowSolver {

    /**
     * Finds the maximum value in each sliding window of size K as it moves
     * from left to right across the array.
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k <= 0) {
            return new int[0];
        }

        int n = nums.length;
        int[] result = new int[n - k + 1];
        int ri = 0; // Result array index

        // Deque stores array indices. 
        // Invariant: The elements corresponding to indices in the deque are stored 
        // in strictly decreasing order. Thus, the index of the maximum element 
        // in the current window is always at the head of the deque.
        Deque<Integer> q = new ArrayDeque<>();

        for (int i = 0; i < n; i++) {
            // 1. Remove indices that are out of the current window boundary
            if (!q.isEmpty() && q.peek() < i - k + 1) {
                q.poll();
            }

            // 2. Maintain monotonic invariant: Remove indices of elements smaller
            // than the current element from the tail of the deque
            while (!q.isEmpty() && nums[q.peekLast()] < nums[i]) {
                q.pollLast();
            }

            // 3. Add current element's index to the tail
            q.offer(i);

            // 4. If window size has reached K, store the maximum in the result
            if (i >= k - 1) {
                result[ri++] = nums[q.peek()];
            }
        }

        return result;
    }
}
```
