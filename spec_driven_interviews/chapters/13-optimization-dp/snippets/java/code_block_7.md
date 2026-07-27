```java
public int[] maxSlidingWindow(int[] nums, int k) {
    if (nums == null || k <= 0) return new int[0];
    int n = nums.length;
    int[] res = new int[n - k + 1];
    int resIndex = 0;
    Deque<Integer> q = new ArrayDeque<>();
    
    for (int i = 0; i < n; i++) {
        // Remove indices outside the current window
        if (!q.isEmpty() && q.peekFirst() < i - k + 1) {
            q.pollFirst();
        }
        // Remove smaller elements (maintain decreasing order)
        while (!q.isEmpty() && nums[q.peekLast()] < nums[i]) {
            q.pollLast();
        }
        q.offerLast(i);
        
        // Record max for the window
        if (i >= k - 1) {
            res[resIndex++] = nums[q.peekFirst()];
        }
    }
    return res;
}
// Time Complexity: O(N) since each element is pushed/popped at most once
// Space Complexity: O(K) for the deque
```
