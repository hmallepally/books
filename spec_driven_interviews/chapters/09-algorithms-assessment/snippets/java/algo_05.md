```java
public int[] maxSlidingWindow(int[] nums, int k) {
    var deque = new ArrayDeque<Integer>();
    var res = new int[nums.length - k + 1];
    int idx = 0;

    for (int i = 0; i < nums.length; i++) {
        while (!deque.isEmpty() && deque.peekFirst() < i - k + 1) deque.pollFirst(); // Expire
        while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) deque.pollLast(); // Kill weaker
        deque.offerLast(i);
        if (i >= k - 1) res[idx++] = nums[deque.peekFirst()];
    }
    return res;
}
```
