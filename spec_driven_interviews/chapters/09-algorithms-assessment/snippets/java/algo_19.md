```java
public int rob(int[] nums) {
    if (nums == null || nums.length == 0) return 0;
    int prev2 = 0, prev1 = 0;

    for (int num : nums) {
        int curr = Math.max(prev1, prev2 + num); // Skip vs Take
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```
