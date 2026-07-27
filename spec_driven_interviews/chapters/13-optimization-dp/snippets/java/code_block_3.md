```java
public int dpStateCompression(int[] nums) {
    if (nums.length == 0) return 0;
    int prev2 = 0; // dp[i-2]
    int prev1 = nums[0]; // dp[i-1]
    for (int i = 1; i < nums.length; i++) {
        int curr = Math.max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```
