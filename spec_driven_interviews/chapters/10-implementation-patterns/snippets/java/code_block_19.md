```java
public boolean isAlternatingParity(int[] nums) {
    if (nums == null || nums.length <= 1) return true;

    for (int i = 0; i < nums.length - 1; i++) {
        // Use Math.abs for safety with negative numbers
        if (Math.abs(nums[i] % 2) == Math.abs(nums[i + 1] % 2)) {
            return false;
        }
    }

    return true;
}
// Time: O(N), Space: O(1)
```
