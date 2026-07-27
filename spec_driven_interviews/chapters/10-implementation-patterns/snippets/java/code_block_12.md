```java
public boolean isMonotonic(int[] nums) {
    if (nums == null || nums.length <= 2) return true;

    boolean increasing = true;
    boolean decreasing = true;

    for (int i = 0; i < nums.length - 1; i++) {
        if (nums[i] > nums[i + 1]) increasing = false;
        if (nums[i] < nums[i + 1]) decreasing = false;
    }

    return increasing || decreasing;
}
// Time: O(N), Space: O(1)
```
