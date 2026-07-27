```java
public void moveZeroes(int[] nums) {
    if (nums == null || nums.length == 0) return;

    // Pass 1: Copy all non-zero elements to the front
    int write = 0;
    for (int read = 0; read < nums.length; read++) {
        if (nums[read] != 0) {
            nums[write++] = nums[read];
        }
    }

    // Pass 2: Fill remaining positions with zeros
    while (write < nums.length) {
        nums[write++] = 0;
    }
}
// Time: O(N), Space: O(1)
```
