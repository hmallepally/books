```java
public int removeDuplicates(int[] nums) {
    if (nums == null || nums.length == 0) return 0;

    int write = 1; // First element is always unique
    for (int read = 1; read < nums.length; read++) {
        if (nums[read] != nums[write - 1]) {
            nums[write++] = nums[read];
        }
    }

    return write;
}
// Time: O(N), Space: O(1)
```
