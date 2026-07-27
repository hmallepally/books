```java
public int binarySearch(int[] nums, int target) {
    // 1. Enforce Pre-conditions
    if (nums == null || nums.length == 0) {
        return -1;
    }

    int left = 0;
    int right = nums.length - 1;

    // Maintain Invariant: target is in nums[left...right]
    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (nums[mid] == target) {
            return mid; // Post-condition satisfied
        } else if (nums[mid] < target) {
            left = mid + 1; // Invariant maintained
        } else {
            right = mid - 1; // Invariant maintained
        }
    }

    return -1; // Search range is empty -> target not in nums
}
```