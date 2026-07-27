```java
package com.aurapay.algorithms;

/**
 * Sorts an array containing numbers from 1 to N in-place.
 * Time Complexity: O(N) where N is the size of the array.
 * Space Complexity: O(1) auxiliary space.
 */
public class CyclicSort {

    /**
     * Sorts the array using the Cyclic Sort pattern.
     */
    public void sort(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            int correctIndex = nums[i] - 1; // In a 1-to-N range, value X belongs at index X-1
            if (nums[i] != nums[correctIndex]) {
                swap(nums, i, correctIndex); // Swap to its correct position
            } else {
                i++; // Only increment when the current element is in its correct place
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```
