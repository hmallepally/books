```java
public int longestSubarray(int[] nums, int k) {
    int left = 0, result = 0, zeroCount = 0;

    for (int right = 0; right < nums.length; right++) {
        if (nums[right] == 0) zeroCount++;

        while (zeroCount > k) {
            if (nums[left] == 0) zeroCount--;
            left++; // Always advance left during shrink
        }

        result = Math.max(result, right - left + 1);
    }
    return result;
}
```
