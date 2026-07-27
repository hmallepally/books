```java
public int maxFrequency(int[] nums, int k) {
    Arrays.sort(nums);
    int left = 0;
    long sum = 0;
    for (int right = 0; right < nums.length; right++) {
        sum += nums[right];
        if ((long)nums[right] * (right - left + 1) - sum > k) {
            sum -= nums[left++];
        }
    }
    return nums.length - left;
}
// Time Complexity: O(N log N) | Space Complexity: O(1)
```
