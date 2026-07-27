```java
public int maximumUniqueSubarray(int[] nums) {
    Set<Integer> set = new HashSet<>();
    int sum = 0, max = 0, left = 0;
    for (int right = 0; right < nums.length; right++) {
        while (set.contains(nums[right])) {
            set.remove(nums[left]);
            sum -= nums[left++]; // Remove duplicate
        }
        set.add(nums[right]);
        sum += nums[right];
        max = Math.max(max, sum);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```
