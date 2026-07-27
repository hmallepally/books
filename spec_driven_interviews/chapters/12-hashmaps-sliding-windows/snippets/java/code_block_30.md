```java
public int subarraysWithKDistinct(int[] nums, int k) {
    return atMostK(nums, k) - atMostK(nums, k - 1);
}
private int atMostK(int[] nums, int k) {
    int[] count = new int[nums.length + 1];
    int left = 0, res = 0, distinct = 0;
    for (int right = 0; right < nums.length; right++) {
        if (count[nums[right]]++ == 0) distinct++;
        while (distinct > k) {
            if (--count[nums[left++]] == 0) distinct--;
        }
        res += right - left + 1;
    }
    return res;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```
