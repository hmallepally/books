```csharp
public int MaxFrequency(int[] nums, int k) {
    Array.Sort(nums);
    int left = 0;
    long sum = 0;
    for (int right = 0; right < nums.Length; right++) {
        sum += nums[right];
        if ((long)nums[right] * (right - left + 1) - sum > k) {
            sum -= nums[left++];
        }
    }
    return nums.Length - left;
}
// Time Complexity: O(N log N) | Space Complexity: O(1)
```