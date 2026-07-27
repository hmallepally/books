```csharp
public int MaxSumSubarray(int[] nums, int k) {
    if (nums == null || nums.Length < k || k <= 0) return 0;

    // Initialize sum of first window
    int windowSum = 0;
    for (int i = 0; i < k; i++) windowSum += nums[i];

    int maxSum = windowSum;

    // Slide the window: add right element, remove left element
    for (int i = k; i < nums.Length; i++) {
        windowSum += nums[i] - nums[i - k];
        maxSum = Math.Max(maxSum, windowSum);
    }

    return maxSum;
}
// Time: O(N), Space: O(1)
```