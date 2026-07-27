```csharp
public int NumSubarrayProductLessThanK(int[] nums, int k) {
    if (k <= 1) return 0;
    int prod = 1, left = 0, count = 0;
    for (int right = 0; right < nums.Length; right++) {
        prod *= nums[right];
        while (prod >= k) prod /= nums[left++]; // Shrink
        count += right - left + 1; // Add valid subarrays
    }
    return count;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```