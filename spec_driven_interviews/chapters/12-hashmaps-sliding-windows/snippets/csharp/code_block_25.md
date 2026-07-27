```csharp
public int MinSubArrayLen(int target, int[] nums) {
    int left = 0, sum = 0, min = int.MaxValue;
    for (int right = 0; right < nums.Length; right++) {
        sum += nums[right];
        while (sum >= target) {
            min = Math.Min(min, right - left + 1);
            sum -= nums[left++];
        }
    }
    return min == int.MaxValue ? 0 : min;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```