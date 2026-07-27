```csharp
public int MaximumUniqueSubarray(int[] nums) {
    HashSet<int> set = new HashSet<int>();
    int sum = 0, max = 0, left = 0;
    for (int right = 0; right < nums.Length; right++) {
        while (set.Contains(nums[right])) {
            set.Remove(nums[left]);
            sum -= nums[left++]; // Remove duplicate
        }
        set.Add(nums[right]);
        sum += nums[right];
        max = Math.Max(max, sum);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```