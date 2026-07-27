```csharp
public int[] TwoSum(int[] nums, int target) {
    Dictionary<int, int> seen = new Dictionary<int, int>();

    for (int i = 0; i < nums.Length; i++) {
        int complement = target - nums[i];
        if (seen.ContainsKey(complement)) {
            return new int[]{seen[complement], i};
        }
        seen[nums[i]] = i;
    }

    return new int[]{}; // Should not reach here per problem guarantee
}
// Time: O(N), Space: O(N)
```