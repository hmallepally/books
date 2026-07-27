```csharp
public bool IsAlternatingParity(int[] nums) {
    if (nums == null || nums.Length <= 1) return true;

    for (int i = 0; i < nums.Length - 1; i++) {
        // Use Math.Abs for safety with negative numbers
        if (Math.Abs(nums[i] % 2) == Math.Abs(nums[i + 1] % 2)) {
            return false;
        }
    }

    return true;
}
// Time: O(N), Space: O(1)
```