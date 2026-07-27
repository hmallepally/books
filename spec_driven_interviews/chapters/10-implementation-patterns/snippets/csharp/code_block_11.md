```csharp
public int PivotIndex(int[] nums) {
    if (nums == null) return -1;

    int totalSum = 0;
    foreach (int num in nums) totalSum += num;

    int leftSum = 0;
    for (int i = 0; i < nums.Length; i++) {
        // rightSum = totalSum - leftSum - nums[i]
        if (leftSum == totalSum - leftSum - nums[i]) return i;
        leftSum += nums[i];
    }

    return -1;
}
// Time: O(N), Space: O(1)
```