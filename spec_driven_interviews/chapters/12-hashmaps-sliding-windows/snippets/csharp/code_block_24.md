```csharp
public int FirstMissingPositive(int[] nums) {
    int i = 0;
    while (i < nums.Length) {
        // Swap to correct position if valid
        if (nums[i] > 0 && nums[i] <= nums.Length && nums[nums[i] - 1] != nums[i]) {
            int temp = nums[nums[i] - 1];
            nums[nums[i] - 1] = nums[i];
            nums[i] = temp;
        } else {
            i++;
        }
    }
    for (i = 0; i < nums.Length; i++) {
        if (nums[i] != i + 1) return i + 1; // Missing
    }
    return nums.Length + 1;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```