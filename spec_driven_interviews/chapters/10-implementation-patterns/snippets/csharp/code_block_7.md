```csharp
public int RemoveDuplicates(int[] nums) {
    if (nums == null || nums.Length == 0) return 0;

    int write = 1; // First element is always unique
    for (int read = 1; read < nums.Length; read++) {
        if (nums[read] != nums[write - 1]) {
            nums[write++] = nums[read];
        }
    }

    return write;
}
// Time: O(N), Space: O(1)
```