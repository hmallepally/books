```csharp
public int RemoveElement(int[] nums, int val) {
    if (nums == null) return 0;

    int write = 0;
    for (int read = 0; read < nums.Length; read++) {
        if (nums[read] != val) {
            nums[write++] = nums[read];
        }
    }

    return write;
}
// Time: O(N), Space: O(1)
```