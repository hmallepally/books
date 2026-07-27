```csharp
public void MoveZeroes(int[] nums) {
    if (nums == null || nums.Length == 0) return;

    // Pass 1: Copy all non-zero elements to the front
    int write = 0;
    for (int read = 0; read < nums.Length; read++) {
        if (nums[read] != 0) {
            nums[write++] = nums[read];
        }
    }

    // Pass 2: Fill remaining positions with zeros
    while (write < nums.Length) {
        nums[write++] = 0;
    }
}
// Time: O(N), Space: O(1)
```