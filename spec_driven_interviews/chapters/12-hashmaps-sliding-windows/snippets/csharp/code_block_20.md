```csharp
public IList<int> FindDuplicates(int[] nums) {
    List<int> res = new List<int>();
    foreach (int num in nums) {
        int idx = Math.Abs(num) - 1;
        if (nums[idx] < 0) res.Add(Math.Abs(num)); // Found duplicate
        else nums[idx] = -nums[idx]; // Mark seen
    }
    return res;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```