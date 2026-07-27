```csharp
public IList<IList<int>> FourSum(int[] nums, int target) {
    Array.Sort(nums);
    IList<IList<int>> res = new List<IList<int>>();
    for (int i = 0; i < nums.Length - 3; i++) {
        if (i > 0 && nums[i] == nums[i-1]) continue;
        for (int j = i + 1; j < nums.Length - 2; j++) {
            if (j > i + 1 && nums[j] == nums[j-1]) continue;
            int L = j + 1, R = nums.Length - 1;
            while (L < R) {
                long sum = (long)nums[i] + nums[j] + nums[L] + nums[R];
                if (sum == target) {
                    res.Add(new List<int>{nums[i], nums[j], nums[L], nums[R]});
                    while (L < R && nums[L] == nums[L+1]) L++;
                    while (L < R && nums[R] == nums[R-1]) R--;
                    L++; R--;
                }
                else if (sum < target) L++;
                else R--;
            }
        }
    }
    return res;
}
// Time Complexity: O(N^3) | Space Complexity: O(1)
```