```csharp
public int[] TwoSumSorted(int[] nums, int target)
{
    int left = 0, right = nums.Length - 1;
    while (left < right)
    {
        int sum = nums[left] + nums[right];
        if (sum == target) return new int[] { left, right };
        else if (sum < target) left++;
        else right--;
    }
    return new int[0];
}
```
