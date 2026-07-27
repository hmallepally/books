```csharp
public int Rob(int[] nums)
{
    if (nums == null || nums.Length == 0) return 0;
    int prev2 = 0, prev1 = 0;

    foreach (int num in nums)
    {
        int curr = Math.Max(prev1, prev2 + num); // Skip vs Take
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```
