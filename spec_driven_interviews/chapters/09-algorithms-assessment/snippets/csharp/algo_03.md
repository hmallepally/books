```csharp
public int SubarraySumEqualsK(int[] nums, int k)
{
    var prefCounts = new Dictionary<int, int>();
    prefCounts[0] = 1;
    int currentSum = 0, count = 0;

    foreach (int num in nums)
    {
        currentSum += num;
        if (prefCounts.TryGetValue(currentSum - k, out int val))
        {
            count += val;
        }
        prefCounts[currentSum] = prefCounts.GetValueOrDefault(currentSum, 0) + 1;
    }
    return count;
}
```
