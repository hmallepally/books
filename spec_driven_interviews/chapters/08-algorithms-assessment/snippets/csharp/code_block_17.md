```csharp
public int SubarraySum(int[] nums, int k) {
    int count = 0, sum = 0;
    var map = new Dictionary<int, int>();
    map[0] = 1;
    foreach (int num in nums) {
        sum += num;
        if (map.ContainsKey(sum - k)) {
            count += map[sum - k];
        }
        if (!map.ContainsKey(sum)) map[sum] = 0;
        map[sum] = map[sum] + 1;
    }
    return count;
}
```