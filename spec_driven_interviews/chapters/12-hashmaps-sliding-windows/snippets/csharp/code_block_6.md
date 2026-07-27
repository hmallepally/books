```csharp
public int SubarraySum(int[] nums, int k) {
    Dictionary<int, int> map = new Dictionary<int, int>();
    map[0] = 1; // Base case
    int sum = 0, count = 0;
    foreach (int num in nums) {
        sum += num;
        // Check if required prefix exists
        if (map.ContainsKey(sum - k)) count += map[sum - k];
        map[sum] = map.GetValueOrDefault(sum, 0) + 1;
    }
    return count;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```