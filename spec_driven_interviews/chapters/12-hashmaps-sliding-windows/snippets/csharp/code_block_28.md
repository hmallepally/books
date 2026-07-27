```csharp
public int NumberOfSubarrays(int[] nums, int k) {
    Dictionary<int, int> map = new Dictionary<int, int>();
    map[0] = 1;
    int sum = 0, count = 0;
    foreach (int num in nums) {
        sum += num % 2;
        count += map.GetValueOrDefault(sum - k, 0);
        map[sum] = map.GetValueOrDefault(sum, 0) + 1;
    }
    return count;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```