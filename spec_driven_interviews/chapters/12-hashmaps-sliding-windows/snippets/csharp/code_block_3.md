```csharp
Dictionary<int, int> map = new Dictionary<int, int>();
map[0] = 1; // Base case for subarrays starting at index 0
int sum = 0, count = 0;
foreach (int num in nums) {
    sum += num;
    if (map.ContainsKey(sum - k)) {
        count += map[sum - k];
    }
    map[sum] = map.GetValueOrDefault(sum, 0) + 1;
}
```