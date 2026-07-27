```csharp
public int FindMaxLength(int[] nums) {
    Dictionary<int, int> map = new Dictionary<int, int>();
    map[0] = -1;
    int sum = 0, max = 0;
    for (int i = 0; i < nums.Length; i++) {
        sum += nums[i] == 0 ? -1 : 1; // Map 0 to -1
        if (map.ContainsKey(sum)) {
            max = Math.Max(max, i - map[sum]);
        } else {
            map[sum] = i; // Store first occurrence
        }
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```