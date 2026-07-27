```csharp
public bool CheckSubarraySum(int[] nums, int k) {
    Dictionary<int, int> map = new Dictionary<int, int>();
    map[0] = -1;
    int sum = 0;
    for (int i = 0; i < nums.Length; i++) {
        sum += nums[i];
        int mod = k == 0 ? sum : ((sum % k) + k) % k;
        if (map.ContainsKey(mod)) {
            if (i - map[mod] > 1) return true; // Length >= 2
        } else {
            map[mod] = i;
        }
    }
    return false;
}
// Time Complexity: O(N) | Space Complexity: O(min(N, K))
```