```java
public boolean checkSubarraySum(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, -1);
    int sum = 0;
    for (int i = 0; i < nums.length; i++) {
        sum += nums[i];
        int mod = k == 0 ? sum : ((sum % k) + k) % k;
        if (map.containsKey(mod)) {
            if (i - map.get(mod) > 1) return true; // Length >= 2
        } else {
            map.put(mod, i);
        }
    }
    return false;
}
// Time Complexity: O(N) | Space Complexity: O(min(N, K))
```
