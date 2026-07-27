```java
Map<Integer, Integer> map = new HashMap<>();
map.put(0, 1); // Base case for subarrays starting at index 0
int sum = 0, count = 0;
for (int num : nums) {
    sum += num;
    if (map.containsKey(sum - k)) {
        count += map.get(sum - k);
    }
    map.put(sum, map.getOrDefault(sum, 0) + 1);
}
```
