```java
public int subarraySumEqualsK(int[] nums, int k) {
    var prefCounts = new HashMap<Integer, Integer>();
    prefCounts.put(0, 1);
    int currentSum = 0, count = 0;

    for (int num : nums) {
        currentSum += num;
        if (prefCounts.containsKey(currentSum - k)) {
            count += prefCounts.get(currentSum - k);
        }
        prefCounts.put(currentSum, prefCounts.getOrDefault(currentSum, 0) + 1);
    }
    return count;
}
```
