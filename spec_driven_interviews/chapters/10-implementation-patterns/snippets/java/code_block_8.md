```java
public int singleNumber(int[] nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num; // Pairs cancel, unique value survives
    }
    return result;
}
// Time: O(N), Space: O(1)
```
