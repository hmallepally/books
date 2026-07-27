```java
public int singleNumber(int[] nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;  // Duplicates cancel: a ^ a = 0, 0 ^ b = b
    }
    return result;
}
```
