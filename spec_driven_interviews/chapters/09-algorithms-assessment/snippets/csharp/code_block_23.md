```csharp
public int SingleNumber(int[] nums) {
    int result = 0;
    foreach (int num in nums) {
        result ^= num;  // Duplicates cancel: a ^ a = 0, 0 ^ b = b
    }
    return result;
}
```
