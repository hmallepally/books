```java
public int[] plusOne(int[] digits) {
    for (int i = digits.length - 1; i >= 0; i--) {
        digits[i]++;
        if (digits[i] < 10) {
            return digits; // No further carry needed
        }
        digits[i] = 0; // Carry to next position
    }

    // All digits were 9 — need a new array [1, 0, 0, ..., 0]
    int[] result = new int[digits.length + 1];
    result[0] = 1;
    return result;
}
// Time: O(N), Space: O(1) amortized (O(N) only for all-9s edge case)
```
