```java
public boolean almostIncreasingSequence(int[] sequence) {
    int count = 0;   // Number of violations
    int badIdx = -1;  // Index of first violation

    for (int i = 0; i < sequence.length - 1; i++) {
        if (sequence[i] >= sequence[i + 1]) {
            count++;
            badIdx = i;
            if (count > 1) return false; // More than one violation
        }
    }

    if (count == 0) return true; // Already strictly increasing

    // Try removing element at badIdx
    if (badIdx == 0 || sequence[badIdx - 1] < sequence[badIdx + 1]) {
        return true;
    }

    // Try removing element at badIdx + 1
    if (badIdx + 2 >= sequence.length || sequence[badIdx] < sequence[badIdx + 2]) {
        return true;
    }

    return false;
}
// Time: O(N), Space: O(1)
```
