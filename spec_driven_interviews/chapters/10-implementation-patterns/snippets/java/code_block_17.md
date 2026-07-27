```java
public boolean areOccurrencesEqual(String s) {
    if (s == null || s.isEmpty()) return true;

    int[] counts = new int[128];
    for (char c : s.toCharArray()) counts[(int) c]++;

    int expected = 0;
    for (int count : counts) {
        if (count > 0) {
            if (expected == 0) expected = count;
            else if (count != expected) return false;
        }
    }

    return true;
}
// Time: O(N), Space: O(1)
```
