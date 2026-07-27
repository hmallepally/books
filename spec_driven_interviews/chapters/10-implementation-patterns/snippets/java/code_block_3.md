```java
public int firstUniqChar(String s) {
    if (s == null || s.isEmpty()) return -1;

    // Pass 1: Count frequency of each character
    int[] counts = new int[256];
    for (int i = 0; i < s.length(); i++) {
        counts[s.charAt(i)]++;
    }

    // Pass 2: Find first character with frequency exactly 1
    for (int i = 0; i < s.length(); i++) {
        if (counts[s.charAt(i)] == 1) return i;
    }

    return -1; // All characters repeat
}
// Time: O(N), Space: O(1) — the int[256] is constant size
```
