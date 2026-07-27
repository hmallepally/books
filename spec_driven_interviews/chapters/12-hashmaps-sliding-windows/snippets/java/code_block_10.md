```java
public String minWindow(String s, String t) {
    int[] map = new int[128];
    for (char c : t.toCharArray()) map[c]++;
    int left = 0, count = t.length(), minLen = Integer.MAX_VALUE, minStart = 0;
    for (int right = 0; right < s.length(); right++) {
        if (map[s.charAt(right)]-- > 0) count--; // Found required char
        while (count == 0) { // All chars found
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                minStart = left;
            }
            if (++map[s.charAt(left++)] > 0) count++; // Removed required char
        }
    }
    return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
}
// Time Complexity: O(N) | Space Complexity: O(1)
```
