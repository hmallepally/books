```java
public int characterReplacement(String s, int k) {
    int[] count = new int[26];
    int maxCount = 0, left = 0, maxLen = 0;
    for (int right = 0; right < s.length(); right++) {
        maxCount = Math.max(maxCount, ++count[s.charAt(right) - 'A']);
        if (right - left + 1 - maxCount > k) { // Invalid window
            count[s.charAt(left++) - 'A']--;
        }
        maxLen = Math.max(maxLen, right - left + 1);
    }
    return maxLen;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```
