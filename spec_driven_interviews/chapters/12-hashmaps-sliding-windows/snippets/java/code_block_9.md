```java
public int lengthOfLongestSubstringKDistinct(String s, int k) {
    Map<Character, Integer> map = new HashMap<>();
    int left = 0, max = 0;
    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        map.put(c, map.getOrDefault(c, 0) + 1);
        while (map.size() > k) { // Invariant broken
            char leftChar = s.charAt(left++);
            map.put(leftChar, map.get(leftChar) - 1);
            if (map.get(leftChar) == 0) map.remove(leftChar);
        }
        max = Math.max(max, right - left + 1);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(K)
```
