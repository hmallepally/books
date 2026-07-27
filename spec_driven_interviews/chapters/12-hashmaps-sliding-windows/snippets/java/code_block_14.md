```java
public boolean checkInclusion(String s1, String s2) {
    if (s1.length() > s2.length()) return false;
    int[] s1map = new int[26], s2map = new int[26];
    for (char c : s1.toCharArray()) s1map[c - 'a']++;
    for (int i = 0; i < s2.length(); i++) {
        s2map[s2.charAt(i) - 'a']++;
        if (i >= s1.length()) s2map[s2.charAt(i - s1.length()) - 'a']--;
        if (Arrays.equals(s1map, s2map)) return true;
    }
    return false;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```
