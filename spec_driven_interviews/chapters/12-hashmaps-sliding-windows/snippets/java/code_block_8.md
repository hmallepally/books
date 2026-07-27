```java
public List<Integer> findAnagrams(String s, String p) {
    List<Integer> res = new ArrayList<>();
    if (s.length() < p.length()) return res;
    int[] pCount = new int[26], sCount = new int[26];
    for (char c : p.toCharArray()) pCount[c - 'a']++;
    for (int i = 0; i < s.length(); i++) {
        sCount[s.charAt(i) - 'a']++;
        if (i >= p.length()) sCount[s.charAt(i - p.length()) - 'a']--; // Contract
        if (Arrays.equals(pCount, sCount)) res.add(i - p.length() + 1); // Match
    }
    return res;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```
