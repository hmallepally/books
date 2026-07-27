```java
public List<Integer> findSubstring(String s, String[] words) {
    List<Integer> res = new ArrayList<>();
    if (s.isEmpty() || words.length == 0) return res;
    int wordLen = words[0].length(), totalLen = wordLen * words.length;
    Map<String, Integer> counts = new HashMap<>();
    for (String w : words) counts.put(w, counts.getOrDefault(w, 0) + 1);
    
    for (int i = 0; i <= s.length() - totalLen; i++) {
        Map<String, Integer> seen = new HashMap<>();
        int j = 0;
        while (j < words.length) {
            String w = s.substring(i + j * wordLen, i + (j + 1) * wordLen);
            if (counts.containsKey(w)) {
                seen.put(w, seen.getOrDefault(w, 0) + 1);
                if (seen.get(w) > counts.get(w)) break;
            } else break;
            j++;
        }
        if (j == words.length) res.add(i);
    }
    return res;
}
// Time Complexity: O(N * M * L) | Space Complexity: O(M)
```
