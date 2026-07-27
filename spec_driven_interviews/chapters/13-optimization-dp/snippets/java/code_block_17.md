```java
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    Set<String> set = new HashSet<>(wordList);
    if (!set.contains(endWord)) return 0;
    
    Queue<String> queue = new ArrayDeque<>();
    queue.offer(beginWord);
    int level = 1;
    
    while (!queue.isEmpty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) { // Level-by-level processing
            String curr = queue.poll();
            char[] chars = curr.toCharArray();
            for (int j = 0; j < chars.length; j++) {
                char orig = chars[j];
                for (char c = 'a'; c <= 'z'; c++) { // Try all mutations
                    if (c == orig) continue;
                    chars[j] = c;
                    String next = new String(chars);
                    if (next.equals(endWord)) return level + 1;
                    if (set.remove(next)) { // remove serves as 'visited' check
                        queue.offer(next);
                    }
                }
                chars[j] = orig; // Backtrack
            }
        }
        level++;
    }
    return 0;
}
// Time Complexity: O(M^2 * N) where M is word length, N is number of words
// Space Complexity: O(M * N)
```
