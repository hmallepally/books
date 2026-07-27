```csharp
public int LadderLength(string beginWord, string endWord, IList<string> wordList) {
    HashSet<string> set = new HashSet<string>(wordList);
    if (!set.Contains(endWord)) return 0;
    
    Queue<string> queue = new Queue<string>();
    queue.Enqueue(beginWord);
    int level = 1;
    
    while (queue.Count > 0) {
        int size = queue.Count;
        for (int i = 0; i < size; i++) { // Level-by-level processing
            string curr = queue.Dequeue();
            char[] chars = curr.ToCharArray();
            for (int j = 0; j < chars.Length; j++) {
                char orig = chars[j];
                for (char c = 'a'; c <= 'z'; c++) { // Try all mutations
                    if (c == orig) continue;
                    chars[j] = c;
                    string next = new string(chars);
                    if (next.Equals(endWord)) return level + 1;
                    if (set.Remove(next)) { // remove serves as 'visited' check
                        queue.Enqueue(next);
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