```csharp
public IList<int> FindSubstring(string s, string[] words) {
    List<int> res = new List<int>();
    if (s.Length == 0 || words.Length == 0) return res;
    int wordLen = words[0].Length, totalLen = wordLen * words.Length;
    Dictionary<string, int> counts = new Dictionary<string, int>();
    foreach (string w in words) counts[w] = counts.GetValueOrDefault(w, 0) + 1;
    
    for (int i = 0; i <= s.Length - totalLen; i++) {
        Dictionary<string, int> seen = new Dictionary<string, int>();
        int j = 0;
        while (j < words.Length) {
            string w = s.Substring(i + j * wordLen, wordLen);
            if (counts.ContainsKey(w)) {
                seen[w] = seen.GetValueOrDefault(w, 0) + 1;
                if (seen[w] > counts[w]) break;
            } else break;
            j++;
        }
        if (j == words.Length) res.Add(i);
    }
    return res;
}
// Time Complexity: O(N * M * L) | Space Complexity: O(M)
```