```csharp
public IList<int> FindAnagrams(string s, string p) {
    List<int> res = new List<int>();
    if (s.Length < p.Length) return res;
    int[] pCount = new int[26], sCount = new int[26];
    foreach (char c in p) pCount[c - 'a']++;
    for (int i = 0; i < s.Length; i++) {
        sCount[s[i] - 'a']++;
        if (i >= p.Length) sCount[s[i - p.Length] - 'a']--; // Contract
        if (pCount.SequenceEqual(sCount)) res.Add(i - p.Length + 1); // Match
    }
    return res;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```