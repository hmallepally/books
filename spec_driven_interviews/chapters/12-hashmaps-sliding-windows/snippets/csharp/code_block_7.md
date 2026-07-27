```csharp
public IList<IList<string>> GroupAnagrams(string[] strs) {
    Dictionary<string, List<string>> map = new Dictionary<string, List<string>>();
    foreach (string s in strs) {
        int[] count = new int[26];
        foreach (char c in s) count[c - 'a']++; // Build signature
        string key = string.Join(",", count);
        if (!map.ContainsKey(key)) map[key] = new List<string>();
        map[key].Add(s);
    }
    return new List<IList<string>>(map.Values);
}
// Time Complexity: O(N * L) | Space Complexity: O(N * L)
```