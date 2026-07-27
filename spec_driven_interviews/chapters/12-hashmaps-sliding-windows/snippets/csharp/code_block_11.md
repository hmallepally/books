```csharp
public IList<IList<string>> GroupStrings(string[] strings) {
    Dictionary<string, List<string>> map = new Dictionary<string, List<string>>();
    foreach (string s in strings) {
        StringBuilder key = new StringBuilder();
        for (int i = 1; i < s.Length; i++) {
            int diff = (s[i] - s[i-1] + 26) % 26; // Circular difference
            key.Append(diff).Append(",");
        }
        string k = key.ToString();
        if (!map.ContainsKey(k)) map[k] = new List<string>();
        map[k].Add(s);
    }
    return new List<IList<string>>(map.Values);
}
// Time Complexity: O(N * L) | Space Complexity: O(N * L)
```