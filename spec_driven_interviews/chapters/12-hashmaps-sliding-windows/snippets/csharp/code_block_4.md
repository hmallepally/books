```csharp
Dictionary<string, List<string>> map = new Dictionary<string, List<string>>();
foreach (string s in strs) {
    int[] count = new int[26];
    foreach (char c in s.ToCharArray()) count[c - 'a']++;
    string key = string.Join(",", count);
    if (!map.ContainsKey(key)) map[key] = new List<string>();
    map[key].Add(s);
}
```