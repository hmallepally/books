```csharp
public IList<IList<string>> GroupAnagrams(string[] strs) {
    if (strs == null || strs.Length == 0) return new List<IList<string>>();
    var map = new Dictionary<string, List<string>>();
    foreach (string s in strs) {
        char[] ca = s.ToCharArray();
        Array.Sort(ca);
        string key = new string(ca);
        if (!map.ContainsKey(key)) {
            map[key] = new List<string>();
        }
        map[key].Add(s);
    }
    return new List<IList<string>>(map.Values);
}
```