```csharp
public bool CheckInclusion(string s1, string s2) {
    if (s1.Length > s2.Length) return false;
    int[] s1map = new int[26], s2map = new int[26];
    foreach (char c in s1) s1map[c - 'a']++;
    for (int i = 0; i < s2.Length; i++) {
        s2map[s2[i] - 'a']++;
        if (i >= s1.Length) s2map[s2[i - s1.Length] - 'a']--;
        if (s1map.SequenceEqual(s2map)) return true;
    }
    return false;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```