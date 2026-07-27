```csharp
public string MinWindow(string s, string t) {
    int[] map = new int[128];
    foreach (char c in t) map[c]++;
    int left = 0, count = t.Length, minLen = int.MaxValue, minStart = 0;
    for (int right = 0; right < s.Length; right++) {
        if (map[s[right]]-- > 0) count--; // Found required char
        while (count == 0) { // All chars found
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                minStart = left;
            }
            if (++map[s[left++]] > 0) count++; // Removed required char
        }
    }
    return minLen == int.MaxValue ? "" : s.Substring(minStart, minLen);
}
// Time Complexity: O(N) | Space Complexity: O(1)
```