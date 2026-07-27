```csharp
public int LengthOfLongestSubstring(string s) {
    HashSet<char> set = new HashSet<char>();
    int left = 0, max = 0;
    for (int right = 0; right < s.Length; right++) {
        // Contract if duplicate found
        while (set.Contains(s[right])) {
            set.Remove(s[left++]);
        }
        set.Add(s[right]); // Add current char
        max = Math.Max(max, right - left + 1);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(min(N, M))
```