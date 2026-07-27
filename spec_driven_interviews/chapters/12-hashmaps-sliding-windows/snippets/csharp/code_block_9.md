```csharp
public int LengthOfLongestSubstringKDistinct(string s, int k) {
    Dictionary<char, int> map = new Dictionary<char, int>();
    int left = 0, max = 0;
    for (int right = 0; right < s.Length; right++) {
        char c = s[right];
        map[c] = map.GetValueOrDefault(c, 0) + 1;
        while (map.Count > k) { // Invariant broken
            char leftChar = s[left++];
            map[leftChar]--;
            if (map[leftChar] == 0) map.Remove(leftChar);
        }
        max = Math.Max(max, right - left + 1);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(K)
```