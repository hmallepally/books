```csharp
public int CharacterReplacement(string s, int k) {
    int[] count = new int[26];
    int maxCount = 0, left = 0, maxLen = 0;
    for (int right = 0; right < s.Length; right++) {
        maxCount = Math.Max(maxCount, ++count[s[right] - 'A']);
        if (right - left + 1 - maxCount > k) { // Invalid window
            count[s[left++] - 'A']--;
        }
        maxLen = Math.Max(maxLen, right - left + 1);
    }
    return maxLen;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```