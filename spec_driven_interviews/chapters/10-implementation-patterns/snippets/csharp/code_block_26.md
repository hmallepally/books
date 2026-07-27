```csharp
public int CommonCharacterCount(string s1, string s2) {
    int[] count1 = new int[26];
    int[] count2 = new int[26];

    foreach (char c in s1) count1[c - 'a']++;
    foreach (char c in s2) count2[c - 'a']++;

    int common = 0;
    for (int i = 0; i < 26; i++) {
        common += Math.Min(count1[i], count2[i]);
    }

    return common;
}
// Time: O(N + M), Space: O(1) — fixed 26-element arrays
```