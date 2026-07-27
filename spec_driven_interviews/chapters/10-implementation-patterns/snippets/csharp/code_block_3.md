```csharp
public int FirstUniqChar(string s) {
    if (string.IsNullOrEmpty(s)) return -1;

    // Pass 1: Count frequency of each character
    int[] counts = new int[256];
    foreach (char c in s) {
        counts[c]++;
    }

    // Pass 2: Find first character with frequency exactly 1
    for (int i = 0; i < s.Length; i++) {
        if (counts[s[i]] == 1) return i;
    }

    return -1; // All characters repeat
}
// Time: O(N), Space: O(1) — the int[256] is constant size
```