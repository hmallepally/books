```csharp
public int LongestCommonSubsequence(string text1, string text2) {
    if (text1.Length < text2.Length) return LongestCommonSubsequence(text2, text1);
    int m = text1.Length, n = text2.Length;
    var prev = new int[n + 1];
    var curr = new int[n + 1];
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            curr[j] = text1[i - 1] == text2[j - 1]
                ? prev[j - 1] + 1
                : Math.Max(prev[j], curr[j - 1]);
        }
        var temp = prev; prev = curr; curr = temp;
        Array.Fill(curr, 0);
    }
    return prev[n];
}
// Time Complexity: O(M * N)
// Space Complexity: O(min(M, N)) - Space compressed DP as taught in the vocabulary section.
```