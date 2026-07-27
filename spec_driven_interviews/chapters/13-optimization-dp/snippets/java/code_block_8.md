```java
public int longestCommonSubsequence(String text1, String text2) {
    if (text1.length() < text2.length()) return longestCommonSubsequence(text2, text1);
    int m = text1.length(), n = text2.length();
    var prev = new int[n + 1];
    var curr = new int[n + 1];
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            curr[j] = text1.charAt(i - 1) == text2.charAt(j - 1)
                ? prev[j - 1] + 1
                : Math.max(prev[j], curr[j - 1]);
        }
        var temp = prev; prev = curr; curr = temp;
        java.util.Arrays.fill(curr, 0);
    }
    return prev[n];
}
// Time Complexity: O(M * N)
// Space Complexity: O(min(M, N)) - Space compressed DP as taught in the vocabulary section.
```
