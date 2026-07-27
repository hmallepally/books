```csharp
public string[] TransformWords(string[] words) {
    if (words == null) return new string[0];
    string[] result = new string[words.Length];

    for (int i = 0; i < words.Length; i++) {
        if (words[i].Length % 2 != 0) {
            result[i] = words[i].ToUpper();
        } else {
            char[] arr = words[i].ToCharArray();
            Array.Reverse(arr);
            result[i] = new string(arr);
        }
    }

    return result;
}
// Time: O(N * K) where K is average word length, Space: O(N * K) for output
```