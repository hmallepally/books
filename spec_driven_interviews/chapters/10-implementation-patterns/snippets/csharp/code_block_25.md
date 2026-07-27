```csharp
public string[] AllLongestStrings(string[] inputArray) {
    // Pass 1: Find the maximum length
    int maxLength = 0;
    foreach (string s in inputArray) {
        if (s.Length > maxLength) {
            maxLength = s.Length;
        }
    }

    // Pass 2: Collect strings matching the max length
    List<string> result = new List<string>();
    foreach (string s in inputArray) {
        if (s.Length == maxLength) {
            result.Add(s);
        }
    }

    return result.ToArray();
}
// Time: O(N), Space: O(N) for output
```