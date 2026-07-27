```csharp
public bool AreOccurrencesEqual(string s) {
    if (string.IsNullOrEmpty(s)) return true;

    int[] counts = new int[128];
    foreach (char c in s) counts[c]++;

    int expected = 0;
    foreach (int count in counts) {
        if (count > 0) {
            if (expected == 0) expected = count;
            else if (count != expected) return false;
        }
    }

    return true;
}
// Time: O(N), Space: O(1)
```