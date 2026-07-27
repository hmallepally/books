```csharp
public char FindTheDifference(string s, string t) {
    char result = (char)0;
    foreach (char c in s) result ^= c;
    foreach (char c in t) result ^= c;
    return result; // Only the unpaired character survives
}
// Time: O(N), Space: O(1)
```