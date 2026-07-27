```csharp
public int FirstUniqueChar(string s)
{
    int[] counts = new int[256];
    foreach (char c in s) counts[c]++;
    for (int i = 0; i < s.Length; i++)
    {
        if (counts[s[i]] == 1) return i;
    }
    return -1;
}
```
