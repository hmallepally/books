```csharp
public bool IsLucky(int n) {
    string s = n.ToString();
    int mid = s.Length / 2;
    int sum1 = 0, sum2 = 0;

    for (int i = 0; i < mid; i++) {
        sum1 += s[i] - '0';       // First half digit
        sum2 += s[i + mid] - '0'; // Second half digit
    }

    return sum1 == sum2;
}
// Time: O(D) where D is digit count, Space: O(D) for string conversion
```