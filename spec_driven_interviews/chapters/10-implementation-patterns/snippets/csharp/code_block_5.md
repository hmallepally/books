```csharp
public bool IsPalindrome(string s) {
    if (s == null) return false;

    int left = 0, right = s.Length - 1;

    while (left < right) {
        // Skip non-alphanumeric from the left
        while (left < right && !char.IsLetterOrDigit(s[left])) {
            left++;
        }
        // Skip non-alphanumeric from the right
        while (left < right && !char.IsLetterOrDigit(s[right])) {
            right--;
        }

        // Compare characters (case-insensitive)
        if (char.ToLower(s[left]) != char.ToLower(s[right])) {
            return false;
        }

        left++;
        right--;
    }

    return true;
}
// Time: O(N), Space: O(1)
```