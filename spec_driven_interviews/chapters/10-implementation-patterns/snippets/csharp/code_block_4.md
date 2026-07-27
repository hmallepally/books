```csharp
public int Compress(char[] chars) {
    if (chars == null || chars.Length == 0) return 0;

    int write = 0; // Write pointer for compressed output
    int read = 0;  // Read pointer scanning input

    while (read < chars.Length) {
        char current = chars[read];
        int count = 0;

        // Count consecutive occurrences of current character
        while (read < chars.Length && chars[read] == current) {
            read++;
            count++;
        }

        // Write the character itself
        chars[write++] = current;

        // Write the count digits (only if count > 1)
        if (count > 1) {
            // Convert count to individual digit characters
            foreach (char digit in count.ToString()) {
                chars[write++] = digit;
            }
        }
    }

    return write;
}
// Time: O(N), Space: O(1) auxiliary
```