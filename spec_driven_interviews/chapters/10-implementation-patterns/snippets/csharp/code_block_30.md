```csharp
public string[] AddBorder(string[] picture) {
    int newWidth = picture[0].Length + 2;
    string[] result = new string[picture.Length + 2];

    // Build the border row
    string border = new string('*', newWidth);

    // Top border
    result[0] = border;

    // Wrap each interior row with side asterisks
    for (int i = 0; i < picture.Length; i++) {
        result[i + 1] = "*" + picture[i] + "*";
    }

    // Bottom border
    result[result.Length - 1] = border;

    return result;
}
// Time: O(rows * cols), Space: O(rows * cols) for output
```