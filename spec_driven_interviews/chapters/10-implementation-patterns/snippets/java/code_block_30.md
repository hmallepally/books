```java
public String[] addBorder(String[] picture) {
    int newWidth = picture[0].length() + 2;
    String[] result = new String[picture.length + 2];

    // Build the border row
    StringBuilder borderRow = new StringBuilder();
    for (int i = 0; i < newWidth; i++) borderRow.append('*');
    String border = borderRow.toString();

    // Top border
    result[0] = border;

    // Wrap each interior row with side asterisks
    for (int i = 0; i < picture.length; i++) {
        result[i + 1] = "*" + picture[i] + "*";
    }

    // Bottom border
    result[result.length - 1] = border;

    return result;
}
// Time: O(rows * cols), Space: O(rows * cols) for output
```
