```csharp
public int[][] FlipAndInvertImage(int[][] image) {
  foreach (int[] row in image) {
    int left = 0, right = row.Length - 1;
    while (left <= right) {
      int temp = row[left] ^ 1;
      row[left] = row[right] ^ 1;
      row[right] = temp;
      left++; right--;
    }
  }
  return image;
}
```