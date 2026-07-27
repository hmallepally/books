```csharp
int top = 0, bottom = matrix.Length - 1;
int left = 0, right = matrix[0].Length - 1;
while (top <= bottom && left <= right) {
  for (int j = left; j <= right; j++) { /* process matrix[top][j] */ }
  top++;
  for (int i = top; i <= bottom; i++) { /* process matrix[i][right] */ }
  right--;
  if (top <= bottom) {
    for (int j = right; j >= left; j--) { /* process matrix[bottom][j] */ }
    bottom--;
  }
  if (left <= right) {
    for (int i = bottom; i >= top; i--) { /* process matrix[i][left] */ }
    left++;
  }
}
```