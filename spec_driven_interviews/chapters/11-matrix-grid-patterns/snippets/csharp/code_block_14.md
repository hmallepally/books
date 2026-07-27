```csharp
public int[][] FloodFill(int[][] image, int sr, int sc, int color) {
  if (image[sr][sc] != color) {
    Dfs(image, sr, sc, image[sr][sc], color);
  }
  return image;
}
private void Dfs(int[][] img, int r, int c, int oldC, int newC) {
  if (r < 0 || r >= img.Length || c < 0 || c >= img[0].Length || img[r][c] != oldC) return;
  img[r][c] = newC; // mark and fill
  Dfs(img, r-1, c, oldC, newC);
  Dfs(img, r+1, c, oldC, newC);
  Dfs(img, r, c-1, oldC, newC);
  Dfs(img, r, c+1, oldC, newC);
}
```