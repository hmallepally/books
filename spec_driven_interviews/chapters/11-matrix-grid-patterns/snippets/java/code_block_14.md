```java
public int[][] floodFill(int[][] image, int sr, int sc, int color) {
  if (image[sr][sc] != color) {
    dfs(image, sr, sc, image[sr][sc], color);
  }
  return image;
}
private void dfs(int[][] img, int r, int c, int oldC, int newC) {
  if (r < 0 || r >= img.length || c < 0 || c >= img[0].length || img[r][c] != oldC) return;
  img[r][c] = newC; // mark and fill
  dfs(img, r-1, c, oldC, newC);
  dfs(img, r+1, c, oldC, newC);
  dfs(img, r, c-1, oldC, newC);
  dfs(img, r, c+1, oldC, newC);
}
```
