```python
def flood_fill(self, image: list[list[int]], sr: int, sc: int, color: int) -> list[list[int]]:
    if image[sr][sc] != color:
        self._dfs(image, sr, sc, image[sr][sc], color)
    return image

def _dfs(self, img: list[list[int]], r: int, c: int, old_c: int, new_c: int) -> None:
    if r < 0 or r >= len(img) or c < 0 or c >= len(img[0]) or img[r][c] != old_c: return
    img[r][c] = new_c # mark and fill
    self._dfs(img, r-1, c, old_c, new_c)
    self._dfs(img, r+1, c, old_c, new_c)
    self._dfs(img, r, c-1, old_c, new_c)
    self._dfs(img, r, c+1, old_c, new_c)
```