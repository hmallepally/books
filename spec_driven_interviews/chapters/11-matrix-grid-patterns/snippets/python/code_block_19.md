```python
def num_islands(self, grid: list[list[str]]) -> int:
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                self._dfs(grid, i, j)
    return count

def _dfs(self, grid: list[list[str]], r: int, c: int) -> None:
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] == '0': return
    grid[r][c] = '0'
    self._dfs(grid, r+1, c); self._dfs(grid, r-1, c)
    self._dfs(grid, r, c+1); self._dfs(grid, r, c-1)
```