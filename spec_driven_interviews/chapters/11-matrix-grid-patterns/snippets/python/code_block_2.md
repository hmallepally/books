```python
dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]

def dfs(grid: list[list[int]], r: int, c: int) -> None:
    if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] == -1: return
    grid[r][c] = -1 # mark visited
    for i in range(4):
        dfs(grid, r + dr[i], c + dc[i])
```