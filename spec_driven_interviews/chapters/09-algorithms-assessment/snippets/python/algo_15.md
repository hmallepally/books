```python
def num_islands(grid: list[list[str]]) -> int:
    def dfs_sink(r: int, c: int) -> None:
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] == '0':
            return
        grid[r][c] = '0' # Sink cell
        dfs_sink(r + 1, c)
        dfs_sink(r - 1, c)
        dfs_sink(r, c + 1)
        dfs_sink(r, c - 1)

    count = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '1':
                count += 1
                dfs_sink(r, c)
                
    return count
```
