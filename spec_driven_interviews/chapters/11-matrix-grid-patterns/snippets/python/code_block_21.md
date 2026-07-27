```python
def shift_grid(self, grid: list[list[int]], k: int) -> list[list[int]]:
    m, n = len(grid), len(grid[0])
    total = m * n
    k %= total
    res = [[0] * n for _ in range(m)]
    
    for r in range(m):
        for c in range(n):
            new_1d = (r * n + c + k) % total
            res[new_1d // n][new_1d % n] = grid[r][c]
    return res
```