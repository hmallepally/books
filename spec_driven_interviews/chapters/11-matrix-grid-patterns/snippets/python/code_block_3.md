```python
# Construction
sum_grid = [[0] * (C + 1) for _ in range(R + 1)]
for r in range(1, R + 1):
    for c in range(1, C + 1):
        sum_grid[r][c] = matrix[r-1][c-1] + sum_grid[r-1][c] + sum_grid[r][c-1] - sum_grid[r-1][c-1]

# Query from (r1, c1) to (r2, c2)
def query(r1: int, c1: int, r2: int, c2: int) -> int:
    return sum_grid[r2+1][c2+1] - sum_grid[r1][c2+1] - sum_grid[r2+1][c1] + sum_grid[r1][c1]
```