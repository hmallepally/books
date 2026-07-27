```python
def pacific_atlantic(self, heights: list[list[int]]) -> list[list[int]]:
    m, n = len(heights), len(heights[0])
    pac, atl = [[False] * n for _ in range(m)], [[False] * n for _ in range(m)]
    
    for i in range(m):
        self._dfs_pa(heights, pac, i, 0)
        self._dfs_pa(heights, atl, i, n-1)
    for j in range(n):
        self._dfs_pa(heights, pac, 0, j)
        self._dfs_pa(heights, atl, m-1, j)
        
    res = []
    for i in range(m):
        for j in range(n):
            if pac[i][j] and atl[i][j]:
                res.append([i, j])
    return res

def _dfs_pa(self, h, v, r, c):
    v[r][c] = True
    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(h) and 0 <= nc < len(h[0]) and not v[nr][nc] and h[nr][nc] >= h[r][c]:
            self._dfs_pa(h, v, nr, nc)
```